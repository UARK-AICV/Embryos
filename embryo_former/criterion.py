# ------------------------------------------------------------------------
# Modified from PDVC(https://github.com/ttengwang/PDVC)
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from multiprocessing import reduction
from operator import is_
import torch
import torch.nn.functional as F
from torch import nn

from misc.detr_utils import box_ops
from misc.detr_utils.misc import (accuracy, get_world_size, is_dist_avail_and_initialized, inverse_sigmoid)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict, losses, aux_losses, opt={}):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            aux_losses: only use these losses for aux
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.aux_losses = aux_losses
        self.opt = opt
        self.frame_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.num_classes)
        self.query_class = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.num_classes)

        self.training=True

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        indices, many2one_indices = indices
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # B, Nq, Num_classes (Nq = n_classes = 4/8)
        
        target_classes = torch.stack([t["labels"] for t in targets])
        loss_ce = self.query_class(src_logits, target_classes)
        losses = {'loss_ce': loss_ce}

        return losses


    # ---------- Frame Refine Prediction ------------
    def get_frame_refine_prediction(self, outputs, targets, pos_weight=None, is_training=True):
        frame_logits = outputs['refine_weight_attn']
        B, L, C = frame_logits.shape
        
        pred_width = None 
        if is_training:
            pred_width = torch.stack([t['boxes_width'] for t in targets])
        else:
            pred_width = outputs['pred_width'].squeeze()

        if B == 1:
            pred_width = pred_width.unsqueeze(0)

        if pos_weight is None:
            ref_pos_weights = []
            for t in targets:
                t_len = t['frame_labels'].shape[0]                
                pad_weight = torch.ones(int(L - t_len)).to(t['frame_labels'].device)
                pos_weight = (torch.arange(int(t_len)) + 1)/t_len
                pos_weight = pos_weight.to(t['frame_labels'].device)
                pos_weight = torch.cat([pos_weight, pad_weight])
                ref_pos_weights.append(pos_weight)
            
            pos_weight = torch.stack(ref_pos_weights)

        pos_weight = pos_weight.unsqueeze(-1).expand(B, L, C)

        if B > 1:
            pred_center  = torch.cumsum(pred_width, 1) - pred_width/2
        else:
            pred_center = torch.cumsum(pred_width, 0) - pred_width/2

        flat_pred_center = pred_center.unsqueeze(1).expand(B, L, C)
        flat_width_center = pred_width.unsqueeze(1).expand(B, L, C)
        
        TEMP = 1e-2
        frame_logits = frame_logits*((flat_width_center+TEMP)/(torch.abs(flat_pred_center - pos_weight) + TEMP))

        return frame_logits

    def loss_frame_labels(self, outputs, targets, indices, num_boxes):
        frame_logits = outputs['pred_frames']
        B, L, C = frame_logits.shape
        list_lens = [t['frame_labels'].shape[0] for t in targets]
        frame_gts = []
        for t in targets:
            t_gts = t['frame_labels']
            t_len = t['frame_labels'].shape[0]
            pad_labels = torch.tensor([self.num_classes]*(L - t_len)).to(t['frame_labels'].device)
            t_gts = torch.cat([t['frame_labels'], pad_labels])
            frame_gts.append(t_gts)
        frame_gts = torch.cat(frame_gts)

        if len(frame_gts.shape) == 1:
            frame_gts = frame_gts.unsqueeze(0)
        
        frame_logits = frame_logits.view(B*L, C)
        frame_gts = frame_gts.view(frame_logits.shape[0]).type(torch.LongTensor).to(frame_logits.device)
        loss = self.frame_criterion(frame_logits, frame_gts)

        # Frame refine:
        if outputs.get('pred_frames_refine') is not None:
            collab_logits = outputs['pred_frames_refine'].view(B*L, C)
            loss_collab = self.frame_criterion(collab_logits, frame_gts)

            return {'loss_frame': loss, 'loss_refine_frame': loss_collab}

        return {'loss_frame': loss}


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 2]
           The target boxes are expected in format (center, length), normalized by the image size.
        """
        indices, many2one_indices = indices
        N = len(indices[-1][0])
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        target_boxes = torch.stack([t['boxes'] for t in targets])

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cl_to_xy(src_boxes),
            box_ops.box_cl_to_xy(target_boxes)
        )
        loss_giou = 1 - torch.diag(giou)
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        self_iou = torch.triu(box_ops.box_iou(box_ops.box_cl_to_xy(src_boxes), box_ops.box_cl_to_xy(src_boxes))[0], diagonal=1)
        sizes = [len(v[0]) for v in indices]
        self_iou_split = 0
        for i, c in enumerate(self_iou.split(sizes, -1)):
            cc = c.split(sizes, -2)[i]
            self_iou_split += cc.sum() / (0.5 * (sizes[i]) * (sizes[i]-1))
        losses['loss_self_iou'] = self_iou_split

        return losses

    def loss_boxes_center(self, outputs, targets, indices, num_boxes):
        indices, many2one_indices = indices
        
        target_boxes = torch.stack([t['boxes'] for  t in targets])
        target_center, target_width = torch.split(target_boxes, [1, 1], 2)
        target_center = target_center.squeeze() # B, Nq

        # Use center from pred width
        B, Nq, dim_w = outputs['pred_width'].shape
        src_width = outputs['pred_width'].squeeze()
        out_center = outputs['pred_center'].squeeze()
        
        if B > 1:
            src_center = torch.cumsum(src_width, 1) - src_width/2
        else:
            src_center = torch.cumsum(src_width, 0) - src_width/2

        losses = {}
        loss_center = F.l1_loss(src_center, target_center.squeeze(), reduction='none')
        loss_center_deform = F.l1_loss(out_center, target_center.squeeze(), reduction='none')
        
        losses['loss_center'] = loss_center.sum()/num_boxes + loss_center_deform.sum()/num_boxes
        return losses
        
    def loss_boxes_width(self, outputs, targets, indices, num_boxes):
        assert 'pred_width' in outputs

        src_width = outputs['pred_width'].squeeze()
        target_width = torch.stack([t['boxes_width'] for t in targets])

        if len(src_width.shape) == 1:
            src_width = src_width.unsqueeze(0)
        
        loss_bbox = F.l1_loss(src_width, target_width, reduction='mean')

        losses = {}
        losses['loss_width'] = loss_bbox 
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'frames': self.loss_frame_labels,
            'width': self.loss_boxes_width,
            'center': self.loss_boxes_center,
            'boxes': self.loss_boxes,
            'labels': self.loss_labels
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                      >> dt['video_target']
        """
        num_boxes_int = sum(len(t["labels"]) for t in targets)
        
        # Retrieve the matching between the outputs of the last layer and the targets
        
        assign_ids = list(range(num_boxes_int))
        last_indices = (
            [(torch.tensor(assign_ids), torch.tensor(assign_ids))],
            None
        )
        outputs['matched_indices'] = last_indices
        
        num_boxes = torch.as_tensor([num_boxes_int], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, last_indices, num_boxes, **kwargs))

        if 'aux_outputs' in outputs:
            aux_indices = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                assign_ids = list(range(num_boxes_int))
                indices = (
                    [(torch.tensor(assign_ids), torch.tensor(assign_ids))],
                    None
                )
                aux_indices.append(indices)

                for loss in self.aux_losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            return losses, last_indices, aux_indices
        return losses, last_indices
