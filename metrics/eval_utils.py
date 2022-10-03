import os 
import os.path as osp
import numpy as np
import torch 

from metrics.classification import Accuracy, Recall, Precision, ConfusionMatrix
from metrics.meter import AverageMeter

# from data.video_dataset import PropSeqDataset, collate_fn
from data.dataset import PropSeqDataset, collate_fn
from misc.detr_utils import box_ops

# ----------------------------------
def get_box_iou_with_classes(outputs, targets, num_classes=4):
    out_boxes = outputs['pred_boxes'].detach().squeeze()
    target_boxes = torch.cat([t['boxes'] for t in targets])
    
    # import pdb; pdb.set_trace()
    out_boxes = box_ops.box_cl_to_xy(out_boxes)
    target_boxes = box_ops.box_cl_to_xy(target_boxes)

    giou_map, iou_map = box_ops.generalized_box_iou(
        box_ops.box_cl_to_xy(out_boxes),
        box_ops.box_cl_to_xy(target_boxes)
    )
    iou, union = box_ops.box_iou(out_boxes, target_boxes)
    inter = iou * (union + 1e-5)
    out_area = out_boxes[:, 1] - out_boxes[:, 0]
    target_area = target_boxes[:, 1] - target_boxes[:, 0]
    prec = inter / target_area
    rec = inter / out_area
    # import pdb; pdb.set_trace()

    result = {
        'giou': {}, 'iou': {}, 'prec': {}, 'rec': {}
    }
    for c in range(num_classes):
        result['giou'][c] = giou_map[c][c].item()
        result['iou'][c] = iou_map[c][c].item()
        result['prec'][c] = prec[c][c].item()
        result['rec'][c] = rec[c][c].item()
    
    return result
    
def boxes_to_prediction_list(boxes, scores, target_sizes):
    B = target_sizes.shape[0]
    n_class = scores.shape[1]
    batch_preds = []
    
    for b in range(B):
        L = int(target_sizes[b].item())
        preds = np.zeros((n_class, L))
        for class_id, (start, end) in enumerate(boxes[b]):
            start = int(start)
            end = int(end)
            pred_class_score = scores[b][class_id]
            preds[class_id, start: end+1] = pred_class_score
        
        preds = np.argmax(preds, axis=0)
        batch_preds.append(torch.Tensor(preds))
    
    res = torch.stack(batch_preds)
    return res

def convert_box_predictions(outputs, target_sizes):
    """
    Args:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
    """
    out_logits = outputs['pred_logits'].detach()
    out_bbox = outputs['pred_boxes'].detach()
    N, N_q, N_class = out_logits.shape
    prob = out_logits.sigmoid()
    id_mask = torch.eye(N_class, N_class).to(prob.device)
    scores = torch.sum(prob*id_mask, dim=-1) 

    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1)
    labels = topk_indexes % N_class
    boxes = box_ops.box_cl_to_xy(out_bbox)
    boxes[boxes < 0] = 0
    boxes[boxes > 1] = 1

    scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    return boxes, scores, labels

def reset_segment_metrics(segment_metrics):
    for mode in segment_metrics:
        for c in segment_metrics[mode]:
            segment_metrics[mode][c].reset()
    
def update_segment_metrics(segment_metrics, segment_score):
    for mode in segment_metrics:
        for c in segment_metrics[mode]:
            segment_metrics[mode][c].update(segment_score[mode][c], 1)

def get_precision_recall_by_class(confusion_matrix):
    precision_mat = confusion_matrix/confusion_matrix.sum(axis=1)
    recall_mat = confusion_matrix/confusion_matrix.sum(axis=0)

    precision_mat = confusion_matrix.diagonal()/confusion_matrix.sum(axis=1)
    recall_mat = confusion_matrix.diagonal()/confusion_matrix.sum(axis=0)
    return precision_mat, recall_mat

