import os 
import os.path as osp
import numpy as np
import torch 

from metrics.classification import Accuracy, Recall, Precision, ConfusionMatrix, F1
from metrics.meter import AverageMeter

# from data.video_dataset import PropSeqDataset, collate_fn
from data.dataset import PropSeqDataset, collate_fn
from misc.detr_utils import box_ops

# ----------------------------------
def get_box_iou_with_classes(outputs, targets, num_classes=4):
    out_boxes = outputs['pred_boxes'].detach() # [B, Nq, 2]
    target_boxes = torch.stack([t['boxes'] for t in targets]) # [B*Nq, 2]
    bs, num_queries = out_boxes.shape[:2]
    
    # src_boxes = out_boxes.flatten(0, 1) # [B*Nq, 2]

    result = {
        'giou': {}, 'iou': {}, 'prec': {}, 'rec': {}
    }
    for c in range(num_classes):
        result['giou'][c] = [] #giou_map[c][c].item()
        result['iou'][c] = [] #iou_map[c][c].item()
        result['prec'][c] = [] #prec[c][c].item()
        result['rec'][c] = [] #rec[c][c].item()
    
    for i in range(bs):
        src_boxes = out_boxes[i]
        tgt_boxes = target_boxes[i]
        src_boxes = box_ops.box_cl_to_xy(src_boxes)
        tgt_boxes = box_ops.box_cl_to_xy(tgt_boxes)

        giou_map, iou_map = box_ops.generalized_box_iou(src_boxes,tgt_boxes)    
        iou, union = box_ops.box_iou(src_boxes, tgt_boxes)
        
        inter = iou * (union + 1e-5)
        src_area = src_boxes[:, 1] - src_boxes[:, 0]
        tgt_area = tgt_boxes[:, 1] - tgt_boxes[:, 0]
        prec = inter / tgt_area
        rec = inter / src_area
        
        for c in range(num_classes):    
            result['giou'][c].append(giou_map[c][c].item()) 
            result['iou'][c].append(iou_map[c][c].item()) 
            result['prec'][c].append(prec[c][c].item()) 
            result['rec'][c].append(rec[c][c].item()) 

    
    for c in range(num_classes):
        result['giou'][c] = np.mean(result['giou'][c])
        result['iou'][c] = np.mean(result['iou'][c])
        result['prec'][c] = np.mean(result['prec'][c])
        result['rec'][c] = np.mean(result['rec'][c])
    
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
    
def update_segment_metrics(segment_metrics, segment_score, n=1):
    for mode in segment_metrics:
        for c in segment_metrics[mode]:
            segment_metrics[mode][c].update(segment_score[mode][c], n)

def get_precision_recall_by_class(confusion_matrix):
    recall_mat = confusion_matrix/confusion_matrix.sum(axis=1)
    precision_mat = confusion_matrix/confusion_matrix.sum(axis=0)
    return precision_mat, recall_mat

def get_accuracy_per_class(confusion_matrix):
    per_class_accuracies = {}
    classes = confusion_matrix.shape[0]
    classes = list(range(classes))
    
    # Calculate the accuracy for each one of our classes
    for idx, cls in enumerate(classes):
        # True negatives are all the samples that are not our current GT class (not the current row) 
        # and were not predicted as the current class (not the current column)
        true_negatives = np.sum(np.delete(np.delete(confusion_matrix, idx, axis=0), idx, axis=1))
        
        # True positives are all the samples of our current GT class that were predicted as such
        true_positives = confusion_matrix[idx, idx]
        
        # The accuracy for the current class is ratio between correct predictions to all predictions
        per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(confusion_matrix)
    
    return per_class_accuracies


from metrics.classification import Accuracy, Recall, Precision, ConfusionMatrix
def create_dict_metrics(num_classes: int=4, save_dir: str=None):
    return {
        'accuracy': Accuracy(nclasses=num_classes), 
        'precision': Precision(nclasses=num_classes), 
        'recall': Recall(nclasses=num_classes),
        'f1': F1(nclasses=num_classes),
        'confusion_matrix': ConfusionMatrix(nclasses=num_classes, savefig_dir=save_dir)
    }