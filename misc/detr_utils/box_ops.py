# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import numpy as np
import math
import torch
from torchvision.ops.boxes import box_area

def box_width_to_xy(x, orig_target_sizes):
    w = x.squeeze(-1)
    N_boxes = w.shape[1] # = N_class
    
    batch_boxes = []
    for i, width in enumerate(x):
        list_boxes =[ [0, width[0].item()] ] 
        for i in range(1, N_boxes):
            cur_right = list_boxes[-1][1] + width[i].item()
            list_boxes.append([list_boxes[-1][1], min(cur_right, 1)])
        list_boxes = torch.tensor(list_boxes)
        batch_boxes.append(list_boxes)
    batch_boxes = torch.stack(batch_boxes)

    scale_fct = torch.stack([orig_target_sizes, orig_target_sizes], dim=1).cpu()
    batch_boxes = batch_boxes * scale_fct[:, None, :]
    return batch_boxes

def convert_to_frame_pred(pred_boxes, batch_lens):
    """
    Args:
        pred_boxes (_type_): converted boxes from  output['pred_width'], 
                                use the above box_width_to_xy function
    Returns:
        _type_: _description_
    """
    B, N_classes, _ = pred_boxes.shape
    batch_preds = []
    for i in range(B):
        L = int(batch_lens[i].item())
        vid_preds = np.zeros(L)
        for class_id, box in enumerate(pred_boxes[i]):
            vid_preds[math.floor(box[0]): math.ceil(box[1])] = class_id
        batch_preds.append(torch.tensor(vid_preds))
    
    return batch_preds


# ---- AUTHOR ------
def box_cl_to_xy(x):
    c, l = x.unbind(-1)
    b = [c - 0.5 * l, c + 0.5 * l]
    return torch.stack(b, dim=-1)

def box_xy_to_cl(x):
    x0, x1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (x1 - x0)]
    return torch.stack(b, dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = boxes1[:, 1] - boxes1[:, 0]
    area2 = boxes2[:, 1] - boxes2[:, 0]
    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M,2]
    inter = (rb - lt).clamp(min=0)  # [N,M,2]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-5)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 1:] >= boxes1[:, :1]).all()
    assert (boxes2[:, 1:] >= boxes2[:, :1]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    area = (rb - lt).clamp(min=0)  # [N,M,2]
    giou = iou - (area - union) / (area + 1e-5)
    return giou, iou

def batch_box_giou(batch_boxes1, batch_boxes2):
    # batch_boxes shape: [B, N_box, 2]
    
    pass