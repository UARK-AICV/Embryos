from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import argparse
import json
import os
import sys
import os.path as osp
from os.path import dirname, abspath
import collections 
from collections import OrderedDict
from tqdm import tqdm
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)

from embryo_former.model import build # Import model
from metrics.meter import AverageMeter

from data.dataset import PropSeqDataset, collate_fn
from misc.detr_utils import box_ops
from utils.writer import print_metrics_frames
from utils.logger import create_logger
from metrics.evaluator import (
    create_dict_metrics
)
from utils.logger import create_logger


def init_args(opt, config_path):
    with open(config_path, 'r') as f:
        load_opt = json.load(f)
        for k, v in load_opt.items():
            vars(opt).update({k:v})

    return opt


def inference(model, data_annot, criterion, save_dir, logger):
    
    vis_dir = osp.join(save_dir, 'vis_test')
    os.makedirs(vis_dir, exist_ok=True)

    criterion.training=False
    opt.batch_size = 1
    val_dataset = PropSeqDataset(data_annot, False, opt)
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    refine_frame_metrics = create_dict_metrics(opt.num_classes)
    width_metrics = create_dict_metrics(opt.num_classes)
    
    loss_manager = {}
    for loss in criterion.weight_dict:
        loss_manager[loss] = AverageMeter()

    predictions = {}
    for batch_id, dt in tqdm(enumerate(val_loader)):
        dt = {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
        dt['video_target'] = [
            {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
            dt['video_target']]
        dt = collections.defaultdict(lambda: None, dt)
        orig_target_sizes = dt['video_length'][:, 1]
        list_keys = dt['video_key']
        for k in list_keys:
            predictions[k] = {}

        # Model inference
        with torch.no_grad():
            output, loss = model(dt, criterion, eval_mode=True)

        for k in loss:
            loss_manager[k].update(loss[k].item())

        frame_labels = [sample['frame_labels'] for sample in dt['video_target']]
        frame_preds = output['pred_frames']
        frame_refine_preds = output['pred_frames_refine']
        
        B, L, C = frame_preds.shape
        
        # Evaluate stage-area width prediction
        out_width = output['pred_width'].detach().cpu().numpy()
        width_boxes = box_ops.box_width_to_xy(out_width, orig_target_sizes)
        width_frame_preds = box_ops.convert_to_frame_pred(width_boxes, orig_target_sizes)
        for metric in width_metrics:
            for b_id in range(B):
                vid_gts = frame_labels[b_id]
                vid_preds = width_frame_preds[b_id].to(vid_gts.device)
                width_metrics[metric].update(vid_preds, vid_gts, is_prob=False)

                predictions[list_keys[b_id]]['seq_len'] = int(orig_target_sizes[b_id])
                predictions[list_keys[b_id]]['frame_labels'] = vid_gts.cpu().squeeze().numpy()
                predictions[list_keys[b_id]]['width_preds'] = vid_preds.detach().cpu().squeeze().numpy()
                predictions[list_keys[b_id]]['width_boxes'] = width_boxes[b_id].cpu().squeeze().numpy()


        # Evaluate Refine Frame classification
        for metric in refine_frame_metrics:
            for b_id in range(B):
                vid_len = int(orig_target_sizes[b_id])
                vid_preds = frame_refine_preds[b_id][:int(orig_target_sizes[b_id])]
                vid_gts = frame_labels[b_id]
                refine_frame_metrics[metric].update(vid_preds, vid_gts, is_prob=True)

                predictions[list_keys[b_id]]['refine_preds'] = vid_preds.detach().cpu().squeeze().numpy()

       
    print_metrics_frames(
        metric_dict={
            'Refine': refine_frame_metrics, 
        }, 
        n_classes = opt.num_classes,
        logger = logger, summary=True
    )
    with open(osp.join(save_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
    pass 

def create_model(ckpt_path, opt, logger=None):
    # Create model
    model, criterion = build(opt)
    # model.criterion = criterion
    model = model.to(opt.device)

    ckpt_data = torch.load(ckpt_path)
    model_weight = ckpt_data['model']
    model.load_state_dict(model_weight)
    model.eval() 

    print_func = print
    if logger is not None:
        print_func = logger.info
    
    print_func(f"Loaded weights from {ckpt_data['epoch']}, best val acc: {ckpt_data['best_acc']}")
    return model, criterion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--focus_head', type=str, default='frame')
    parser.add_argument('--trained_dir', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    
    opt = parser.parse_args()
    save_dir = osp.join(opt.trained_dir, f'Validation-{opt.focus_head}')

    os.makedirs(save_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    logger = create_logger(osp.join(save_dir, 'log_test.txt'))

    CONFIG_PATH = osp.join(opt.trained_dir, 'config.json')
    opt = init_args(opt, CONFIG_PATH)
    
    ckpt_path = osp.join(opt.trained_dir, f'model_best_{opt.focus_head}.pth')
    
    model, criterion = create_model(ckpt_path, opt, logger)
    print(f'Create model successfully, weight: {ckpt_path}')

    inference(model, opt.test_annot, criterion, save_dir, logger)
    
