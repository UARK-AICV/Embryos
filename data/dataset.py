from collections import defaultdict
from itertools import chain
import sys
import json
import os
import scipy
import numpy as np
import random
import pickle
import pandas as pd
import cv2
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset, DataLoader

from data.transform import get_train_transform, get_val_transform

def collate_fn(batch):
    batch_size = len(batch)
    feature_size = batch[0][0].shape[1]

    feature_list, gt_timestamps_list, labels, frame_labels, list_frames, gt_raw_timestamp, raw_duration, key = zip(*batch)
    # C, H, W = list_frames[0][0].shape
    max_video_length = max([x.shape[0] for x in feature_list])

    gt_timestamps = list(chain(*gt_timestamps_list))

    video_tensor = torch.FloatTensor(batch_size, max_video_length, feature_size).zero_()
    video_pos_weight = torch.zeros(batch_size, max_video_length, dtype=torch.float32) # Create position weight for collaborative head
    video_length = torch.FloatTensor(batch_size, 3).zero_()  # true length, sequence length
    video_mask = torch.BoolTensor(batch_size, max_video_length).zero_()

    max_caption_num = len(labels[0])

    # lnt_boxes_tensor = torch.zeros(batch_size, max_proposal_num, 4)
    gt_boxes_tensor = torch.zeros(batch_size, max_caption_num, 2)
    gt_boxWidth_tensor = torch.zeros(batch_size, max_caption_num)

    total_caption_idx = 0
    total_proposal_idx = 0

    for idx in range(batch_size):
        video_len = feature_list[idx].shape[0]
        # proposal_length = len(timestamps_list[idx])
        gt_proposal_length = len(gt_timestamps_list[idx]) # n snippets in video[idx]
        
        video_tensor[idx, :video_len, :] = torch.from_numpy(feature_list[idx])
        video_pos_weight[idx, :video_len] = (torch.arange(int(video_len)) + 1)/video_len

        video_length[idx, 0] = float(video_len)
        video_length[idx, 1] = raw_duration[idx]
        video_length[idx, 2] = gt_proposal_length
        video_mask[idx, :video_len] = True

        box_info_tensor = torch.tensor(
            [[(ts[1] + ts[0]) / (2 * raw_duration[idx]), (ts[1] - ts[0]+1) / raw_duration[idx]] for ts in gt_raw_timestamp[idx]]).float()
        gt_boxes_tensor[idx, :gt_proposal_length] = box_info_tensor
        gt_boxWidth_tensor[idx, :gt_proposal_length] = torch.tensor([(ts[1] - ts[0+1]) / raw_duration[idx] for ts in gt_raw_timestamp[idx]])
        total_caption_idx += gt_proposal_length

    gt_boxes_mask = (gt_boxes_tensor != 0).sum(2) > 0

    target = [
        {
            'boxes': torch.tensor(
                [[(ts[1] + ts[0]) / (2 * raw_duration[i]), (ts[1] - ts[0]+1) / raw_duration[i]] for ts in gt_raw_timestamp[i]]
            ).float(),
            'labels': torch.tensor(labels[i]).long(),
            'frame_labels': torch.tensor(frame_labels[i]),
            'masks': None,
            'image_id': vid,
            'boxes_width': torch.tensor([(ts[1] - ts[0]+1) / raw_duration[i] for ts in gt_raw_timestamp[i]]),
        } for i, vid in enumerate(list(key))
    ] 

    dt = {
        "video":
            {
                "tensor": video_tensor,  # tensor,      (video_num, video_len, video_dim)
                "length": video_length,
                "mask": video_mask,  # tensor,      (video_num, video_len,)
                "key": list(key),  # list,        (video_num)
                "target": target,
                "duration": raw_duration,
                "pos_weight": video_pos_weight,
            },
        "gt":
            {
                "featstamps": gt_timestamps,  # list,        (gt_all_event_num, 2)
                "timestamp": list(gt_raw_timestamp),  # list (len: video_num) of tensors (shape: (gt_event_num, 2))
                # "gather_idx": caption_gather_idx,  # tensor,      (gt_all_event_num)
                "boxes": gt_boxes_tensor,
                "boxes_mask": gt_boxes_mask,
                "boxes_width": gt_boxWidth_tensor
            },
    }
    dt = {k1 + '_' + k2: v2 for k1, v1 in dt.items() for k2, v2 in v1.items()}
    
    return dt


class StageDataset(Dataset):

    def __init__(self, anno_file, is_training, opt):

        super(StageDataset, self).__init__()
        self.anno = json.load(open(anno_file, 'r'))
        self.keys = list(self.anno.keys())
        self.opt = opt
        self.is_training = is_training
        self.feature_dim = self.opt.feature_dim
        self.num_queries = opt.num_queries

        if is_training:
            self.frame_transform = get_train_transform()
        else:
            self.frame_transform = get_val_transform()

    def __len__(self):
        return len(self.keys)

    def process_time_step(self, duration, timestamps_list, feature_length):
        duration = np.array(duration)
        timestamps = np.array(timestamps_list)
        feature_length = np.array(feature_length)
        featstamps = feature_length * timestamps / duration
        featstamps = np.minimum(featstamps, feature_length - 1).astype('int')
        featstamps = np.maximum(featstamps, 0).astype('int')
        return featstamps.tolist()

    def __getitem__(self, idx):
        raise NotImplementedError()


class PropSeqDataset(StageDataset):
    def __init__(self, anno_file, is_training, opt):
        super(PropSeqDataset, self).__init__(anno_file, is_training, opt)

    def load_feats(self, key):
        vid_feat = np.load(self.anno[key]['feat_path'])
        return vid_feat
        
    def random_split(self, time_stamps, duration):
        random_ids = []
        new_timestamps = []
        min_cut = int(duration*0.1)

        for stamp in time_stamps:
            start, end = stamp 
            stamp_len = end-start +1
            stamp_ids = list(range(start, end+1, 1))
            
            if stamp_len < min_cut:
                choose_ids = stamp_ids
            else:
                choose_ids = np.random.choice(
                    stamp_ids, random.randrange(min_cut, len(stamp_ids)+1), replace=False
                )
            
            random_ids.extend(choose_ids)
            if len(new_timestamps) == 0:
                new_timestamps.append([0, len(choose_ids)-1])
            else:
                new_timestamps.append([new_timestamps[-1][1]+1, new_timestamps[-1][1] + len(choose_ids)])

        random_ids = np.sort(random_ids)
        assert new_timestamps[-1][1] == len(random_ids)-1, 'Wrong split'
        
        return random_ids, new_timestamps

    def get_frame_position_weights(self):
        pass 

    def __getitem__(self, idx):
        key = str(self.keys[idx])
        feats = self.load_feats(key) # [L, d]
        duration = self.anno[key]['duration']
        gt_timestamps = self.anno[key]['timestamps']  # [gt_num, 2], e.g: [[0, 0.5]], len = num_stages
        
        action_labels = self.anno[key].get('action_labels', [0] * len(gt_timestamps)) # ==> stage label
        frame_labels = self.anno[key]['frame_labels'] # len = L
        frame_paths = self.anno[key]['frame_paths'] # len = L
        L = len(frame_paths)

        list_imgs = []
        assert max(action_labels) <= self.opt.num_classes

        random_ids = None
        if self.is_training and self.opt.random_split:
            random_ids, gt_timestamps = self.random_split(gt_timestamps, duration)
        else:
            random_ids = list(range(duration))

        frame_labels = [frame_labels[i] for i in random_ids]
        frame_paths = [frame_paths[i] for i in random_ids]
        feats = np.stack([feats[i] for i in random_ids], axis=0)
        duration = L = len(frame_labels)

        gt_featstamps = self.process_time_step(duration, gt_timestamps, L)
        return feats, gt_featstamps, action_labels, frame_labels, list_imgs, gt_timestamps, duration, key


# ------------------------------------ UTILS ------------------------------------
def load_img(fpath, color, target_size, interpolation=cv2.INTER_LINEAR):
    cv_img = cv2.imread(fpath)
    if cv_img is None:
        print(fpath)
    cv_img = cv2.resize(cv_img, dsize=target_size, interpolation=interpolation)
    
    if color.upper() =='gray':
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    if color.upper() =='RGB':
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    return cv_img

def iou(interval_1, interval_2):
    interval_1, interval_2 = map(np.array, (interval_1, interval_2))
    start, end = interval_2[None, :, 0], interval_2[None, :, 1]
    start_i, end_i = interval_1[:, None, 0], interval_1[:, None, 1]
    intersection = np.minimum(end, end_i) - np.maximum(start, start_i)
    union = np.minimum(np.maximum(end, end_i) - np.minimum(start, start_i), end - start + end_i - start_i)
    iou = intersection.clip(0) / (union + 1e-8)
    return iou


