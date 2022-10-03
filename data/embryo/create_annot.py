import sys 
import os 
import os.path as osp 
import pandas as pd 
import numpy as np 
import cv2 
import json
from tqdm import tqdm

DATA_NAME = sys.argv[1]
CSV_PATH = sys.argv[2]
IMG_DIR = sys.argv[3]
FEAT_SAVE_DIR = sys.argv[4]
save_path = sys.argv[5]

need_convert_stage, stage_map = False, None
if DATA_NAME  == 'human':
    need_convert_stage = False 

def convert_stage(str_stage: str):
    if stage_map.get(str_stage) is None:
        return '' 
    return int(stage_map[str_stage]) 

def create_annot_file(data_df, save_path):
    feat_dir = FEAT_SAVE_DIR
    list_videos = data_df['Folder'].unique().tolist()
    
    print(f'n videos: {len(list_videos)}')
    result = {}
    for vid in tqdm(list_videos):
        vid_id = vid
        vid_df = data_df[data_df['Folder'] == vid]
        list_stages = vid_df['EmbryoStage'].values.tolist() 
        
        N = len(list_stages)
        N_classes = len(set(list_stages))
    
        timestamps = [] 
        first_stamp = [0]
        next_stamp = []
        for cur in range(N):
            if cur == N-1:
                next_stamp.append(cur)
                timestamps.append(next_stamp)
                break
            if list_stages[cur+1] != list_stages[cur]:
                if len(first_stamp) != 2:
                    first_stamp.append(cur)
                    timestamps.append(first_stamp)
                    next_stamp.append(cur+1)
                else:
                    next_stamp.append(cur)
                    timestamps.append(next_stamp)
                    next_stamp = [cur+1]

        action_labels = []
        for stamp in timestamps:
            if list_stages[stamp[0]] not in action_labels:
                action_labels.append(list_stages[stamp[0]])

        vid_feat_path = osp.join(feat_dir, f'{vid}.npy')

        vid_feat = np.load(vid_feat_path)
        feat_len = vid_feat.shape[0]
        if feat_len != N:
            print(f'Fail vid_feat {vid}, feat_len: {feat_len}, N: {N}')
            continue
        
        frame_labels = vid_df['EmbryoStage'].values.tolist()
        list_fpaths = vid_df['FullFile'].values.tolist()
        list_full_fpaths = []
        for fpath in list_fpaths:
            if not osp.isfile(fpath):
                list_full_fpaths.append(osp.join(IMG_DIR, fpath))
            else:
                list_full_fpaths.append(fpath)

        result[vid_id] = {
            'duration': N, 'timestamps': timestamps, 'action_labels': action_labels,
            'feat_path': vid_feat_path, 
            'feat_shape': vid_feat.shape, 'frame_labels': frame_labels,
            'frame_paths': list_full_fpaths
        }
    
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)

def main():
    data_df = pd.read_csv(CSV_PATH)
    create_annot_file(data_df, save_path)

if __name__ == '__main__':
    main()