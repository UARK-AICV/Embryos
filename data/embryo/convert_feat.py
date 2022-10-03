import os 
import os.path as osp
import torch 
import numpy as np
import json 
from tqdm import tqdm
import sys 

FEAT_SAVE_DIR = sys.argv[1]
feat_dir = FEAT_SAVE_DIR

count = 0
video_dict = {}
for fname in tqdm(os.listdir(FEAT_SAVE_DIR)):
    if '.pth' not in fname:
        continue
    fpath = osp.join(FEAT_SAVE_DIR, fname)
    vid_feat = torch.load(fpath)
    if count == 0:
        print(f'fpath: {fpath.split("/")[-1]}, shape: {vid_feat.shape}')
    
    np_vid_feat = vid_feat.detach().cpu().numpy()
    save_path = osp.join(feat_dir, fpath.replace('.pth', '.npy'))
    np.save(save_path, np_vid_feat)


