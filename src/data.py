import os
import json
import torch
import einops
import random
import os.path
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from vidtransforms import *
from einops import rearrange
from torchvision import transforms
from torch.utils.data import Dataset


class readFeatureHMDB51(Dataset):
    def __init__(self, root: str, frames: int, fpsR: list, ensemble: int, mode: str, sampling_fps=None, fixed_fps=None):
        self.root = root
        self.mode = mode
        self.fpsR = fpsR
        self.frames = frames
        self.ensemble = ensemble
        self.name_map = json.load(open('../data/HMDB51/HMDB51_clsname.json'))
        self.sampling_fps=sampling_fps
        self.fixed_fps = fixed_fps

        if mode in ['test', 'val']:
            mode_ = 'test'
        elif mode == 'train':
            mode_ = 'train'
        split_file = os.path.join('../data/HMDB51', '{}_split01.txt'.format(mode_))
        split_info = self.read_txt(split_file)

        self.video_class_info = [self.root + vid[:-4] + 'npy' for _ in range(self.ensemble) for vid in split_info]
        self.video_path = [self.root + vid.split('/')[1][:-4] + 'npy' for _ in range(self.ensemble) for vid in split_info]
    
    def __len__(self):
        return len(self.video_path)

    def read_txt(self, filepath):
        file = open(filepath)
        txtdata = file.readlines()
        file.close()
        return txtdata

    def fill_list_to_length(self, a, frames):
        """
        Fill the list a to length 'frames' by repeating its elements.
        The elements from the end of the list are repeated more if necessary.
        """
        filled_list = []
        len_a = len(a)
        repeats = frames // len_a  # Number of times the whole list can be repeated
        additional_elements = frames % len_a  # Number of additional elements needed

        for i in range(len_a):
            # Repeat each element 'repeats' times or 'repeats + 1' times for the last few elements
            repeat_times = repeats + 1 if i >= len_a - additional_elements else repeats
            filled_list.extend([a[i]] * repeat_times)

        return filled_list

    def _select_indices(self, numF):
        if self.fixed_fps is not None:
            allInd = np.arange(numF)
            if len(allInd) < self.frames: 
                allInd = self.fill_list_to_length(allInd, self.frames)
                assert len(allInd) == self.frames
            start = np.random.randint(low=0, high=len(allInd) - self.frames + 1)
            indices = allInd[start:start + self.frames]
            return indices

        elif self.sampling_fps is not None:
            fpsR_ = 1 / 30 * self.sampling_fps
            # fpsR_ = 1 / 10      # (1 / 30fps * 3fps)
            allInd = np.arange(np.random.randint(1 / fpsR_), numF, 1 / fpsR_, dtype=int)
            if len(allInd) < self.frames: 
                allInd = self.fill_list_to_length(allInd, self.frames)
                assert len(allInd) == self.frames
            start = np.random.randint(low=0, high=len(allInd) - self.frames + 1)
            indices = allInd[start:start + self.frames]
            return indices
        
        else:
            fpsR_ = random.choice(self.fpsR)
            allInd = np.linspace(np.random.randint(1 / fpsR_), numF - 1, num=round(fpsR_ * numF), endpoint=True, dtype=int)
            if len(allInd) <= self.frames: allInd = np.linspace(0, numF - 1, num=int(numF), endpoint=True, dtype=int)
            start = np.random.randint(len(allInd) - self.frames)
            indices = allInd[start:start + self.frames]
            return indices

    def __getitem__(self, idx):
        vpath = self.video_path[idx]
        action_name_ = self.video_class_info[idx].split('/')[-2]
        action_name = self.name_map[action_name_]
        feature = torch.from_numpy(np.load(vpath))
        num_frames = feature.size(0)
        # if num_frames > self.frames:
        # apply best_effort # TODO: FIX that branch would be determined through arg
        indices = self._select_indices(num_frames)
        out = feature[indices]
        # else:
        #     # pad by the last feature
        #     out = feature[-1, :][None, :].repeat([self.frames, 1])
        #     out[0:num_frames, :] = feature
        return out, action_name

# More datasets to be continued
