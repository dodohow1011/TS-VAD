import os
import sys
import torch
import random
import numpy as np
import torch.nn.functional as F
from kaldi.util.io import read_matrix

from .utils import (files_to_list, files_to_dict, load_scp_to_torch)

class Dataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_dir='./data/train', nframes=40):
        
        self.utt2feat   = files_to_dict(os.path.join(training_dir,'feats.scp'))
        self.utt2nframe = files_to_dict(os.path.join(training_dir,'utt2num_frames'))       
        self.utt2target = files_to_dict(os.path.join(training_dir,'dense_targets.scp'))
        self.utt2iv     = files_to_dict(os.path.join(training_dir,'ivector_online.scp'))
        self.utt_list   = [k for k in self.utt2target.keys() if int(self.utt2nframe[k]) >= nframes ]
        
        self.nframes = nframes

    def __getitem__(self, index):
        utt         = self.utt_list[index]
        feat_length = int(self.utt2nframe[utt])

        assert feat_length >= self.nframes

        feat        = load_scp_to_torch(self.utt2feat[utt]).unsqueeze(0)
        target      = load_scp_to_torch(self.utt2target[utt])[:, 1::2]
        ivectors    = load_scp_to_torch(self.utt2iv[utt]).mean(dim=0)
        
        max_start  = feat_length - self.nframes
        feat_start = random.randint(0, max_start)
        feat       = feat[:, feat_start:(feat_start+self.nframes)]
        target     = target[feat_start:(feat_start+self.nframes)]

        return feat, target, ivectors
    
    def __len__(self):
        return len(self.utt_list)


class EvalDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, feats_dir, ivectors_dir):
        
        self.utt2feat   = files_to_dict(os.path.join(feats_dir,'feats.scp'))
        self.utt2iv     = files_to_dict(os.path.join(ivectors_dir,'ivector_online.scp'))
        self.utt_list   = [k for k in self.utt2feat.keys()]
        
    def __getitem__(self, index):
        utt         = self.utt_list[index]

        feat        = load_scp_to_torch(self.utt2feat[utt]).unsqueeze(0)
        ivectors    = load_scp_to_torch(self.utt2iv[utt]).mean(dim=0)
        
        return utt, feat, ivectors
    
    def __len__(self):
        return len(self.utt_list)
