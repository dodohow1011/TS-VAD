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
    def __init__(self, training_dir='./data', crop_length=40):
        
        self.utt2feat   = files_to_dict(os.path.join(training_dir,'train/feats.scp'))
        self.utt2nframe = files_to_dict(os.path.join(training_dir,'train/utt2num_frames'))       
        self.utt2target = files_to_dict(os.path.join(training_dir,'dense_targets.scp'))
        self.utt2iv     = files_to_dict(os.path.join(training_dir,'ivector_online.scp'))
        self.utt_list   = [k for k in self.utt2target.keys()]
        
        self.crop_length = crop_length

    def __getitem__(self, index):
        utt         = self.utt_list[index]
        feat_length = int(self.utt2nframe[utt])

        feat        = load_scp_to_torch(self.utt2feat[utt]).unsqueeze(0)
        target      = load_scp_to_torch(self.utt2target[utt])[:, 0::2]
        ivectors    = load_scp_to_torch(self.utt2iv[utt])[0]
        
        if feat_length < self.crop_length:
            feat   = F.pad(feat, (0, 0, 0, self.crop_length-feat_length))
            target = F.pad(target, (0, 0, 0, self.crop_length-feat_length))
            mask   = F.pad(torch.ones(feat_length), (0, self.crop_length-feat_length))
        
        else:
            max_start  = feat_length - self.crop_length
            feat_start = random.randint(0, max_start)
            feat       = feat[:, feat_start:(feat_start+self.crop_length)]
            target     = target[feat_start:(feat_start+self.crop_length)]
            mask       = torch.ones(self.crop_length)

        return feat, target, ivectors, mask
    
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
        ivectors    = load_scp_to_torch(self.utt2iv[utt])
        for i in range(ivectors.size(0)):
            assert torch.equal(ivectors[0], ivectors[i])
        ivectors = ivectors[0]
        
        return utt, feat, ivectors
    
    def __len__(self):
        return len(self.utt_list)
