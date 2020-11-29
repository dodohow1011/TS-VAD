import os
import numpy as np
import torch
import random
from kaldi.util.io import read_matrix

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    files = [f.rstrip().split() for f in files]
    return files


def files_to_dict(filename):
    """
    Takes a text file of filenames and makes a dict of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    files = dict([f.rstrip().split() for f in files])
    return files

def load_scp_to_torch(scp_path):
    """
    Loads data into torch array
    """
    data = read_matrix(scp_path).numpy()
    return torch.from_numpy(data).float()

def load_wav_to_torch(scp_path):
    """
    Loads wavdata into torch array
    """
    data = read_matrix(scp_path).numpy().reshape(-1)
    data = data / MAX_WAV_VALUE
    return torch.from_numpy(data).float()


def load_spk_to_torch(spk_id, length=1):
    """
    Loads spk_id into torch tensor
    """
    return torch.ones(1, dtype=torch.long) * int(spk_id)
