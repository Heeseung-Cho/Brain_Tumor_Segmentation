import matplotlib.pyplot as plt
import numpy as np
import random
import SimpleITK as sitk  # For loading the dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import math

def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    """    
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def get_datapath(datadir, random_state, test_size = 0.25):
    dirs = []
    images = []
    masks = []
    for dirname, _, filenames in os.walk(datadir):
        for filename in filenames:
            if 'mask'in filename:
                dirs.append(dirname.replace(datadir, ''))
                masks.append(filename)
                images.append(filename.replace('_mask', ''))

    image_list = []
    mask_list = []
    for i in range(len(dirs)):  
        imagePath = os.path.join(datadir, dirs[i], images[i])
        maskPath = os.path.join(datadir, dirs[i], masks[i])    
        image_list.append(imagePath)
        mask_list.append(maskPath)       
    return image_list, mask_list


class DataSegmentationLoader(Dataset):
    def __init__(self, path_list,ground_list = []):
        self.sample = path_list
        self.ground_truth = []
        if len(ground_list) > 0:
            self.ground_truth = ground_list
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        #Load Data        
        data = read_img(self.sample[idx]).reshape(3,256,256)/255
        if len(self.ground_truth) > 0:
            label = read_img(self.ground_truth[idx]).reshape(1,256,256)/255
        else:
            label = np.zeros((1,256,256))
        
        return torch.from_numpy(data).float(), torch.from_numpy(label).long()