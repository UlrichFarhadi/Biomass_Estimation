#---------- Imports ----------
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
from torchmetrics import MeanAbsolutePercentageError as MAPE
import matplotlib as mpl
import re
import json
from tqdm import tqdm


#---------- Create The Dataset Class ----------
class LettuceDataset(Dataset):
    # data is the cocatenated 
    def __init__(self, data, labels): 
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

#   def __getitem__(self, idx):
    #     x_1 = torch.tensor(np.array(self.data[idx])).float()
    #     #x_2 = torch.tensor(np.array(self.data[idx][1])).float()
    #     #x_data = [x_1, x_2]
    #     x_data = x_1
    #     #x_data = torch.tensor(self.data[idx]).float().unsqueeze(0)
    #     y_label = torch.tensor(self.labels[idx]).float()
    #     return (x_data, y_label)

    def __getitem__(self, idx):
    
        T_ColorJitter = transforms.ColorJitter(brightness=.5, hue=.3, saturation=.4, contrast=.2)
        # T_RandomHorizontalFlip = transforms.RandomHorizontalFlip(0.3)
        # T_RandomVerticalFlip =  transforms.RandomVerticalFlip(0.3)
        T_GaussianBlur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))

        #transforms_list = [T_ColorJitter, T_RandomHorizontalFlip, T_RandomVerticalFlip, T_GaussianBlur]
        transforms_list = [T_ColorJitter, T_GaussianBlur]
        data_aug = transforms.Compose(transforms_list)
        t_topil = transforms.ToPILImage()
        TT = transforms.ToTensor()
        aug_rgb = data_aug(t_topil(torch.tensor(self.data[idx][:3,:,:]).float()))
        temp_data = self.data[idx]
        temp_data[:3,:,:] =  np.array(TT(aug_rgb))

        
        x_1 = torch.tensor(np.array(temp_data)).float()
        #x_2 = torch.tensor(np.array(self.data[idx][1])).float()
        #x_data = [x_1, x_2]
        x_data = x_1
        #x_data = torch.tensor(self.data[idx]).float().unsqueeze(0)
        y_label = torch.tensor(self.labels[idx]).float()
        return (x_data, y_label)


def data_splittage(dataset_size, percentage_train_validation_test):
    p_train = percentage_train_validation_test[0] / 100
    p_validation = percentage_train_validation_test[1] / 100

    amt_train = int(dataset_size * p_train)
    amt_valid = int(dataset_size * p_validation)
    amt_test = dataset_size - amt_train - amt_valid

    return [amt_train, amt_valid, amt_test]