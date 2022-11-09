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

    def __getitem__(self, idx):
        x_data = torch.tensor(self.data[idx]).float().unsqueeze(0)
        y_label = torch.tensor(self.labels[idx])
        return (x_data, y_label)


def data_splittage(dataset_size, percentage_train_validation_test):
    p_train = percentage_train_validation_test[0] / 100
    p_validation = percentage_train_validation_test[1] / 100

    amt_train = dataset_size * p_train
    amt_valid = dataset_size * p_validation
    amt_test = dataset_size - amt_test - amt_valid

    return [amt_train, amt_valid, amt_test]