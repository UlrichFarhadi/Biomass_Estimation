#---------- Library Imports ----------
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import sequential_creator
import glob
from torchmetrics import MeanAbsolutePercentageError as MAPE
import matplotlib as mpl
import re
import json
from tqdm import tqdm
import pytorch_lightning as pl


class BiomassModel(pl.LightningModule):
    def __init__(self, regression_head, Resnet_RGB_and_depth, train_loader, validation_loader, test_loader, lr = 1e-3):
        super().__init__()
        self.lr = lr
        self.layers = regression_head
        self.Resnet = Resnet_RGB_and_depth

    def forward(self, x):  # (B, n_channels, h, w)
        return self.layers(x) # (B, n_pts, h, w)

    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, _):
        rgb_depth, weights = batch  # (B, h, w), (B, 2, 2xy), (B,)

        #split rgb_depth to rgb and depth

        rgb_out = self.resnet_rgb(rgb)
        depth_out = self.resnet_depth(depth)

        reg_input = torch.reshape(torch.stack((rgb_out, depth_out), dim= 0), (-1,))
        pred = self.forward(reg_input)  # (B, n_pts, h, w)
        loss_fn = MAPE()
        loss = loss_fn(pred, weights)
        return loss
    #------------ Resnet --------------------
    def resnet_rgb(self,img):
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.Resnet.to('cuda')
        with torch.no_grad():
            return self.Resnet(input_batch)

    def resnet_depth(self,img):
        return self.Resnet(img)
        

def get_trainer(max_epochs=10):
    loggerT = TensorBoardLogger("tb_logs", name="my_model")
    return pl.Trainer(
        gpus=[0], 
        logger=loggerT,# enable_checkpointing=False,
        max_epochs=max_epochs,
        #progress_bar_refresh_rate=0,
        #enable_model_summary=False,
    )

