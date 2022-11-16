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
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping


class BiomassModel(pl.LightningModule):
    def __init__(self, regression_head, resnet_model_RGB_and_depth, lr = 1e-3):
        super().__init__()
        self.lr = lr
        self.layers = regression_head
        self.Resnet = resnet_model_RGB_and_depth

    def forward(self, x):  # (B, n_channels, h, w)
        return self.layers(x) # (B, n_pts, h, w)

    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, _):
        depth_rgb, weights = batch  # weights fresh , dry

        depth = depth_rgb[0]
        rgb = depth_rgb[1]
        #testsphape = rgb.shape
        #testsphape = depth.shape
        #split rgb_depth to rgb and depth
        #depth = depth.permute(0, 3, 1, 2)
        with torch.no_grad():
            rgb_out = self.resnet_rgb(rgb)
            depth_out = self.resnet_depth(depth)

        reg_input = torch.reshape(torch.stack(( depth_out,rgb_out), dim= 2), (weights.shape[0],-1))
        #print(reg_input.shape)
        #torch.squeeze(input)
        #testsphape = reg_input.shape
        pred = self.forward(reg_input)  # (B, n_pts, h, w)
        loss_fn = MAPE().to(pred.device)
        loss = loss_fn(pred, weights)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        depth_rgb, weights = batch  # weights fresh , dry

        depth = depth_rgb[0]
        rgb = depth_rgb[1]
        #print(depth_rgb.shape)
        #split rgb_depth to rgb and depth
        #depth = depth.permute(0, 3, 1, 2)
        with torch.no_grad():
            rgb_out = self.resnet_rgb(rgb)
            depth_out = self.resnet_depth(depth)

        reg_input = torch.reshape(torch.stack(( depth_out,rgb_out), dim= 2), (weights.shape[0],-1))
        
        pred = self.forward(reg_input)  # (B, n_pts, h, w)
        loss_fn = MAPE().to(pred.device)
        loss = loss_fn(pred, weights)
        self.log("validation_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        depth_rgb, weights = batch  # weights fresh , dry

        depth = depth_rgb[0]
        rgb = depth_rgb[1]
        #split rgb_depth to rgb and depth
        with torch.no_grad():
            rgb_out = self.resnet_rgb(rgb)
            depth_out = self.resnet_depth(depth)

        reg_input = torch.reshape(torch.stack(( depth_out,rgb_out), dim= 2), (weights.shape[0],-1))
        pred = self.forward(reg_input)  # (B, n_pts, h, w)
        loss_fn = MAPE().to(pred.device)
        loss = loss_fn(pred, weights)
        self.log("test_loss", loss)
        return loss
        

    
    #------------ Resnet --------------------
    def resnet_rgb(self, img):
        # preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        # input_tensor = preprocess(img)
        input_batch = img  #.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.Resnet.to('cuda')
        with torch.no_grad():
            return self.Resnet(input_batch)

    def resnet_depth(self,img):
        return self.Resnet(img)
        

def get_trainer():
    epochs = 10
    loggerT = pl_loggers.TensorBoardLogger(save_dir="logs/", name="my_model")
    early_stop_callback = EarlyStopping(monitor="validation_loss", min_delta=0.00, patience=3, verbose=False, mode="max")
    return pl.Trainer(
        accelerator="auto", 
        auto_select_gpus=True, 
        logger=loggerT,# enable_checkpointing=False,
        max_epochs=epochs,
        callbacks=[early_stop_callback],
        #progress_bar_refresh_rate=0,
        #enable_model_summary=False,
    )

