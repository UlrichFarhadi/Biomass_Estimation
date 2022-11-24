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
import torchmetrics
import matplotlib as mpl
import re
import json
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, GT):
        #error = ((torch.abs(prediction - GT))/GT) 
        error = torch.abs(prediction - GT)
        return torch.mean(torch.log(torch.cosh(error + 1e-12)))


class BiomassModel(pl.LightningModule):
    def __init__(self, CNNmodel, lr = 1e-3):
        super().__init__()
        self.lr = lr
        self.model = CNNmodel
        #self.loss_func = torch.nn.MSELoss()
        self.loss_func = LogCoshLoss()

    def prediction(self, img):
        with torch.no_grad():
            return self.forward(torch.unsqueeze(img, dim=0))
        # #self.device = torch.device('cuda:0')
        # # print(depth.device)
        # # print(rgb.device)
        # with torch.no_grad():
        #     rgb_out = self.resnet_rgb(torch.unsqueeze(rgb, dim=0,))
        #     depth_out = self.resnet_depth(torch.unsqueeze(depth, dim=0))

        #     reg_input = torch.reshape(torch.stack((depth_out, rgb_out), dim= 1), (1,-1))
        #     pred = self.forward(reg_input)
        #     return pred

    def forward(self, x):  # (B, n_channels, h, w)
        return self.model(x) # (B, n_pts, h, w)

    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, _):
        img,weights = batch
        pred = self.forward(img)
        loss = self.loss_func(pred,weights)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img,weights = batch
        pred = self.forward(img)
        loss = self.loss_func(pred,weights)
        self.log("validation_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        img,weights = batch
        pred = self.forward(img)
        loss = self.loss_func(pred,weights)
        self.log("test_loss", loss)
        return loss
        

def get_trainer():
    epochs = 40
    loggerT = pl_loggers.TensorBoardLogger(save_dir="logs/", name="my_model")
    early_stop_callback = EarlyStopping(monitor="validation_loss", min_delta=0.00, patience=10, verbose=False, mode="max")
    return pl.Trainer(
        accelerator="auto", 
        auto_select_gpus=True, 
        logger=loggerT,# enable_checkpointing=False,
        max_epochs=epochs,
        #callbacks=[early_stop_callback],
        #progress_bar_refresh_rate=0,
        #enable_model_summary=False,
    )

