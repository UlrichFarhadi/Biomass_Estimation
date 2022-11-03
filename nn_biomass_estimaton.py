#---------- Imports ----------
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os


#---------- Load Dataset ----------
import glob
image_list = []
for filename in glob.glob('Images/*.png'): #assuming gif
    im=Image.open(filename)
    im = np.array(im)
    image_list.append(im)

# Test if the images are correctly shown
# cv2.imshow("Test", image_list[0])
# cv2.waitKey(0)

# #---------- Load Model ----------
resnet_version = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
model_RGB = torch.hub.load('pytorch/vision:v0.10.0', resnet_version[3], pretrained=True)
model_D = torch.hub.load('pytorch/vision:v0.10.0', resnet_version[3], pretrained=True)

model_RGB.eval() # We don't want the resnet to update
model_D.eval()
# Remember to use torch.no_grad() when training