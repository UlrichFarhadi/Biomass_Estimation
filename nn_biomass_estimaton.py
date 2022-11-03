#---------- Imports ----------
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import sequential_creator
import glob
from torchmetrics import MeanAbsolutePercentageError as MAPE

#---------- Load Dataset ----------
#DUMMY CODE PLS DELTE
img_depht = Image.open("Dataset/RGB_1.png") #Image.open("Dataset/Debth_1.png")
img_rgb = Image.open("Dataset/RGB_1.png")

# #---------- Load ResNet Backbone (Pretrained Model) ----------
resnet_version = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
model_RGB = torch.hub.load('pytorch/vision:v0.10.0', resnet_version[3], pretrained=True)
#model_D = torch.hub.load('pytorch/vision:v0.10.0', resnet_version[3], pretrained=True)

model_RGB.eval() # We don't want the resnet to update
#model_D.eval()

# Remember to use torch.no_grad() when training

#---------- Create Regression Head  ----------
head_input_neurons = 2000
head_hidden = [1000, 500, 250]
head_output_neurons = 1
head_activation = torch.nn.ReLU()
regression_head = sequential_creator.make_model(input=head_input_neurons, hidden=head_hidden, output=head_output_neurons, activation=head_activation)

print(regression_head)



#------------ Resnet --------------------
def resnet_rgb(img):
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
        model_RGB.to('cuda')
    with torch.no_grad():
        return model_RGB(input_batch)

def resnet_depht(img):
    return resnet_rgb(img)



def train_step(rgb,depht):

    rgb_out = resnet_rgb(rgb)
    depht_out = resnet_depht(depht)
    
    reg_input = torch.reshape(torch.stack((rgb_out,depht_out),dim= 0),(-1,))

    optimizer = torch.optim.NAdam(regression_head.parameters(),lr = 0.001)
    loss_fn = MAPE() #(pred,target) preds (Tensor) – Predictions from model,  target (Tensor) – Ground truth values

    pred = regression_head(reg_input)
    
    optimizer.zero_grad()
    #ground_truth = pred
    loss = loss_fn(pred,ground_truth)
    print(loss)
    loss.backward()

    optimizer.step()


train_step(img_rgb,img_depht)
    




# # sample execution (requires torchvision)
# input_image  = Image.open("Images/p (1).png")
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

# with torch.no_grad():
#     output = model(input_batch)
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)
