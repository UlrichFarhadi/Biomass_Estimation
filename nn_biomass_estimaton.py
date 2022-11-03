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
import matplotlib as mpl
import re
import json
from tqdm import tqdm

#---------- Load Dataset ----------
def load_all_images():
    with open("Dataset/result.json") as f:
        data = f.read()
        rgb_list = []
        depth_list = []
        fresh_weight_list = []
        dry_weight_list = []
        for filename in tqdm(glob.glob('Dataset/Debth*.png')): #assuming gif
            num = re.findall(r'\d+',filename )

            img_depth = Image.open("Dataset/Debth_" + num[0] +".png") #Image.open("Dataset/Debth_1.png")
            img_depth = (img_depth / np.linalg.norm(img_depth))*255
            cm = mpl.cm.get_cmap('jet')
            img_depth = cm(np.array(img_depth))
            img_depth = Image.fromarray(np.uint8(img_depth[:,:,:3]*255))

            img_rgb = Image.open("Dataset/RGB_" + num[0] +".png")
            js = json.loads(data)

            FreshWeight = js.get( num[0]).get("FreshWeightShoot")
            DryWeight = js.get( num[0]).get("DryWeightShoot")

            rgb_list.append(img_rgb)
            depth_list.append(img_depth)
            fresh_weight_list.append(FreshWeight)
            dry_weight_list.append(DryWeight)
    return rgb_list, depth_list, fresh_weight_list, dry_weight_list
rgb_list, depth_list, fresh_weight_list, dry_weight_list = load_all_images()

# #---------- Load ResNet Backbone (Pretrained Model) ----------
resnet_version = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
model_RGB = torch.hub.load('pytorch/vision:v0.10.0', resnet_version[3], weights="ResNet101_Weights.DEFAULT")
#model_D = torch.hub.load('pytorch/vision:v0.10.0', resnet_version[3], pretrained=True)

model_RGB.eval() # We don't want the resnet to update
#model_D.eval()

# Remember to use torch.no_grad() when training

#---------- Create Regression Head  ----------
head_input_neurons = 2000
head_hidden = [1000, 500, 250]
head_output_neurons = 2
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



def train_step(rgb,depht,ground_truth):

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


train_step(rgb_list[0], depth_list[0],torch.tensor(np.array([ fresh_weight_list[0], dry_weight_list[0]])))
    




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
