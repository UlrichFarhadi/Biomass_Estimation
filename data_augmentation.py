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

#---------- Create The Dataset Class ----------
class BiomassDataset(Dataset):
    # data is the cocatenated 
    def __init__(self, data, labels): 
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_data = torch.tensor(self.data[idx]).float().unsqueeze(0)
        y_label = torch.tensor(int(self.labels[idx]))
        return (x_data, y_label)

#---------- Load The Images ----------
images = []
for filename in glob.glob('Images/*.png'): #assuming gif
    im=Image.open(filename).convert('RGB')
    #im = np.array(im)
    images.append(im)
amount_of_non_augmented_images = len(images)
print(amount_of_non_augmented_images)
# cv2.imshow("AS", np.array(images[0]))
# cv2.waitKey(0)

#---------- Create data augmentation and also augmentations to convert to ResNET input format ----------
amount_of_augmentated_images = 100

T_ColorJitter = transforms.ColorJitter(brightness=.5, hue=.3, saturation=.4, contrast=.2)
T_RandomHorizontalFlip = transforms.RandomHorizontalFlip(0.3)
T_RandomVerticalFlip =  transforms.RandomVerticalFlip(0.3)
T_GaussianBlur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))

transforms_list = [T_ColorJitter, T_RandomHorizontalFlip, T_RandomVerticalFlip, T_GaussianBlur]
data_aug = transforms.Compose(transforms_list)

#---------- Create transforms for resnet input ----------
preprocess_to_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Applying the augmentations
images_augmented = []
t_topil = transforms.ToPILImage()
print("Augmenting RGB Images")
for i in tqdm(range(amount_of_augmentated_images)):
    img_idx = i % amount_of_non_augmented_images
    augmented_img = data_aug(images[img_idx])   # Apply data augmentations
    #augmented_img = images[img_idx]
    augmented_img = preprocess_to_resnet(augmented_img) # Apply resnet augmentations
    t_topil(augmented_img).show()
    cv2.waitKey(100)
    images_augmented.append(augmented_img)
print("Amount of augmented images", len(images_augmented))





# print(processed_images[0].shape)
# input_batch = processed_images[0].unsqueeze(0)
# print(input_batch.shape)

