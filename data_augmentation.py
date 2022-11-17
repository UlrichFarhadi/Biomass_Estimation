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

#---------- Load The Images ----------
# images = []
# for filename in glob.glob('Images/*.png'): #assuming gif
#     im=Image.open(filename).convert('RGB')
#     #im = np.array(im)
#     images.append(im)
# amount_of_non_augmented_images = len(images)
# print(amount_of_non_augmented_images)
# # cv2.imshow("AS", np.array(images[0]))
# # cv2.waitKey(0)

#---------- Create data augmentation and also augmentations to convert to ResNET input format ----------
def augment_data(rgb_images, depth_images, fresh_weight_GT, dry_weight_GT, amount_of_augmentated_images):
    try:
        len(rgb_images) == len(depth_images)
    except:
        raise Exception("Amount of rgb_images is not equal to amount of depth images")
        return -1

    amount_of_non_augmented_images = len(rgb_images)
    T_ColorJitter = transforms.ColorJitter(brightness=.5, hue=.3, saturation=.4, contrast=.2)
    T_RandomHorizontalFlip = transforms.RandomHorizontalFlip(0.3)
    T_RandomVerticalFlip =  transforms.RandomVerticalFlip(0.3)
    T_GaussianBlur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))

    #transforms_list = [T_ColorJitter, T_RandomHorizontalFlip, T_RandomVerticalFlip, T_GaussianBlur]
    transforms_list = [T_ColorJitter, T_GaussianBlur]
    data_aug = transforms.Compose(transforms_list)

    #---------- Create transforms for resnet input ----------
    preprocess_to_resnet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Applying the augmentations
    rgb_images_augmented = []
    depth_images_augmented = []
    fresh_weight_GT_extended = []
    dry_weight_GT_extended = []
    t_topil = transforms.ToPILImage()
    print("Augmenting RGB Images")
    for i in tqdm(range(amount_of_augmentated_images)):
        img_idx = i % amount_of_non_augmented_images
        #augmented_img = data_aug(rgb_images[img_idx])   # Apply data augmentations
        augmented_img = rgb_images[img_idx]
        augmented_img = preprocess_to_resnet(augmented_img) # Apply resnet augmentations
        #t_topil(augmented_img).show()
        #cv2.waitKey(100)
        rgb_images_augmented.append(augmented_img)
        depth_images_augmented.append(preprocess_to_resnet(depth_images[img_idx]))
        fresh_weight_GT_extended.append(fresh_weight_GT[img_idx])
        dry_weight_GT_extended.append(dry_weight_GT[img_idx])
    print("Amount of augmented images", len(rgb_images_augmented))

    return depth_images_augmented, rgb_images_augmented, fresh_weight_GT_extended, dry_weight_GT_extended


# print(processed_images[0].shape)
# input_batch = processed_images[0].unsqueeze(0)
# print(input_batch.shape)

