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
from torch.utils.data import DataLoader

#---------- Personal python files imports ----------
import load_lettuce_dataset
import data_augmentation
import model_and_training_files
import lettuce_dataset


#---------- Hyperparameters ----------
batch_size = 16
learning_rate = 1e-3

#---------- Other parameters ----------
augmented_dataset_size = 2000 # Size of the augmented dataset, so if the original dataset contained 103 images, they would be augmented and made into 2000 images
                              # One thing to note is that if this number is much bigger than the size of the original dataset, then they would most likely end up being duplicates, since there is not that many augmentations implemented at the moment

#---------- Load Lettuce Dataset ----------
rgb_list, depth_list, fresh_weight_list, dry_weight_list = load_lettuce_dataset.load_all_images()

#---------- Augment Lettuce Dataset ----------
depth_img_aug, rgb_imgs_aug, fresh_weight_GT, dry_weight_GT = data_augmentation.augment_data(rgb_images=rgb_list, depth_images=depth_list, fresh_weight_GT=fresh_weight_list, dry_weight_GT=dry_weight_list, amount_of_augmentated_images=augmented_dataset_size)

#---------- Create data loaders ----------
# Concatenate the depth and rgb images
full_dataset = []
full_dataset_labels = []
for i in range(augmented_dataset_size):
    full_dataset.append([depth_img_aug[i], rgb_imgs_aug[i]])
    full_dataset_labels.append([fresh_weight_GT[i], dry_weight_GT[i]])

# Define the dataset
dataset = lettuce_dataset.LettuceDataset(full_dataset, full_dataset_labels)

# Split the dataset in train, validation and test
splitted_data = lettuce_dataset.data_splittage(augmented_dataset_size, [75, 12.5, 12.5])
train_set, validation_set, test_set = torch.utils.data.random_split(dataset, splitted_data)
#train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [7500, 1250, 1250])

# Create dataloaders for train, validation and test data
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
validation_loader = DataLoader(dataset = validation_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

train_features, train_labels = next(iter(train_loader))
#print(train_features.shape())
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

#---------- Model Definition ----------
# Load ResNet Backbone (Pretrained Model)
resnet_version = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
# We can reuse the same resnet model since weights are not updated and both RGB and Depth are same format
resnet_model_RGB_and_depth = torch.hub.load('pytorch/vision:v0.10.0', resnet_version[3], weights="ResNet101_Weights.DEFAULT")
resnet_model_RGB_and_depth.eval() # We don't want the resnet to update (Sets a flag, kind of a switch to turn off gradient computation)

# Create Regression Head
head_input_neurons = 2000 # 1000 output feature vector from RGB resnet and 1000 from depth resnet
head_hidden = [1000, 500, 250]
head_output_neurons = 2
head_activation = torch.nn.ReLU()
regression_head = sequential_creator.make_model(input=head_input_neurons, hidden=head_hidden, output=head_output_neurons, activation=head_activation)

print(regression_head)

#---------- Training ----------
# Define the model with Pytorch Lightning
model = model_and_training_files.BiomassModel(regression_head=regression_head, 
                                                resnet_model_RGB_and_depth=resnet_model_RGB_and_depth, 
                                                train_loader=train_features, validation_loader=validation_loader, 
                                                test_loader=test_loader,
                                                lr=learning_rate)

trainer = model_and_training_files.get_trainer(max_epochs=1)    # gets the trainer (which is a class that takes the model and dataset)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader, test_loaders=test_loader) # Train the model

#---------- Save the weights ----------
torch.save(model, "saved_models/best_weights_V1.plk") # Saves the regression head
