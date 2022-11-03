#---------- Imports ----------
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import cv2

#---------- Load The Images ----------
images = []
for filename in glob.glob('Images/*.png'): #assuming gif
    im=Image.open(filename).convert('RGB')
    #im = np.array(im)
    images.append(im)
    
# cv2.imshow("AS", np.array(images[0]))
# cv2.waitKey(0)

#---------- Create Variation Data Augmentation To Expand Dataset ----------
amount_of_augmentated_images = 2000

T_ColorJitter = transforms.ColorJitter(brightness=.5, hue=.3, saturation=.4, contrast=.2)
T_RandomHorizontalFlip = transforms.RandomHorizontalFlip(0.3)
T_RandomVerticalFlip =  transforms.RandomVerticalFlip(0.3)
T_GaussianBlur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))

transforms_list = [T_ColorJitter, T_RandomHorizontalFlip, T_RandomVerticalFlip, T_GaussianBlur]
data_aug = transforms.Compose(transforms_list)

# # Testing the augmentations
img = Image.open("Dataset/RGB_200.png")
for i in range(100):
    re = transforms.Resize(256)
    jitter = transforms.ElasticTransform(alpha=50.0)
    img_resized = re(img)
    augmented_img = data_aug(img_resized)
    cv2.imshow("ass", np.array(augmented_img))
    cv2.waitKey(0)

# for i in range(amount_of_augmentated_images):






# print(processed_images[0].shape)
# input_batch = processed_images[0].unsqueeze(0)
# print(input_batch.shape)


#---------- Create Resnet Augmentations ----------
preprocess_to_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#---------- Apply Resnet Augmentations ----------
resnet_processed_images = []
for img in images:
    resnet_processed_images.append(preprocess_to_resnet(img))
