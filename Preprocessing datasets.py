# Preprocessing dataset
The datasets, named Machine Learning Dataset 1 and Machine Learning Dataset 2, offer an extensive array of information regarding poultry health diagnostics. These datasets consist of meticulously annotated data, employing Polymerase Chain Reaction (PCR) alongside farm-labeled fecal images. The images were captured in the regions of Arusha and Kilimanjaro in Tanzania during the timeframe spanning September 2020 to February 2021.

The dataset encompasses distinct categories that represent various poultry health conditions. The 'healthy' class features fecal material that is typical and normal, sourced from poultry farms. The 'cocci' class, on the other hand, highlights fecal samples from chickens affected by Coccidiosis disease. Furthermore, the 'salmo' class illustrates fecal images from chickens that were inoculated with Salmonella disease, with the images taken one week post-inoculation. Similarly, the 'ncd' class showcases fecal images from chickens inoculated with Newcastle disease, captured within a three-day window post-inoculation.

This diverse dataset offers valuable insights for the development of machine learning models dedicated to poultry disease diagnosis based on fecal images. The inclusion of various disease classes allows for the robust training and evaluation of models across different scenarios of poultry health.


# Imports 
# open folder
from os import makedirs
from os import listdir
import os
from shutil import copyfile

# open image
import cv2
from PIL import Image

# random 
from random import seed
from random import random

# basic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# image augmentation
import albumentations as A





## 01.Make Dataset_train_valid_test
---
Because it is necessary to divide the data into the respective directories in order to support images for Artificial Neural Network learning using Keras and Tensorflow tools:
![pre-precessiong](../picture/pre-precessiong.png)

**Train subdirectory** is data to use for neural network learning to find optimal weight and make architecture.

**Validation subdirectory** is data to make sure the neural network model is not overfitting with training datasets.

**Testing subdirectory** is data to make final capability with our model




# you can `download dataset form https://doi.org/10.5281/zenodo.5801834, https://doi.org/10.5281/zenodo.4628934`
# and bring all together in a folder 
# you must rename image data for pcr dataset such as pcrcocci.1.jpg to cocci.x.jpg
# (x follow by after order from farmmer-labeled dataset)
src_directory = 'all' # path diretories have all data

# random data form all diretories to of each subdirectories
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    
    # train valid test split
    if random() < val_ratio_1: # 80/20
        if random() < val_ratio_2: # 50/50 
            dst_dir = 'test/'
        else :
            dst_dir = 'valid/'
            
    # copyimages into subdirectories
    if file.startswith('cocci'):
        dst = dataset_home + dst_dir + 'cocci/'+ file
        copyfile(src, dst)
    elif file.startswith('healthy'):
        dst = dataset_home + dst_dir + 'healthy/' + file
        copyfile(src, dst)
    elif file.startswith('ncd'):
        dst = dataset_home + dst_dir + 'ncd/' + file
        copyfile(src, dst)
    elif file.startswith('salmo'):
        dst = dataset_home + dst_dir + 'salmo/' + file
        copyfile(src, dst)


# Recheck
# cocci in train
train_cocci = len(listdir('dataset_train_valid_test/train/cocci/'))
# cocci in test
test_cocci = len(listdir('dataset_train_valid_test/test/cocci/'))
# cocci in valid
valid_cocci = len(listdir('dataset_train_valid_test/valid/cocci/'))

# Show percent of each subdirectories in cocci class
print(f"train : {round(train_cocci/(train_cocci+test_cocci+valid_cocci),3)}")
print(f"test : {round(test_cocci/(train_cocci+test_cocci+valid_cocci),3)}")
print(f"validate : {round(valid_cocci/(train_cocci+test_cocci+valid_cocci),3)}")




train_cocci: 2103
test_cocci: 2477
valid_cocci: 374
train : 0.425
test : 0.5
validate : 0.075
