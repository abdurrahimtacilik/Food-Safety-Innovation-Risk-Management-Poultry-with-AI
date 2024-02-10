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
