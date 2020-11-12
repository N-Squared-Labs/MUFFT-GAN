import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image
import random
import torchvision.transforms as transforms


# -----------------------------------
#       Utility Functions
# -----------------------------------

# Creates a path for the output generated images to be saved
def generate_output_dir(opt):
    output_dir = 'results/' + opt.translation + '/'
    if(os.path.exists(output_dir)):
        print('Model has already been generated')
        exit(1)
    else:
        os.makedirs(output_dir)
        opt.output_dir = output_dir

def create_dataset(directory):
    dataset = []
    for root, _, files in sorted(os.walk(directory, followlinks=True)):
        for filename in files:
            if is_image(filename):
                image_path = os.path.join(root, filename)
                dataset.append(image_path)
    return dataset

def is_image(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.png', '.JPG', '.PNG', '.jpeg', '.JPEG', '.tif', '.TIF'])

# Turns a tensor array into a numpy image array
# def tensor_to_image(image_arr):


# Saves a numpy image to the disk
# def save_image(image, image_path):


# -----------------------------------
#       Unpaired Dataset Class
# -----------------------------------
# Dataset class specifically for unpaired datasets with images from domain A and domain B
class UnpairedDataset():
    def __init__(self, opt):
        self.opt = opt
        # Grab directory paths for domains A and B
        self.domain_A = opt.input_dir_A
        self.domain_B = opt.input_dir_B

        # Load images from domains A and B
        self.A_paths = sorted(create_dataset(self.domain_A))
        self.B_paths = sorted(create_dataset(self.domain_B))
        self.size = len(self.A_paths)+len(self.B_paths)

    def __getitem__(self, index):
        # Get image from domain A at index, get random image from domain B
        A_path = self.A_paths[index]
        B_path = self.B_paths[random.randint(0, len(self.B_paths)-1)]

        #Convert to RGB Images
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        image_transformation = self.transform_image(self.opt, grayscale=False, resize=True)
        A_final = image_transformation(A)
        B_final = image_transformation(B)

        return {'A': A_final, 'B': B_final, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return self.size

    def transform_image(self, opt, grayscale=False, resize=True, method=Image.BICUBIC):
        transformations = []
        if grayscale:
            transformations.append(transforms.GrayScale(1))
        if resize:
            transformations.append(transforms.Resize((opt.img_size, opt.img_size), method))
        # Can add more types of transforms here

        # Convert to tensor
        transformations.append(transforms.ToTensor())
        return transforms.Compose(transformations)

    


# -----------------------------------
#       Image Reservoir Class
# -----------------------------------

#  Stores previously generated images to train and update the Discriminator on
# class Reservoir():





# -----------------------------------
#       Display Class
# -----------------------------------

#  Helper functions to display / save images and print useful logging information
# class Display():



