import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2

# Initializes fixed parameter values 
def init_config(opt):
    opt.device = torch.device('cpu')
    
    # Generate and Check Output Directory
    output_dir = generate_output_dir(opt)
    if(os.path.exists(output_dir)):
        print('Model has already been generated')
        exit(1)
    else:
        os.makedirs(output_dir)
        opt.output_dir = output_dir

    return opt

# Returns the path for the output generated images to be saved
def generate_output_dir(opt):
    output_dir = 'results/' + opt.translation + '/'
    return output_dir

# Read in the input image
def read_input(opt):
    img = cv2.imread(os.path.join(opt.input_dir, opt.input_name))
    return np2torch(img)

