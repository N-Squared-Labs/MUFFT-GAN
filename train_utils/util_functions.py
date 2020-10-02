import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def init_config(opt):
    # Initialize Fixed Parameters -- NEED TO IMPLEMENT
    return opt

def generate_output_dir(opt):
    output_dir = 'results/' + opt.translation + '/'
    return output_dir