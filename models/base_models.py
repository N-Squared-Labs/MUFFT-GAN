import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(opt.img_channels, opt.fcd, opt.kernel_dim, opt.padding, 1))
        for i in range(1, opt.num_layers-2):
            input_size = int(opt.fcd/pow(2,(i+1)))
            layers.append(nn.ConvTranspose2d(max(2*input_size,opt.min_fcd), max(input_size,opt.min_fcd),
                                             opt.kernel_dim, 1, opt.padding))
            layers.append(BatchNorm2d(max(input_size, opt.min_fcd)))
            layers.append(nn.ReLu(True))
        layers.append(nn.Conv2d(max(opt.fcd, opt.min_fcd), 1, kernel_size=opt.kernel_dim,
                      stride=1, padding=opt.padding))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.model(x)
        return x

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        layers = []
        # 3 x 32 x 32 => 32 x 30 x 30 => 
        layers.append(nn.ConvTranspose2d(opt.img_channels, opt.fcd, opt.kernel_dim, opt.padding, 1))
        for i in range(1, opt.num_layers-2):
            input_size = int(opt.fcd/pow(2,(i+1)))
            layers.append(nn.ConvTranspose2d(max(2*input_size,opt.min_fcd), max(input_size,opt.min_fcd),
                                             opt.kernel_dim, 1, opt.padding))
            layers.append(BatchNorm2d(max(input_size, opt.min_fcd)))
            layers.append(nn.ReLu(True))
        layers.append(nn.Conv2d(max(opt.fcd, opt.min_fcd), opt.img_channels, kernel_size=opt.kernel_dim, 
                      stride=1, padding=opt.padding))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):   # Add y when we include pyramid structure
        x = self.model(x)
        return x
