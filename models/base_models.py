import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        layers = []
        for i in range(1, num_layers):
            layers.append(nn.ConvTranspose2d(max(2*N,opt.min_nfc), max(N,opt.min_nfc),
                                             opt.ker_size, 1, opt.padd_size))
            layers.append(BatchNorm2d(max(N, opt.min_nfc)))
            layers.append(nn.ReLu(True))
        nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
        nn.Tanh()
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x
