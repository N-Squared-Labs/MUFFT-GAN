import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# PatchGAN Discriminator simple implementation
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(opt.img_channels, opt.ndf, opt.kernel_dim, opt.stride, opt.padding))
        layers.append(nn.LeakyReLU(0.2, True))
        
        depth_scale_curr = 1
        depth_scale_prev = 1
        for i in range(1, opt.num_layers-2):
            depth_scale_prev = depth_scale_curr
            depth_scale_curr = min(2**i, opt.fcd)
            layers.append(nn.Conv2d(opt.ndf*depth_scale_prev, opt.ndf*depth_scale_curr,
                          opt.kernel_dim, opt.stride, opt.padding))
            layers.append(nn.BatchNorm2d(opt.ndf*depth_scale_curr))
            layers.append(nn.LeakyReLU(0.2, True))
        
        depth_scale_prev = depth_scale_curr
        depth_scale_curr = min(2**i, opt.fcd)
        layers.append(nn.Conv2d(opt.ndf*depth_scale_prev, opt.ndf*depth_scale_curr,
                          opt.kernel_dim, 1, opt.padding))
        layers.append(nn.BatchNorm2d(opt.ndf*depth_scale_curr))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(opt.ndf*depth_scale_prev, 1, opt.kernel_dim, 1, opt.padding))

        self.model = nn.Sequential(*layers)


    def compute_D_loss(self, reals, fakes, criterion):
        fake_predictions = forward(fakes)
        fake_label = torch.full(reals.size(0), 0, dtype=torch.float, device=opt.device)
        loss_fakes = criterion(fake_predictions, fake_label).mean()

        real_predictions = forward(reals)
        real_label = torch.full(reals.size(0), 1, dtype=torch.float, device=opt.device)
        loss_reals = criterion(real_predictions, real_label).mean()

        loss = (loss_fakes + loss_reals) / 2
        return loss


    def forward(self, x):
        return self.model(x)
        


# PatchGAN Generator simple implementation
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(opt.nz, opt.ngf, opt.kernel_dim, opt.stride, opt.padding))
        layers.append(nn.ReLU(True))
        
        
        for i in range(opt.num_layers-2):
            depth_scale = 2 ** (opt.num_layers-2-i)
            layers.append(nn.ConvTranspose2d(opt.ngf*depth_scale, int(opt.ngf*depth_scale/2),
                          opt.kernel_dim, opt.stride, opt.padding))
            layers.append(nn.BatchNorm2d(int(opt.ngf*depth_scale / 2)))
            layers.append(nn.ReLU(True))
        
        
        layers.append(nn.ConvTranspose2d(opt.ngf, opt.img_channels, opt.kernel_dim, 1, opt.padding))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)

    def compute_G_loss(self, fake_predictions, criterion):
        fake_label = torch.full(reals.size(0), 0, dtype=torch.float, device=opt.device)
        loss = criterion(fake_predictions, fake_label).mean() * self.opt(lambda_G)
        return loss

    def forward(self, x):
        return self.model(x)
