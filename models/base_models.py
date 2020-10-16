import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


# PatchGAN Discriminator simple implementation
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        layers = []
        # layers.append(PrintLayer())
        # start with [3, 3, 11, 11]
        layers.append(nn.Conv2d(opt.img_channels, opt.ndf, opt.kernel_dim, opt.stride, opt.padding))
        # layers.append(PrintLayer())
        # torch.Size([3, 32, 9, 9])
        layers.append(nn.LeakyReLU(0.2, True))
        
        layers.append(nn.Conv2d(opt.ndf, opt.ndf*2, opt.kernel_dim, opt.stride, opt.padding))
        # layers.append(PrintLayer())
        # torch.Size([3, 64, 7, 7])
        layers.append(nn.BatchNorm2d(opt.ndf*2))
        layers.append(nn.LeakyReLU(0.2, True))

        layers.append(nn.Conv2d(opt.ndf*2, opt.ndf*4, opt.kernel_dim, opt.stride, opt.padding))
        # layers.append(PrintLayer())
        # torch.Size([3, 128, 5, 5])
        layers.append(nn.BatchNorm2d(opt.ndf*4))
        layers.append(nn.LeakyReLU(0.2, True))

        layers.append(nn.Conv2d(opt.ndf*4, opt.ndf*8, opt.kernel_dim, opt.stride, opt.padding))
        # layers.append(PrintLayer())
        # torch.Size([3, 256, 3, 3])
        layers.append(nn.BatchNorm2d(opt.ndf*8))
        layers.append(nn.LeakyReLU(0.2, True))
        
        # layers.append(PrintLayer())
        # torch.Size([3, 256, 3, 3])
        layers.append(nn.Conv2d(opt.ndf*8, 1, opt.kernel_dim, 1, opt.padding))

        # depth_scale_curr = 1
        # depth_scale_prev = 1
        # for i in range(1, opt.num_layers-2):
        #     # i = 1: prev=1, curr=2, conv2d=32->64
        #     depth_scale_prev = depth_scale_curr
        #     depth_scale_curr = min(2**i, opt.fcd)
        #     layers.append(nn.Conv2d(opt.ndf*depth_scale_prev, opt.ndf*depth_scale_curr,
        #                   opt.kernel_dim, opt.stride, opt.padding))
        #     layers.append(nn.BatchNorm2d(opt.ndf*depth_scale_curr))
        #     layers.append(nn.LeakyReLU(0.2, True))
        
        # depth_scale_prev = depth_scale_curr
        # depth_scale_curr = min(2**opt.num_layers-2, opt.fcd)
        # layers.append(nn.Conv2d(opt.ndf*depth_scale_prev, opt.ndf*depth_scale_curr,
        #                   opt.kernel_dim, 1, opt.padding))
        # layers.append(nn.BatchNorm2d(opt.ndf*depth_scale_curr))
        # layers.append(nn.LeakyReLU(0.2, True))
        # layers.append(nn.Conv2d(opt.ndf*depth_scale_curr, 1, opt.kernel_dim, 1, opt.padding))

        self.model = nn.Sequential(*layers)


    def compute_D_loss(self, reals, fakes, criterion, opt):
        print("fakes:", fakes.shape)
        fake_predictions = self.forward(fakes)
        fake_label =  torch.tensor(0, dtype=torch.float, device=opt.device)
        fake_label = fake_label.expand_as(fake_predictions)
        print('Fake Predictions: ', fake_predictions.shape)
        print('Fake Labels: ', fake_label.shape)
        loss_fakes = criterion(fake_predictions, fake_label).mean()

        real_predictions = self.forward(reals)
        # real_label = torch.full((reals.size(0),), 1, dtype=torch.float, device=opt.device)
        real_label =  torch.tensor(1, dtype=torch.float, device=opt.device)
        real_label = real_label.expand_as(real_predictions)
        print('Real Predictions: ', real_predictions.shape)
        print('Real Labels: ', real_label.shape)
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

        # input 3x100x1x1 -> output 32
        depth_scale = 2 ** (opt.num_layers-2)
        layers.append(nn.ConvTranspose2d(opt.nz, opt.ngf * depth_scale, opt.kernel_dim, 1, 0))
        layers.append(nn.ReLU(True))
        
        
        for i in range(opt.num_layers-2):
            # depth scale for num_layers=5: 8, 4, 2
            # first run through opt.ngf*depth_scale = 256, 128, 3, 3
            depth_scale = 2 ** (opt.num_layers-2-i)
            layers.append(nn.ConvTranspose2d(opt.ngf*depth_scale, int(opt.ngf*depth_scale/2), 
                                             opt.kernel_dim, opt.stride, opt.padding))
            layers.append(nn.BatchNorm2d(int(opt.ngf*depth_scale / 2)))
            layers.append(nn.ReLU(True))
    
        
        
        layers.append(nn.ConvTranspose2d(opt.ngf, opt.img_channels, opt.kernel_dim, opt.stride, opt.padding))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)

    def compute_G_loss(self, fake_predictions, criterion, opt):
        fake_label =  torch.tensor(0, dtype=torch.float, device=opt.device)
        fake_label = fake_label.expand_as(fake_predictions)

        loss = criterion(fake_predictions, fake_label).mean() * opt.lambda_G
        return loss

    def forward(self, x):
        return self.model(x)
