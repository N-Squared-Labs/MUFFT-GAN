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
       
        layers = []
        # layers.append(PrintLayer())
        layers.append(nn.Conv2d(opt.img_channels, opt.ndf, opt.kernel_dim, opt.stride, opt.padding))
        layers.append(nn.LeakyReLU(0.2, True))
        
        layers.append(nn.Conv2d(opt.ndf, opt.ndf*2, opt.kernel_dim, opt.stride, opt.padding))
        # layers.append(PrintLayer())
        layers.append(nn.BatchNorm2d(opt.ndf*2))
        layers.append(nn.LeakyReLU(0.2, True))

        layers.append(nn.Conv2d(opt.ndf*2, opt.ndf*4, opt.kernel_dim, opt.stride, opt.padding))
        # layers.append(PrintLayer())
        layers.append(nn.BatchNorm2d(opt.ndf*4))
        layers.append(nn.LeakyReLU(0.2, True))

        layers.append(nn.Conv2d(opt.ndf*4, opt.ndf*8, opt.kernel_dim, opt.stride, opt.padding))
        # layers.append(PrintLayer())
        layers.append(nn.BatchNorm2d(opt.ndf*8))
        layers.append(nn.LeakyReLU(0.2, True))
        
        layers.append(nn.Conv2d(opt.ndf*8, 1, opt.kernel_dim, 1, 0))
        # layers.append(PrintLayer())

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
        # print("fakes:", fakes.shape)
        fake_predictions = self.forward(fakes)
        # print("fake:\n", fake_predictions)
        fake_label =  torch.tensor(0, dtype=torch.float, device=opt.device)
        fake_label = fake_label.expand_as(fake_predictions)
        loss_fakes = criterion(fake_predictions, fake_label).mean()

        real_predictions = self.forward(reals)
        # real_label = torch.full((reals.size(0),), 1, dtype=torch.float, device=opt.device)
        real_label =  torch.tensor(1, dtype=torch.float, device=opt.device)
        real_label = real_label.expand_as(real_predictions)
        # print('Real Predictions: ', real_predictions.shape)
        # print('Real Labels: ', real_label.shape)
        loss_reals = criterion(real_predictions, real_label).mean()

        loss = (loss_fakes + loss_reals) / 2

        # ----------------------
        # Calculate Metrics
        # ----------------------
        fakes_class = fake_predictions.mean().item()
        reals_class = real_predictions.mean().item()

        return loss, fakes_class, reals_class


    def forward(self, x):
        # print("Discriminator")
        return self.model(x)

# Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1

# Output width = (Output width + padding width right + padding width left - kernel width) / (stride width) + 1
# class Generator(nn.Module):
#     def __init__(self, opt):
#         super(Generator, self).__init__()
#         layers=[]
#         layers.append(PrintLayer())
#         layers.append(nn.ReflectionPad2d(3))
#         # 64 x 64 x 3
#         layers.append(nn.Conv2d(opt.img_channels, opt.ngf, kernel_size=7, stride=1, padding=0))
#         # 64 x 64 x 32
#         layers.append(nn.BatchNorm2d(opt.ngf))
#         layers.append(nn.ReLU(True))
#         layers.append(PrintLayer())
        
#         # 64 x 64 x 32 -> 32 x 32 x 64 -> 16 x 16 x 128 -> 8 x 8 x 256
#         for i in range(opt.num_layers-2):
#             depth_scale = 2 ** i
#             target_depth = opt.ngf*depth_scale*2
#             layers.append(nn.Conv2d(opt.ngf*depth_scale, target_depth, kernel_size=4, stride=2, padding=1, bias=1))
#             layers.append(nn.BatchNorm2d(target_depth))
#             layers.append(nn.ReLU(True))
#             layers.append(PrintLayer())
        
#         # 8 x 8 x 256 -> 8 x 8 x 256
#         target_depth = 2 ** (opt.num_layers-2)
#         for i in range(opt.resnet_blocks):
#             layers.append(ResnetBlock(opt, target_depth*opt.ngf))  
#             layers.append(PrintLayer())

#         # 8 x 8 x 256 -> 16 x 16 x 128
#         for i in range(opt.num_layers-2):
#             depth_scale = 2 ** (opt.num_layers-2-i) # 3, 2, 1
#             target_depth = int(opt.ngf * depth_scale / 2)
#             layers.append(nn.ConvTranspose2d(opt.ngf*depth_scale, target_depth, kernel_size=4, stride=2, padding=1, bias=1))
#             layers.append(nn.BatchNorm2d(target_depth))
#             layers.append(nn.ReLU(True))
#             layers.append(PrintLayer())

#         layers.append(nn.ReflectionPad2d(3))
#         layers.append(nn.Conv2d(opt.ngf, opt.img_channels, kernel_size=7, stride=1, padding=0))
#         layers.append(PrintLayer())
#         layers.append(nn.Tanh())
#         self.model = nn.Sequential(*layers)

#     def compute_G_loss(self, fake_predictions, criterion, opt):
#         fake_label =  torch.tensor(0, dtype=torch.float, device=opt.device)
#         fake_label = fake_label.expand_as(fake_predictions)

#         loss = criterion(fake_predictions, fake_label).mean() * opt.lambda_G
#         return loss

#     def forward(self, x):
#         print("Generator")
#         return self.model(x)


# class ResnetBlock(nn.Module):
#     def __init__(self, opt, dimension):
#         super(ResnetBlock, self).__init__()
#         layers=[]
#         layers.append(nn.ReflectionPad2d(1))
#         layers.append(nn.Conv2d(dimension, dimension, kernel_size=3, padding=0, bias=1))
#         layers.append(nn.BatchNorm2d(dimension))
#         layers.append(nn.ReLU(True))

#         # Not sure if we should use dropout, but maybe
#         layers.append(nn.Dropout(0.5))
        
#         layers.append(nn.ReflectionPad2d(1))
#         layers.append(nn.Conv2d(dimension, dimension, kernel_size=3, padding=0, bias=1))
#         layers.append(nn.BatchNorm2d(dimension))

#         self.model = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return x + self.model(x)


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
