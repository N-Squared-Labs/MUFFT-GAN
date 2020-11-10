import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# -------------------------------------------------
# Reference Equations
# -------------------------------------------------

# Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
# Output width = (Output width + padding width right + padding width left - kernel width) / (stride width) + 1

# -------------------------------------------------
# Discriminator
# -------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        depth_scale_curr = 1
        depth_scale_prev = 1
        for i in range(1, num_layers-2):
            depth_scale_prev = depth_scale_curr
            depth_scale_curr = min(2**i, fcd)
            layers.append(nn.Conv2d(ndf*depth_scale_prev, ndf*depth_scale_curr, 4, 2, 1))
            layers.append(nn.BatchNorm2d(ndf*depth_scale_curr))
            layers.append(nn.LeakyReLU(0.2, True))
        
        depth_scale_prev = depth_scale_curr
        depth_scale_curr = min(2**(num_layers-2), fcd)
        layers.append(nn.Conv2d(ndf*depth_scale_prev, ndf*depth_scale_curr, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf*depth_scale_curr))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(ndf*depth_scale_curr, 1, 4, 1, 1, bias=False))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

    def compute_D_loss(self, reals, fakes, criterion, opt):
        fake_predictions = self.forward(fakes)
        fake_label =  torch.tensor(0, dtype=torch.float, device=opt.device)
        fake_label = fake_label.expand_as(fake_predictions)
        loss_fakes = criterion(fake_predictions, fake_label).mean()

        real_predictions = self.forward(reals)
        real_label =  torch.tensor(1, dtype=torch.float, device=opt.device)
        real_label = real_label.expand_as(real_predictions)
        loss_reals = criterion(real_predictions, real_label).mean()
       
        loss = (loss_fakes + loss_reals) / 2
        fakes_class = fake_predictions.mean().item()
        reals_class = real_predictions.mean().item()

        return loss, fakes_class, reals_class

# -------------------------------------------------
# Generator
# -------------------------------------------------
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        layers=[]
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(opt.img_channels, opt.ngf, kernel_size=7, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(opt.ngf))
        layers.append(nn.ReLU(True))
        
        # 64 x 64 x 32 -> 32 x 32 x 64 -> 16 x 16 x 128 -> 8 x 8 x 256
        for i in range(opt.num_layers-2):
            depth_scale = 2 ** i
            target_depth = opt.ngf*depth_scale*2
            layers.append(nn.Conv2d(opt.ngf*depth_scale, target_depth, kernel_size=4, stride=2, padding=1, bias=1))
            layers.append(nn.BatchNorm2d(target_depth))
            layers.append(nn.ReLU(True))
        
        # 8 x 8 x 256 -> 8 x 8 x 256
        target_depth = 2 ** (opt.num_layers-2)
        for i in range(opt.resnet_blocks):
            layers.append(ResnetBlock(opt, target_depth*opt.ngf))  

        # 8 x 8 x 256 -> 16 x 16 x 128
        for i in range(opt.num_layers-2):
            depth_scale = 2 ** (opt.num_layers-2-i)
            target_depth = int(opt.ngf * depth_scale / 2)
            layers.append(nn.ConvTranspose2d(opt.ngf*depth_scale, target_depth, kernel_size=4, stride=2, padding=1, bias=1))
            layers.append(nn.BatchNorm2d(target_depth))
            layers.append(nn.ReLU(True))

        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(opt.ngf, opt.img_channels, kernel_size=7, stride=1, padding=0))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def compute_G_loss(self, fake_predictions, criterion, opt):
        fake_label =  torch.tensor(0, dtype=torch.float, device=opt.device)
        fake_label = fake_label.expand_as(fake_predictions)
        loss = criterion(fake_predictions, fake_label).mean() * opt.lambda_G
        return loss

# -------------------------------------------------
# Resnet Block -- Generator
# -------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, opt, dimension):
        super(ResnetBlock, self).__init__()
        layers=[]
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(dimension, dimension, kernel_size=3, padding=0, bias=1))
        layers.append(nn.BatchNorm2d(dimension))
        layers.append(nn.ReLU(True))

        # Not sure if we should use dropout, but maybe
        layers.append(nn.Dropout(0.5))
        
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(dimension, dimension, kernel_size=3, padding=0, bias=1))
        layers.append(nn.BatchNorm2d(dimension))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.model(x)

# -------------------------------------------------
# PatchNCE Model
# -------------------------------------------------
class NCE_MLP(nn.module):
    def __init__(self, opt):
        super(NCE_MLP, self).__init__()
        self.opt = opt
        self.init_gain = 0.02

    def forward(self, features, num_patches=16, patch_ids=None):
        return_ids = []
        return_features = []
        for mlp_id, feature in enumerate(features):
            layers = []
            layers.append(nn.Linear(feature.shape[1], opt.image_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(opt.image_channels, opt.image_channels))
            self.mlp = nn.Sequential(*layers)
            self.mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        # May need to initialize this net but not sure, in CUTGAN they call init_net
        




# -------------------------------------------------
# Helper Print Function
# -------------------------------------------------
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x