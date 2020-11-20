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
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        layers = []
        layers.append(nn.Conv2d(opt.img_channels, opt.ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        depth_scale_curr = 1
        depth_scale_prev = 1
        for i in range(1, opt.num_layers-2):
            depth_scale_prev = depth_scale_curr
            depth_scale_curr = min(2**i, opt.fcd)
            layers.append(nn.Conv2d(opt.ndf*depth_scale_prev, opt.ndf*depth_scale_curr, 4, 2, 1))
            layers.append(nn.BatchNorm2d(opt.ndf*depth_scale_curr))
            layers.append(nn.LeakyReLU(0.2, True))
        
        depth_scale_prev = depth_scale_curr
        depth_scale_curr = min(2**(opt.num_layers-2), opt.fcd)
        layers.append(nn.Conv2d(opt.ndf*depth_scale_prev, opt.ndf*depth_scale_curr, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.ndf*depth_scale_curr))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(opt.ndf*depth_scale_curr, 1, 4, 1, 1, bias=False))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

    def compute_D_loss(self, fake_B, real_B, criterion):
        fakes = fake_B.detach()
        fake_predictions = self.forward(fakes)
        fake_labels = self.fake_label.expand_as(fake_predictions)
        loss_fakes = criterion(fake_predictions, fake_labels).mean()
        real_predictions = self.forward(real_B)
        real_labels = self.real_label.expand_as(real_predictions)
        loss_reals = criterion(real_predictions, real_labels).mean()

        loss = (loss_fakes + loss_reals) / 2
        return loss

# -------------------------------------------------
# Generator
# -------------------------------------------------
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt=opt
        self.register_buffer('fake_label', torch.tensor(0.0))
        layers=[]
        # layers.append(PrintLayer())
        layers.append(nn.ReflectionPad2d(3))
        # layers.append(PrintLayer())
        layers.append(nn.Conv2d(opt.img_channels, opt.ngf, kernel_size=7, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(opt.ngf))
        layers.append(nn.ReLU(True))
        # layers.append(PrintLayer())
        
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

    def forward(self, x, layers=[], encode=False):
        feature = x
        features = []
        if len(layers) > 0:
            for layer_id, layer in enumerate(self.model):
                feature = layer(feature)
                if layer_id in layers:
                    features.append(feature)
                if layer_id == layers[-1] and encode:
                    return features
            return feature, features
        else:
            return self.model(x)

    def compute_G_loss(self, netD, netG, netF, fake_B, idt_B, real_A, real_B, criterion, NCE_criterion):
        fakes = fake_B
        fake_predictions = netD(fakes)
        fake_labels = self.fake_label.expand_as(fake_predictions)
        loss_G = criterion(fake_predictions, fake_labels).mean() * self.opt.lambda_G
        print("loss_G:", loss_G.item())
        # print("r_a:", real_A.shape)
        # print("f_b:", fake_B.shape)
        # real_A = torch.squeeze(real_A)
        # fake_B = torch.squeeze(fake_B)
        # print("squeezed_r_a:", real_A.shape)
        # print("squeezed_f_b:", fake_B.shape)
        loss_NCE = self.compute_NCE_loss(netG, netF, real_A, fake_B, NCE_criterion)
        print("loss_NCE:", loss_NCE.item())
        loss_NCE_Y = self.compute_NCE_loss(netG, netF, real_B, idt_B, NCE_criterion)
        print("loss_NCE_Y", loss_NCE_Y.item())
        loss = loss_G + (loss_NCE + loss_NCE_Y)/2
        return loss

    def compute_NCE_loss(self, netG, netF, source, target, NCE_criterion):
        nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        num_nce_layers = len(nce_layers)
        feature_source = netG(source, nce_layers, encode=True)
        feature_target = netG(target, nce_layers, encode=True)

        feature_source_pool, sample_ids = netF(feature_source, self.opt.num_patches, None)
        feature_target_pool, _ = netF(feature_target, self.opt.num_patches, sample_ids)

        print("Source Feature Pool:", feature_source_pool[0].shape)
        print("Target Feature Pool:", feature_target_pool[0].shape)

        nce_loss = 0.0
        for f_tar, f_src, crit, nce_layer in zip(feature_target_pool, feature_source_pool, NCE_criterion, nce_layers):
            loss = crit(f_tar, f_src) * self.opt.lambda_NCE
            nce_loss += loss.mean()

        return nce_loss / num_nce_layers
        
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
class NCE_MLP(nn.Module):
    def __init__(self, opt):
        super(NCE_MLP, self).__init__()
        self.opt = opt
        self.init_gain = 0.02

    def forward(self, features, num_patches=64, patch_ids=None):
        return_features = []
        return_ids = []
        for mlp_id, feature in enumerate(features):
            layers = []
            layers.append(nn.Linear(feature.shape[1], self.opt.img_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.opt.img_channels, self.opt.img_channels))
            mlp = nn.Sequential(*layers)
            mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        # Utilize GPUs, need to fully implement
        self.to(self.opt.device) # Need to create an option for a GPU ID
        # CutGAN had init_weights here, but we do it in train_pyramid

        for feature_id, feature in enumerate(features):
            reshape = feature.permute(0, 2, 3, 1).flatten(1, 2)
            patch_id = torch.randperm(reshape.shape[1], device = features[0].device)
            patch_id = patch_id[:int(min(self.opt.num_patches, patch_id.shape[0]))]
            sample = reshape[:, patch_id, :].flatten(0, 1)

            mlp = getattr(self, 'mlp_%d' % feature_id)
            sample = mlp(sample)

            return_features.append(sample)

        return return_features, return_ids

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