from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


dataroot = "../datasets/"
workers = 0
batch_size = 16
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
fcd = 64
num_epochs = 100
lr = 0.0002
beta1 = 0.5
ngpu = 1
kernel_dim = 4
stride = 2
padding = 1
num_layers = 5
img_channels = 3

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        # ---------------------------------
        #     Original Generator Code
        # ---------------------------------
        # super(Generator, self).__init__()
        # self.ngpu = ngpu
        # self.main = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 32 x 32
        #     nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. (nc) x 64 x 64
        # )

        # ---------------------------------
        #     For Loop Generator Code
        # ---------------------------------
        # super(Generator, self).__init__()
        # self.ngpu = ngpu
        # layers = []
        # depth_scale = 2 ** (num_layers-2)
        # layers.append(nn.ConvTranspose2d(nz, ngf * depth_scale, 4, 1, 0, bias=False))
        # layers.append(nn.BatchNorm2d(ngf * depth_scale))
        # layers.append(nn.ReLU(True))

        # for i in range(num_layers-2):
        #     # depth scale for num_layers=5: 8, 4, 2
        #     # first run through ngf*depth_scale = 256, 128, 3, 3
        #     depth_scale = 2 ** (num_layers-2-i)
        #     layers.append(nn.ConvTranspose2d(ngf*depth_scale, int(ngf*depth_scale/2), 
        #                                      4, 2, 1, bias=False))
        #     layers.append(nn.BatchNorm2d(int(ngf*depth_scale / 2)))
        #     layers.append(nn.ReLU(True))
    
        # layers.append(nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False))
        # layers.append(nn.Tanh())

        # self.main = nn.Sequential(*layers)

        # def forward(self, input):
        # return self.main(input)


        # ---------------------------------
        #     Resnet Generator Code
        # ---------------------------------
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

#         layers.append(nn.Dropout(0.5))
        
#         layers.append(nn.ReflectionPad2d(1))
#         layers.append(nn.Conv2d(dimension, dimension, kernel_size=3, padding=0, bias=1))
#         layers.append(nn.BatchNorm2d(dimension))

#         self.model = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return x + self.model(x)

    
        # ---------------------------------
        #     PatchGan Resnet Generator Code
        # ---------------------------------

        super(generator, self).__init__()
        # Unet encoder
        d = 64
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)

        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = F.tanh(d8)
        return o


netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        layers = []
        layers.append(nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        depth_scale_curr = 1
        depth_scale_prev = 1
        for i in range(1, num_layers-2):
            # i = 1: prev=1, curr=2, conv2d=32->64
            depth_scale_prev = depth_scale_curr
            depth_scale_curr = min(2**i, fcd)
            layers.append(nn.Conv2d(ndf*depth_scale_prev, ndf*depth_scale_curr,
                          4, 2, 1))
            layers.append(nn.BatchNorm2d(ndf*depth_scale_curr))
            layers.append(nn.LeakyReLU(0.2, True))
        
        depth_scale_prev = depth_scale_curr
        depth_scale_curr = min(2**(num_layers-2), fcd)
        layers.append(nn.Conv2d(ndf*depth_scale_prev, ndf*depth_scale_curr,
                          4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf*depth_scale_curr))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(ndf*depth_scale_curr, 1, 4, 1, 1, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)

netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print(netD)

# criterion = nn.BCELoss()
BCE_loss = nn.BCELoss().cuda()
L1_loss = nn.L1Loss().cuda()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        label =  torch.tensor(1, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        label = label.expand_as(output)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        label =  torch.tensor(0, dtype=torch.float, device=device)
        label = label.expand_as(output)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1




plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
