# -----------------------------------------------------
# Discriminator Models
# -----------------------------------------------------


# -----------------------
# Hard-coded implementation DC Gan Discriminator
# -----------------------
# layers = []
# # layers.append(PrintLayer())
# layers.append(nn.Conv2d(opt.img_channels, opt.ndf, opt.kernel_dim, opt.stride, opt.padding))
# layers.append(nn.LeakyReLU(0.2, True))

# layers.append(nn.Conv2d(opt.ndf, opt.ndf*2, opt.kernel_dim, opt.stride, opt.padding))
# # layers.append(PrintLayer())
# layers.append(nn.BatchNorm2d(opt.ndf*2))
# layers.append(nn.LeakyReLU(0.2, True))

# layers.append(nn.Conv2d(opt.ndf*2, opt.ndf*4, opt.kernel_dim, opt.stride, opt.padding))
# # layers.append(PrintLayer())
# layers.append(nn.BatchNorm2d(opt.ndf*4))
# layers.append(nn.LeakyReLU(0.2, True))

# layers.append(nn.Conv2d(opt.ndf*4, opt.ndf*8, opt.kernel_dim, opt.stride, opt.padding))
# # layers.append(PrintLayer())
# layers.append(nn.BatchNorm2d(opt.ndf*8))
# layers.append(nn.LeakyReLU(0.2, True))

# layers.append(nn.Conv2d(opt.ndf*8, 1, opt.kernel_dim, 1, 0))
# # layers.append(PrintLayer())



# -----------------------
# For Loop CutGan Discriminator
# -----------------------
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

        # self.model = nn.Sequential(*layers)














# -----------------------------------------------------
# Generator Models
# -----------------------------------------------------


# -----------------------
# CutGAN Generator Model For Loop
# -----------------------
# PatchGAN Generator simple implementation
# class Generator(nn.Module):
#     def __init__(self, opt):
#         super(Generator, self).__init__()
#         layers = []

#         # input 3x100x1x1 -> output 32
#         depth_scale = 2 ** (opt.num_layers-2)
#         layers.append(nn.ConvTranspose2d(opt.nz, opt.ngf * depth_scale, opt.kernel_dim, 1, 0))
#         layers.append(nn.ReLU(True))
        
        
#         for i in range(opt.num_layers-2):
#             # depth scale for num_layers=5: 8, 4, 2
#             # first run through opt.ngf*depth_scale = 256, 128, 3, 3
#             depth_scale = 2 ** (opt.num_layers-2-i)
#             layers.append(nn.ConvTranspose2d(opt.ngf*depth_scale, int(opt.ngf*depth_scale/2), 
#                                              opt.kernel_dim, opt.stride, opt.padding))
#             layers.append(nn.BatchNorm2d(int(opt.ngf*depth_scale / 2)))
#             layers.append(nn.ReLU(True))
    
        
        
#         layers.append(nn.ConvTranspose2d(opt.ngf, opt.img_channels, opt.kernel_dim, opt.stride, opt.padding))
#         layers.append(nn.Tanh())
        
#         self.model = nn.Sequential(*layers)

#     def compute_G_loss(self, fake_predictions, criterion, opt):
#         fake_label =  torch.tensor(0, dtype=torch.float, device=opt.device)
#         fake_label = fake_label.expand_as(fake_predictions)

#         loss = criterion(fake_predictions, fake_label).mean() * opt.lambda_G
#         return loss

#     def forward(self, x):
#         return self.model(x)