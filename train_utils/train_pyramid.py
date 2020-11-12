import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from config import load_args
from train_utils.train_pyramid import *
import train_utils.util_functions as util_func
import models.base_models as base
import torchvision.transforms as transforms


def train_pyramid(opt, dataloader):
    # Intialize the discriminator and generator for single layer
    netD, netG, netF = init_layer_models()
    
    # [TESTING] Train a single layer 
    train_layer(netD, netG, netF, opt, dataloader)

def init_layer_models(opt):
    netD = base.Discriminator(opt).to(opt.device)
    util_func.init_weights(netD)
    netG = base.Generator(opt).to(opt.device)
    util_func.init_weights(netG)
    netF = base.NCE_MLP(opt).to(opt.device)
    util_func.init_weights(netF)

    return netD, netG, netF

def init_mlp_weights(data, opt):
    reals = torch.cat((data['A'], data['B']), dim=0)
    fake = netG(reals)
    fake_B = fake[:data['A'].size(0)]
    idt_B = fake[data['A'].size(0):]
    netD.compute_D_loss().backward()
    netGcompute_G_loss().backward()
    optimizer_F = torch.optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    return optimizer_F

def train_layer(netD, netG, netF, opt, dataloader):
    # make sure to check criteriongan for calculations with tensors

    # Make loss criterion and optimizers
    mse_criterion = nn.MSELoss().to(opt.device)
    nce_criterion = []
    for nce_layer in [int(i) for i in opt.nce_layers.split(',')]:
        nce_criterion.append(PatchNCELoss(opt).to(self.device))
    self.idt_criterion = torch.nn.L1Loss().to(opt.device)
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizer_F = None

    for epoch in range(opt.num_epochs):
        for i, data in enumerate(dataset):
            if epoch == 0 and i == 0:
                optimizer_F = init_mlp_weights(data, opt)

        # Generate Fakes
        reals = torch.cat((data['A'], data['B']), dim=0)
        fake = netG(reals)
        fake_B = fake[:data['A'].size(0)]
        idt_B = fake[data['A'].size(0):]
        netD.compute_D_loss().backward()
        netGcompute_G_loss().backward()
        optimizer_F = torch.optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

        # update D
        set_requires_grad(netD, True)
        optimizer_D.zero_grad()
        loss_D = netD.compute_D_loss()
        loss_D.backward()
        optimizer_D.step()

        # update G
        set_requires_grad(netD, False)
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        loss_G = netG.compute_G_loss()
        loss_G.backward()
        optimizer_G.step()
        optimizer_F.step()

### Old DCGAN/CUT Code ###
# def train_layer(netD, netG, opt, dataloader):
#     criterion = nn.MSELoss().to(opt.device) #will be patchNCE
#     #criterion = nn.BCELoss().to(opt.device)

#     # Latent vectors
#     fixed_noise = torch.randn(64, opt.nz, 1, 1, device=opt.device)

#     # Context: reals = 1, fakes = 0
#     # Setup Adam optimizers for both G and D
#     optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
#     optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
#     img_list = []
#     G_losses = []
#     D_losses = []
#     print("HERE")
#     #fixed_noise = torch.randn(25, opt.nz, 1, 1, device=opt.device)
#     fixed_noise = torch.randn(25, opt.nz, 1, 1, device=opt.device)

#     for epoch in range(opt.num_epochs):
#         # start_time = time.time()
#         dRec = 0
#         gRec = 0
#         for i, data in enumerate(dataloader, 0):

#             # ----------------------
#             # Update Discriminator
#             # ----------------------
#             netD.zero_grad()
#             # Get real images
#             reals = data[0].to(opt.device)
#             # Generate fake images
#             #noise = torch.randn(reals.size(0), opt.img_channels, opt.img_size, opt.img_size, device=opt.device)
#             noise = torch.randn(reals.size(0), opt.nz, 1, 1, device=opt.device)
#             fakes = netG(noise)
#             # Discriminator forward training pass, compute loss
#             d_loss, fakes_class, reals_class = netD.compute_D_loss(reals, fakes, criterion, opt)
#             dRec = d_loss
#             d_loss.backward(retain_graph=True)
#             optimizerD.step()

#             # ----------------------
#             # Update Generator
#             # ----------------------
#             netG.zero_grad()
#             # Have discriminator predict on fakes
#             fake_predictions = netD(fakes)
#             g_loss = netG.compute_G_loss(fake_predictions, criterion, opt)
#             gRec = g_loss
#             g_loss.backward()
#             optimizerG.step()

#         # if epoch % opt.snapshot_interval == 0:
#         if epoch % 1 == 0:
#             print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
#             % (epoch, opt.num_epochs, dRec.item(), gRec.item(), fakes_class, reals_class))
#             plt.figure()
#             plt.axis("off")
#             plt.title("Training Images")
#             fakes = netG(fixed_noise).detach().cpu()
#             mult =  torch.tensor(0.5, dtype=torch.float)
#             fakes = fakes * mult.expand_as(fakes)
#             fakes = fakes + mult.expand_as(fakes)
#             print(fakes)
#             # trans = transforms.ToPILImage()
#             # fakes = trans(fakes[0])
#             # Nick Test
#             translated_fake = np.einsum('kli->lik',fakes[0])
#             plt.imshow(translated_fake)
#             # Nick Test
#             # plt.imshow(fakes)
#             # Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
#             filename = "generated/generated" + str(epoch) + ".png"
#             plt.savefig(filename)
            
### End Old Code ###