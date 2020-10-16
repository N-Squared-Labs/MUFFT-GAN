import torch.optim as optim
from config import load_args
from train_utils.train_pyramid import *
from train_utils.util_functions import *
from models.base_models import *


def train_pyramid(opt, dataloader):
    # Intialize the discriminator and generator
    netD, netG = init_models(opt)
    weights_init(netD)
    weights_init(netG)
    
    # [TESTING] Train a single layer 
    train_layer(netD, netG, opt, dataloader)


def train_layer(netD, netG, opt, dataloader):
    criterion = nn.MSELoss().to(opt.device) #will be patchNCE

    # Latent vectors
    fixed_noise = torch.randn(64, opt.nz, 1, 1, device=opt.device)

    # Context: reals = 1, fakes = 0
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(opt.num_epochs):
        # start_time = time.time()
        for i, data in enumerate(dataloader, 0):

            # ----------------------
            # Update Discriminator
            # ----------------------
            netD.zero_grad()
            # Get real images
            reals = data[0].to(opt.device)
            # Generate fake images
            noise = torch.randn(reals.size(0), opt.nz, 1, 1, device=opt.device)
            fakes = netG(noise)
            # Discriminator forward training pass, compute loss
            d_loss = netD.compute_D_loss(reals, fakes, criterion, opt)
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            # ----------------------
            # Update Generator
            # ----------------------
            netG.zero_grad()
            # Have discriminator predict on fakes
            fake_predictions = netD(fakes)
            g_loss = netG.compute_G_loss(fake_predictions, criterion, opt)
            g_loss.backward()
            optimizerG.step()
            



    # for epoch in range(opt.num_epochs):
    #     # For each batch in the dataloader
    #     for i, data in enumerate(dataloader, 0):
    #
    #         ############################
    #         # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    #         ###########################
    #         ## Train with all-real batch
    #         netD.zero_grad()
    #         # Format batch
    #         real_cpu = data[0].to(device)
    #         b_size = real_cpu.size(0)
    #         label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    #         # Forward pass real batch through D
    #         output = netD(real_cpu).view(-1)
    #         # Calculate loss on all-real batch
    #         errD_real = criterion(output, label)
    #         # Calculate gradients for D in backward pass
    #         errD_real.backward()
    #         D_x = output.mean().item()
    #
    #         ## Train with all-fake batch
    #         # Generate batch of latent vectors
    #         noise = torch.randn(b_size, nz, 1, 1, device=device)
    #         # Generate fake image batch with G
    #         fake = netG(noise)
    #         label.fill_(fake_label)
    #         # Classify all fake batch with D
    #         output = netD(fake.detach()).view(-1)
    #         # Calculate D's loss on the all-fake batch
    #         errD_fake = criterion(output, label)
    #         # Calculate the gradients for this batch
    #         errD_fake.backward()
    #         D_G_z1 = output.mean().item()
    #         # Add the gradients from the all-real and all-fake batches
    #         errD = errD_real + errD_fake
    #         # Update D
    #         optimizerD.step()
    #
    #         ############################
    #         # (2) Update G network: maximize log(D(G(z)))
    #         ###########################
    #         netG.zero_grad()
    #         label.fill_(real_label)  # fake labels are real for generator cost
    #         # Since we just updated D, perform another forward pass of all-fake batch through D
    #         output = netD(fake).view(-1)
    #         # Calculate G's loss based on this output
    #         errG = criterion(output, label)
    #         # Calculate gradients for G
    #         errG.backward()
    #         D_G_z2 = output.mean().item()
    #         # Update G
    #         optimizerG.step()
    #
    #         # Output training stats
    #         if i % 50 == 0:
    #             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
    #                   % (epoch, num_epochs, i, len(dataloader),
    #                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    #
    #         # Save Losses for plotting later
    #         G_losses.append(errG.item())
    #         D_losses.append(errD.item())
    #
    #         # Check how the generator is doing by saving G's output on fixed_noise
    #         if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
    #             with torch.no_grad():
    #                 fake = netG(fixed_noise).detach().cpu()
    #             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    #
    #         iters += 1
            

def init_models(opt):
    netD = Discriminator(opt).to(opt.device)
    netG = Generator(opt).to(opt.device)
    return netD, netG

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)