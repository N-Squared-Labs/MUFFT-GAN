import pathlib
import torch
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngf):
        # ---------------------------------
        #     PatchGan Resnet Generator Code
        # ---------------------------------
        super(Generator, self).__init__()
        # Unet encoder
        d = ngf
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

class Discriminator(nn.Module):
    # initializers
    def __init__(self, ngf):
        super(Discriminator, self).__init__()
        d = ngf
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        print(input.shape)
        print(label.shape)
        x = torch.cat([input, label], 1)
        print(x.shape)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


def train():
    # args
    input_dir = "train"
    output_dir = "results"
    batch_size = 8
    ngf = 64
    ndf = 64
    lr = 0.0002
    beta1 = 0.5
    num_epochs = 20
    l1_lambda = 100

    # create output folder and load in input
    cwd = pathlib.Path.cwd()
    tmp = cwd / output_dir
    tmp.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    path = cwd / "../datasets"
    print(path)
    subfolder = input_dir
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]
    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1
        n += 1
    print("dataset length:", n)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)

    # get fixed model snapshot batch
    test = train_loader.__iter__().__next__()[0]
    img_size = test.size()[2]
    fixed_x = test[:, :, :, 0:img_size]
    # fixed_y = test[:, :, :, img_size:]

    netG = Generator(ngf).cuda()
    netD = Discriminator(ndf).cuda()
    netG.apply(weights_init)
    netD.apply(weights_init)
    BCE_loss = nn.BCELoss().cuda()
    L1_loss = nn.L1Loss().cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(num_epochs):
        D_losses = []
        G_losses = []
        for x, _ in train_loader:
            # Update D
            netD.zero_grad()
            y = x[:, :, :, img_size:]
            x = x[:, :, :, 0:img_size]
            x, y = Variable(x.cuda()), Variable(y.cuda())
            # real_pred = netD(x, y).squeeze()
            # D_real_loss = BCE_loss(real_pred, Variable(torch.ones(real_pred.size()).cuda()))

            generated_fakes = netG(x)
            fake_pred = netD(x, generated_fakes).squeeze()
            D_fake_loss = BCE_loss(fake_pred, Variable(torch.zeros(fake_pred.size()).cuda()))
            D_train_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss.backward()
            D_optimizer.step()
            D_losses.append(D_train_loss.data[0])

            # Update G
            netG.zero_grad()
            ## Code uses this but unsure if we need since it's called above? ##
            generated_fakes = netG(x)
            fake_pred = netD(x, generated_fakes).squeeze()
            ## End unsure part ##
            G_train_loss = BCE_loss(fake_pred, Variable(torch.ones(fake_pred.size()).cuda())) + l1_lambda * L1_loss(
                generated_fakes, y)
            G_train_loss.backward()
            G_optimizer.step()
            G_losses.append(G_train_loss.data[0])

        print('[%d/%d] - loss_d: %.3f, loss_g: %.3f' % (
        (epoch+1), num_epochs, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))))
        # Print images
        # if epoch % 1 == 0:
        #     plt.figure()
        #     plt.axis("off")
        #     plt.title("Training Images")
        #     fakes = netG(fixed_noise).detach().cpu()
        #     mult =  torch.tensor(0.5, dtype=torch.float)
        #     fakes = fakes * mult.expand_as(fakes)
        #     fakes = fakes + mult.expand_as(fakes)
        #     print(fakes)
        #     translated_fake = np.einsum('kli->lik',fakes[0])
        #     plt.imshow(translated_fake)
        #     filename = "generated/generated" + str(epoch) + ".png"
        #     plt.savefig(filename)


if __name__ == "__main__":
    train()