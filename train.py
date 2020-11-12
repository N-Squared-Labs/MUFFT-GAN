import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config import load_args
# from train_utils.train_pyramid import *
from train_utils.util_functions import *

if __name__ == '__main__':
    # Load and Manipulate Config Parameters
    arg_parser = load_args()
    opt = arg_parser.parse_args()
    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # opt = init_config(opt)

    # dataset = dset.ImageFolder(root=opt.input_dir,
    #                            transform=transforms.Compose([
    #                            transforms.Resize(opt.img_size),
    #                            transforms.CenterCrop(opt.img_size),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                        ]))
    # dataloader = torch.utils.data.DataLoader(dataset, opt.batch_size, shuffle=True)
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(opt.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.savefig("test.png")

    dataloader = UnpairedDataset(opt)
    dataset_size = len(dataloader)
    print(dataset_size)
    real_batch = next(iter(dataloader))
    print(real_batch)
    for i, data in enumerate(dataloader):
        batch_size = data["A"].size(0)
        print(batch_size)
        print(data["A"][:batch_size].size(0))
        print(data["A"].shape)
        break
    
    # Create Training Variables and call Training
    # real_img = read_input(opt)
    # gens = []
    # noise = []
    # reals = []
    # noise_amps = []
    # train_pyramid(opt, gens, noise, reals, noise_amps)
    # train_pyramid(opt, dataloader)
