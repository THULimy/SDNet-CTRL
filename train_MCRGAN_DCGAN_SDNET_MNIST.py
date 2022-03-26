### Import
# from __future__ import print_function
import argparse
import random # to set the python random seed
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import torch.optim as optim
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler
# Ignore excessive warnings
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)
import torchvision.transforms.functional as FF
from utils import *
from losses import *

from torch.utils.tensorboard import SummaryWriter

# Set random seeds and deterministic pytorch for reproducibility
manualSeed = 100
random.seed(manualSeed)  # python random seed
torch.manual_seed(manualSeed)  # pytorch random seed
np.random.seed(manualSeed)  # numpy random seed
torch.backends.cudnn.deterministic = True

### Train
def train(netG, netD, device, dataloader, optimizerG, optimizerD, schedulerG, schedulerD, criterion, num_epochs, writer, model_dir, n_iter_dis):
    netG.train()
    netD.train()
    iters = 0
    for epoch in range(1,num_epochs+1):
        for i, (X, cls_label) in enumerate(dataloader):
            # update_stepsize
            # only netD have dictnet for now, may include netG in the future
            if epoch>1:
                netD.module.update_stepsize() if isinstance(netD, torch.nn.DataParallel) else netD.update_stepsize()

            #*****
            # Update Discriminator/Encoder
            #*****
            for _ in range(n_iter_dis):
                netG.zero_grad()
                
                X = X.to(device)
                b_size = X.size(0)

                Z = netD(X)
                X_bar = netG(Z.reshape(b_size,-1,1,1))
                Z_bar = netD(X_bar.detach())
                
                errD, detailed_loss_D = criterion(Z, Z_bar, cls_label)
                errD = errD
                
                optimizerD.zero_grad()
                errD.backward()
                optimizerD.step()

            #*****
            # Update Generator/Decoder
            #*****
            netG.zero_grad()

            Z = netD(X)
            X_bar = netG(Z.reshape(b_size,-1,1,1))
            Z_bar = netD(X_bar)
            
            errG, detailed_loss_G = criterion(Z, Z_bar, cls_label)
            errG = (-1) * errG
            
            optimizerG.zero_grad()
            errG.backward()
            optimizerG.step()

            # Check how the generator is doing by saving G's output
            if (iters % 500 == 0) or (i == len(dataloader)-1):
                
                print(f"iter:{iters}")
                print(f"errD:{errD}")
                print(f"ErrG:{errG}")

                save_fig(vutils.make_grid(X_bar[:64].detach().cpu(), padding=2, normalize=True), epoch, iters, model_dir)

            iters += 1

            # write to logs
            writer.add_scalar('Loss/errD', errD.item(), iters)
            writer.add_scalar('Loss/errG', errG.item(), iters)

            if criterion.train_mode == 'multi':
                writer.add_scalar('Loss/errD-loss-z', detailed_loss_D[0].item(), iters)
                writer.add_scalar('Loss/errD-loss-h', detailed_loss_D[1].item(), iters)
                writer.add_scalar('Loss/errD-loss-class', detailed_loss_D[2].item(), iters)
                writer.add_scalar('Loss/errG-loss-z', - detailed_loss_G[0].item(), iters)
                writer.add_scalar('Loss/errG-loss-h', - detailed_loss_G[1].item(), iters)
                writer.add_scalar('Loss/errG-loss-class', - detailed_loss_G[2].item(), iters)

        schedulerD.step()
        schedulerG.step()
        print("-", schedulerD.last_epoch, schedulerD.get_last_lr(), optimizerD.param_groups[0]['lr'])

        writer.add_images('Images/X', normalize_back(X[:16].cpu(), mean=0.5, std=0.5), epoch)
        writer.add_images('Images/X_bar', normalize_back(X_bar[:16].cpu(), mean=0.5, std=0.5), epoch)

        # save models
        if epoch % 5 == 0: 
            torch.save(netG.state_dict(), os.path.join(model_dir, 'checkpoints', 'model-G-epoch{}.pt'.format(epoch)))
            torch.save(netD.state_dict(), os.path.join(model_dir, 'checkpoints', 'model-D-epoch{}.pt'.format(epoch)))

def main():
    ### Hyper-Parameters
    dataset = 'MNIST'
    arch = 'DCGAN-SDNET' 
    workers = 2
    batch_size = 2048
    image_size = 32
    nc = 1 if 'MNIST' in dataset else 3
    nz = 128
    ngf = 64
    ndf = 64
    num_epochs = 500
    lr = 0.0001
    n_iter_dis = 1
    ngpu = 1
    gam1 = 1.
    gam2 = 1.
    eps = 0.5
    num_class = 10
    train_mode = 'multi'
    scheduler_steps = [100, 200]

    model_dir = os.path.join('./saved_models',
               'MCRGAN_{}_arch{}_data{}_nz{}_epo{}_bs{}_lr{}_niter{}_schestep{}_gam1{}_gam2{}_eps{}_optim{}_seed{}'.format(
                    train_mode, arch, dataset, nz, num_epochs, batch_size, lr, n_iter_dis, scheduler_steps, gam1, gam2, eps, 'Adam', manualSeed))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir, 'checkpoints'))
        os.makedirs(os.path.join(model_dir, 'figures'))
        os.makedirs(os.path.join(model_dir, 'plabels'))
        os.makedirs(os.path.join(model_dir, 'images'))

    writer = SummaryWriter(log_dir=model_dir+'/log/')

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Load the dataset
    if dataset == 'CIFAR10':
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True,
                                                shuffle=True, num_workers=workers)
    elif dataset == 'MNIST':
        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])
        trainset = datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True,
                                                shuffle=True, num_workers=workers)

    elif dataset == 'TMNIST':
        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    MyAffineTransform(choices=[[0, 1], [0, 1.5], [0, 0.5], [-45, 1], [45, 1]]),
                    transforms.Normalize(0.5, 0.5)])
        trainset = datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True,
                                                shuffle=True, num_workers=workers)

    # Create the generator
    from models.sdnet_DCGAN import Generator, Discriminator

    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netD = Discriminator(ngpu, nz, ndf, nc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Initialize cri
    criterionMCRGAN = MCRGANloss(gam1=gam1, gam2=gam2, eps=eps, num_class=num_class, mode=train_mode)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.0, 0.9))

    schedulerD = lr_scheduler.MultiStepLR(optimizerD, scheduler_steps, gamma=0.1)
    schedulerG = lr_scheduler.MultiStepLR(optimizerG, scheduler_steps, gamma=0.1)


    # start training.
    train(netG, netD, device, trainloader, optimizerG, optimizerD, schedulerG, schedulerD, criterionMCRGAN, num_epochs, writer, model_dir, n_iter_dis)
        
if __name__ == '__main__':
    main()














