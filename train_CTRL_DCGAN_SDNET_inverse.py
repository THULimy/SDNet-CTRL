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
from utils import weights_init, save_fig, MyAffineTransform, normalize_back
from losses import MaximalCodingRateReduction, MCRGANloss
from config import update_config
from config import config as cfg

import time 
from torch.utils.tensorboard import SummaryWriter

# Set random seeds and deterministic pytorch for reproducibility
manualSeed = 100
random.seed(manualSeed)  # python random seed
torch.manual_seed(manualSeed)  # pytorch random seed
np.random.seed(manualSeed)  # numpy random seed
torch.backends.cudnn.deterministic = True

### save config
def _to_yaml(obj, filename=None, default_flow_style=False,
             encoding="utf-8", errors="strict",
             **yaml_kwargs):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding=encoding, errors=errors) as f:
        obj.dump(stream=f, default_flow_style=default_flow_style, **yaml_kwargs)

### Train
def train(model, device, dataloader, optimizer, scheduler, criterion, num_epochs, writer, model_dir, n_iter_dis):
    model.train()
    iters = 0
    for epoch in range(1,num_epochs+1):
        start_time = time.time()
        for i, (X, cls_label) in enumerate(dataloader):
            # update_stepsize
            # only netD have dictnet for now, may include netG in the future
            model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()
            
            #*****
            # Update Discriminator/Encoder
            #*****

            for _ in range(n_iter_dis):                
                X = X.to(device)

                # Z = model.module.f_forward(X)
                # X_bar = model.module.g_forward(Z.reshape(len(Z),-1,1,1))
                # Z_bar = model.module.f_forward(X_bar.detach())
                Z, X_bar = model(X)
                Z_bar, _ = model(X_bar.detach())
                
                errD, detailed_loss_D = criterion(Z, Z_bar, cls_label)
                errD = errD
                
                optimizer.zero_grad()
                errD.backward()
                optimizer.step()

            #*****
            # Update Generator/Decoder
            #*****
            model.zero_grad()

            # Z = model.module.f_forward(X)
            # X_bar = model.module.g_forward(Z.reshape(len(Z),-1,1,1))
            # Z_bar = model.module.f_forward(X_bar)
            Z, X_bar = model(X)
            Z_bar, _ = model(X_bar.detach())
            
            errG, detailed_loss_G = criterion(Z, Z_bar, cls_label)
            errG = (-1) * errG
            
            optimizer.zero_grad()
            errG.backward()
            optimizer.step()

            # Check how the generator is doing by saving G's output
            if (iters % 500 == 0) or (i == len(dataloader)-1):
                
                print(f"iter:{iters}")
                print(f"errD:{errD}")
                print(f"ErrG:{errG}")
                save_fig(vutils.make_grid(X[:64].detach().cpu(), padding=2, normalize=True), epoch, iters, model_dir, tail='input')
                save_fig(vutils.make_grid(X_bar[:64].detach().cpu(), padding=2, normalize=True), epoch, iters, model_dir, tail='transcription')

            iters += 1

            # write to logs
            writer.add_scalar('Loss/errD', errD.item(), iters)
            writer.add_scalar('Loss/errG', errG.item(), iters)

            if criterion.train_mode == 'multi':
                writer.add_scalar('Loss/errD-loss-z', - detailed_loss_D[0].item(), iters)
                writer.add_scalar('Loss/errD-loss-h', - detailed_loss_D[1].item(), iters)
                writer.add_scalar('Loss/errD-loss-class', - detailed_loss_D[2].item(), iters)
                writer.add_scalar('Loss/errG-loss-z', - detailed_loss_G[0].item(), iters)
                writer.add_scalar('Loss/errG-loss-h', - detailed_loss_G[1].item(), iters)
                writer.add_scalar('Loss/errG-loss-class', - detailed_loss_G[2].item(), iters)
            
        end_time = time.time()
        time_diff = end_time - start_time
        print(i, time_diff)

        if scheduler:
            scheduler.step()
            print("-", scheduler.last_epoch, scheduler.get_last_lr(), optimizer.param_groups[0]['lr'])

        writer.add_images('Images/X', normalize_back(X[:16].cpu(), mean=0.5, std=0.5), epoch)
        writer.add_images('Images/X_bar', normalize_back(X_bar[:16].cpu(), mean=0.5, std=0.5), epoch)

        # save models
        if epoch % 5 == 0: 
            torch.save(model.module.state_dict(), os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch)))

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='should add the .yaml file',
                        default=None,
                        type=str,
                        )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    if args.cfg or args.opts:
        update_config(cfg, args)

def main():
    parse_args()
    print(cfg)
    dataset = cfg['TRAIN']['DATASET']
    arch = cfg['TRAIN']['ARCH']
    batch_size = cfg['TRAIN']['BATCH_SIZE']
    nz = cfg['TRAIN']['NZ']
    ngf = cfg['TRAIN']['NGF']
    ndf = cfg['TRAIN']['NGF']
    num_epochs = cfg['TRAIN']['EPOCHS']
    lr = cfg['TRAIN']['LR']
    n_iter_dis = cfg['TRAIN']['N_ITER_DIS']
    gam1 = cfg['TRAIN']['GAM1']
    gam2 = cfg['TRAIN']['GAM2']
    eps = cfg['TRAIN']['EPS']
    train_mode = cfg['TRAIN']['MODE']
    scheduler_factor = cfg['TRAIN']['LR_SCHE_FACTOR']
    scheduler_steps = cfg['TRAIN']['LR_SCHE_STEP']
    adam_beta1 = cfg['TRAIN']['ADAM_BETA1']
    adam_beta2 = cfg['TRAIN']['ADAM_BETA2']
    lmda = cfg['MODEL']['LAMBDA'][0]
    num_layer = cfg['MODEL']['NUM_LAYERS']

    workers = 2
    num_class = 10
    image_size = 32
    nc = 1 if 'MNIST' in dataset else 3

    model_dir = os.path.join('./saved_models',
               'CTRL_{}_DCGAN-SDNET-arch{}_data{}_lmda{}_nlayer{}_nz{}_epo{}_bs{}_lr{}_niter{}_schestep{}_gam1{}_gam2{}_eps{}_optim{}_beta1{}_beta2{}_seed{}'.format(
                    train_mode, arch, dataset, lmda, num_layer, nz, num_epochs, batch_size, lr, n_iter_dis, scheduler_steps, gam1, gam2, eps, 'Adam', adam_beta1, adam_beta2, manualSeed))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir, 'checkpoints'))
        os.makedirs(os.path.join(model_dir, 'figures'))
        os.makedirs(os.path.join(model_dir, 'plabels'))
        os.makedirs(os.path.join(model_dir, 'images'))

    _to_yaml(cfg, os.path.join(model_dir, 'config.yaml'))
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
    from models.sdnet_DCGAN import Generator, Discriminator, InverseNet2, InverseNet4
    
    if arch == 'INVERSE2':
        model = InverseNet2(nz, ngf, nc).to(device)
    elif arch == 'INVERSE4':
        model = InverseNet4(nz, ngf, nc).to(device)

    # Handle multi-gpu if desired
    print(torch.cuda.device_count())
    model = nn.DataParallel(model)

    # init model once
    with torch.no_grad():
        print("====================")
        inputx = torch.zeros([100, nc, image_size, image_size]).cuda()
        # print(inputx)
        _ = model.module.f_forward(inputx)
        print("====================")

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    model.apply(weights_init)

    # Initialize cri
    criterionMCRGAN = MCRGANloss(gam1=gam1, gam2=gam2, eps=eps, num_class=num_class, mode=train_mode)

    # Setup Adam optimizers for both G and D
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2))

    if not scheduler_steps[0]:
        scheduler = None
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, scheduler_steps, gamma=scheduler_factor)


    # start training.
    train(model, device, trainloader, optimizer, scheduler, criterionMCRGAN, num_epochs, writer, model_dir, n_iter_dis)
        
if __name__ == '__main__':
    main()














