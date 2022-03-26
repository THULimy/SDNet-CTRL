import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .dictnet import DictBlock
from config import config as _cfg

cfg = _cfg
class DictConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DictConv2d, self).__init__()
        assert stride <= 2
        self.in_stride = stride
        self.stride = 1 if cfg['MODEL']['POOLING'] else stride

        self.dn = DictBlock(
            in_channels, out_channels, stride=self.stride, kernel_size=kernel_size, padding=padding,
            mu=cfg['MODEL']['MU'], lmbd=cfg['MODEL']['LAMBDA'][0],
            n_dict=cfg['MODEL']['EXPANSION_FACTOR'], non_negative=cfg['MODEL']['NONEGATIVE'],
            n_steps=cfg['MODEL']['NUM_LAYERS'], FISTA=cfg['MODEL']['ISFISTA'], w_norm=cfg['MODEL']['WNORM'],
            padding_mode=cfg['MODEL']['PAD_MODE']
        )

    def forward(self, x):
        out, rc = self.dn(x)

        if cfg['MODEL']['POOLING'] and self.in_stride == 2:
            out = F.max_pool2d(out, kernel_size=2, stride=2)
        # if self.stride == 2:

        return out

    def inverse(self, z):
        z_tilde = z
        with torch.no_grad():

            if cfg['MODEL']['POOLING'] and self.in_stride == 2:
                z_tilde = F.interpolate(z_tilde, scale_factor=2, mode="bilinear")

            x_title = F.conv_transpose2d(
                z_tilde, self.dn.weight,
                bias=None, stride=self.dn.stride, padding=self.dn.padding,
                output_padding=self.dn.conv_transpose_output_padding
            )

        return x_title

# Generator
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu, nz, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            DictConv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf) x 16 x 16
            DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 8 x 8
            DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 4 x 4
            DictConv2d(ndf * 4, nz, 4, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, input):
        return F.normalize(self.main(input))

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()
