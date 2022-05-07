import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .dictnet import DictBlock
from config import config as cfg

class DictConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, lmbd=None):
        super(DictConv2d, self).__init__()
        assert stride <= 2
        self.in_stride = stride
        self.stride = 1 if cfg['MODEL']['POOLING'] else stride
        # print(cfg)
        self.dn = DictBlock(
            in_channels, out_channels, stride=self.stride, kernel_size=kernel_size, padding=padding,
            mu=cfg['MODEL']['MU'], lmbd=lmbd if lmbd else cfg['MODEL']['LAMBDA'][0],
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

        if cfg['MODEL']['POOLING'] and self.in_stride == 2:
            z_tilde = F.interpolate(z_tilde, scale_factor=2, mode="bilinear")

        x_tilde = F.conv_transpose2d(
            z_tilde, self.dn.weight,
            bias=None, stride=self.dn.stride, padding=self.dn.padding,
            output_padding=self.dn.conv_transpose_output_padding
        )

        return x_tilde

    # def inverse(self, z):
    #     z_tilde = z
    #     with torch.no_grad():

    #         if cfg['MODEL']['POOLING'] and self.in_stride == 2:
    #             z_tilde = F.interpolate(z_tilde, scale_factor=2, mode="bilinear")

    #         x_title = F.conv_transpose2d(
    #             z_tilde, self.dn.weight,
    #             bias=None, stride=self.dn.stride, padding=self.dn.padding,
    #             output_padding=self.dn.conv_transpose_output_padding
    #         )

    #     return x_title


# Generator
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
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

class Generator_for_STL(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator_for_STL, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 3 x 3
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 6 x 6
            nn.ConvTranspose2d( ngf * 8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 12 x 12
            nn.ConvTranspose2d( ngf * 4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 24 x 24
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 48 x 48
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 96 x 96
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nz, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, nz, 4, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, input, norm=True):
        out = self.main(input)
        if norm:
            out = F.normalize(out)
        return out

class Discriminator_for_STL(nn.Module):
    def __init__(self, nz, ndf, nc):
        super(Discriminator_for_STL, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf) x 48 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 24 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 3 x 3
            nn.Conv2d(ndf * 16, nz, 3, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, input, norm=True):
        out = self.main(input)
        if norm:
            out = F.normalize(out)
        return out

# Discriminator
class Discriminator_SDNet_for_STL(nn.Module):
    def __init__(self, nz, ndf, nc):
        super(Discriminator_SDNet_for_STL, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            DictConv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf) x 48 x 48
            DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 24 x 24
            DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 12 x 12
            DictConv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 6 x 6
            DictConv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 3 x 3
            DictConv2d(ndf * 16, nz, 3, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, input, norm=True):
        out = self.main(input)
        if norm:
            out = F.normalize(out)
        return out

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

# Discriminator
class Discriminator_SDNet(nn.Module):
    def __init__(self, nz, ndf, nc):
        super(Discriminator_SDNet, self).__init__()
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

    def forward(self, input, norm=True):
        out = self.main(input)
        if norm:
            out = F.normalize(out)
        return out

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()


class BatchNorm(nn.BatchNorm2d):
    def inverse(self, z):

        bias = self.bias[None, :, None, None]
        weight = self.weight[None, :, None, None]
        if self.training:
            # for debug 
            # here I guess we should use mean of the batch?
            mean = self.running_mean[None, :, None, None]
            var = self.running_var[None, :, None, None]
            # raise ValueError()

        else:
            mean = self.running_mean[None, :, None, None]
            var = self.running_var[None, :, None, None]

        mid = (z - bias) / weight
        x_title = mid * (var ** 0.5) + mean

        return x_title

class LeakyReLU(nn.LeakyReLU):
    def inverse(self, z):
        inv_negative_slope = self.negative_slope
        inv_inplace = self.inplace
        x_title = F.leaky_relu(z, 1.0/inv_negative_slope, inv_inplace)
        return x_title

class Conv2d(nn.Conv2d):
    def inverse(self, x):
        weight = self.weight
        x_title = F.conv_transpose2d(x, weight, stride=self.stride, padding=self.padding)
        #print(x.shape, weight.shape, x_title.shape)
        return x_title

class InverseNet(nn.Module):

    def __init__(self, nz, ndf, nc):
        super(InverseNet, self).__init__()

        # f layers
        self.conv1 = DictConv2d(nc, ndf, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.conv3 = DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)
        self.conv4 = DictConv2d(ndf * 4, nz, 4, 1, 0, bias=False)
        self.flatten = nn.Flatten()

        # g layers
        self.g_bn1 = nn.BatchNorm2d(ndf * 4)
        self.g_relu1 = nn.ReLU(True)
        self.g_bn2 = nn.BatchNorm2d(ndf * 2)
        self.g_relu2 = nn.ReLU(True)
        self.g_bn3 = nn.BatchNorm2d(ndf * 1)
        self.g_relu3 = nn.ReLU(True)
        self.g_tanh = nn.Tanh()

    def _forward(self, x, norm=True):

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.flatten(self.conv4(out3))
        if norm:
            out4 = F.normalize(out4)
        return out4

    def _inverse(self, z):
        out4 = z #z.view(len(z),-1,1,1)
        out3 = self.g_relu1(self.g_bn1(self.conv4.inverse(out4)))
        out2 = self.g_relu2(self.g_bn2(self.conv3.inverse(out3)))
        out1 = self.g_relu3(self.g_bn3(self.conv2.inverse(out2)))
        x_hat = self.g_tanh(self.conv1.inverse(out1))

        return x_hat

    def forward(self, x, norm=True):
        z = self._forward(x, norm)
        x_hat = self._inverse(z.reshape(len(z),-1,1,1))
        return z, x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

class InverseNet_for_STL(nn.Module):
    def __init__(self, nz, ndf, nc):
        super(InverseNet_for_STL, self).__init__()

        # f layers
        self.conv1 = DictConv2d(nc, ndf, 4, 2, 1, bias=False) ## 96*96 -> 48*48
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False) ## 48*48 -> 24*24
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.conv3 = DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False) ## 24*24 -> 12*12
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)
        self.conv4 = DictConv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False) ## 12*12 -> 6*6
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=False)
        self.conv5 = DictConv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False) ## 6*6 -> 3*3
        self.bn5 = nn.BatchNorm2d(ndf * 16)
        self.relu5 = nn.LeakyReLU(0.2, inplace=False)
        self.conv6 = DictConv2d(ndf * 16, nz, 3, 1, 0, bias=False) ## 3*3 -> 1*1
        self.flatten = nn.Flatten()

        # g layers
        self.g_bn1 = nn.BatchNorm2d(ndf * 16)
        self.g_relu1 = nn.ReLU(True)
        self.g_bn2 = nn.BatchNorm2d(ndf * 8)
        self.g_relu2 = nn.ReLU(True)
        self.g_bn3 = nn.BatchNorm2d(ndf * 4)
        self.g_relu3 = nn.ReLU(True)
        self.g_bn4 = nn.BatchNorm2d(ndf * 2)
        self.g_relu4 = nn.ReLU(True)
        self.g_bn5 = nn.BatchNorm2d(ndf * 1)
        self.g_relu5 = nn.ReLU(True)
        self.g_tanh = nn.Tanh()

    def _forward(self, x, norm=True):

        out1 = self.relu1(self.conv1(x)) #48*48
        out2 = self.relu2(self.bn2(self.conv2(out1))) #24*24
        out3 = self.relu3(self.bn3(self.conv3(out2))) #12*12
        out4 = self.relu4(self.bn4(self.conv4(out3))) #6*6
        out5 = self.relu5(self.bn5(self.conv5(out4))) #3*3

        out6 = self.flatten(self.conv6(out5)) 
        if norm:
            out6 = F.normalize(out6)
        return out6

    def _inverse(self, z):
        out6 = z #z.view(len(z),-1,1,1)
        out5 = self.g_relu1(self.g_bn1(self.conv6.inverse(out6)))
        out4 = self.g_relu2(self.g_bn2(self.conv5.inverse(out5)))
        out3 = self.g_relu3(self.g_bn3(self.conv4.inverse(out4)))
        out2 = self.g_relu4(self.g_bn4(self.conv3.inverse(out3)))
        out1 = self.g_relu5(self.g_bn5(self.conv2.inverse(out2)))
        x_hat = self.g_tanh(self.conv1.inverse(out1))

        return x_hat

    def forward(self, x, norm=True):
        z = self._forward(x, norm)
        x_hat = self._inverse(z.reshape(len(z),-1,1,1))
        return z, x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

class InverseNet_Baseline_for_STL(nn.Module):
    def __init__(self, nz, ndf, nc):
        super(InverseNet_Baseline_for_STL, self).__init__()

        # f layers
        self.conv1 = Conv2d(nc, ndf, 4, 2, 1, bias=False) ## 96*96 -> 48*48
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False) ## 48*48 -> 24*24
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.conv3 = Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False) ## 24*24 -> 12*12
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)
        self.conv4 = Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False) ## 12*12 -> 6*6
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=False)
        self.conv5 = Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False) ## 6*6 -> 3*3
        self.bn5 = nn.BatchNorm2d(ndf * 16)
        self.relu5 = nn.LeakyReLU(0.2, inplace=False)
        self.conv6 = Conv2d(ndf * 16, nz, 3, 1, 0, bias=False) ## 3*3 -> 1*1
        self.flatten = nn.Flatten()

        #print(self.conv1.weight.shape,self.conv2.weight.shape,self.conv3.weight.shape,self.conv4.weight.shape,)
        # g layers
        self.g_bn1 = nn.BatchNorm2d(ndf * 16)
        self.g_relu1 = nn.ReLU(True)
        self.g_bn2 = nn.BatchNorm2d(ndf * 8)
        self.g_relu2 = nn.ReLU(True)
        self.g_bn3 = nn.BatchNorm2d(ndf * 4)
        self.g_relu3 = nn.ReLU(True)
        self.g_bn4 = nn.BatchNorm2d(ndf * 2)
        self.g_relu4 = nn.ReLU(True)
        self.g_bn5 = nn.BatchNorm2d(ndf * 1)
        self.g_relu5 = nn.ReLU(True)
        self.g_tanh = nn.Tanh()

    def _forward(self, x, norm=True):

        out1 = self.relu1(self.conv1(x)) #48*48
        out2 = self.relu2(self.bn2(self.conv2(out1))) #24*24
        out3 = self.relu3(self.bn3(self.conv3(out2))) #12*12
        out4 = self.relu4(self.bn4(self.conv4(out3))) #6*6
        out5 = self.relu5(self.bn5(self.conv5(out4))) #3*3

        out6 = self.flatten(self.conv6(out5)) 
        if norm:
            out6 = F.normalize(out6)
        return out6

    def _inverse(self, z):
        out6 = z #z.view(len(z),-1,1,1)
        out5 = self.g_relu1(self.g_bn1(self.conv6.inverse(out6)))
        out4 = self.g_relu2(self.g_bn2(self.conv5.inverse(out5)))
        out3 = self.g_relu3(self.g_bn3(self.conv4.inverse(out4)))
        out2 = self.g_relu4(self.g_bn4(self.conv3.inverse(out3)))
        out1 = self.g_relu5(self.g_bn5(self.conv2.inverse(out2)))
        x_hat = self.g_tanh(self.conv1.inverse(out1))
        return x_hat

    def forward(self, x, norm=True):
        z = self._forward(x, norm)
        x_hat = self._inverse(z.reshape(len(z),-1,1,1))
        return z, x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

class InverseNet_last_layer_sparse(nn.Module):

    def __init__(self, nz, ndf, nc, last_lmbd=None):
        super(InverseNet_last_layer_sparse, self).__init__()

        # f layers
        self.conv1 = DictConv2d(nc, ndf, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.conv3 = DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)
        self.conv4 = DictConv2d(ndf * 4, nz, 4, 1, 0, bias=False, lmbd=last_lmbd)
        self.flatten = nn.Flatten()

        # g layers
        self.g_bn1 = nn.BatchNorm2d(ndf * 4)
        self.g_relu1 = nn.ReLU(True)
        self.g_bn2 = nn.BatchNorm2d(ndf * 2)
        self.g_relu2 = nn.ReLU(True)
        self.g_bn3 = nn.BatchNorm2d(ndf * 1)
        self.g_relu3 = nn.ReLU(True)
        self.g_tanh = nn.Tanh()

    def _forward(self, x, norm=True):

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.flatten(self.conv4(out3))
        if norm:
            out4 = F.normalize(out4)
        return out4

    def _inverse(self, z):
        out4 = z #z.view(len(z),-1,1,1)
        out3 = self.g_relu1(self.g_bn1(self.conv4.inverse(out4)))
        out2 = self.g_relu2(self.g_bn2(self.conv3.inverse(out3)))
        out1 = self.g_relu3(self.g_bn3(self.conv2.inverse(out2)))
        x_hat = self.g_tanh(self.conv1.inverse(out1))

        return x_hat

    def forward(self, x, norm=True):
        z = self._forward(x, norm)
        x_hat = self._inverse(z.reshape(len(z),-1,1,1))
        return z, x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()


class InverseNet_Baseline(nn.Module):

    def __init__(self, nz, ndf, nc):
        super(InverseNet_Baseline, self).__init__()

        # f layers
        self.conv1 = Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.conv3 = Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)
        self.conv4 = Conv2d(ndf * 4, nz, 4, 1, 0, bias=False)
        self.flatten = nn.Flatten()

        #print(self.conv1.weight.shape,self.conv2.weight.shape,self.conv3.weight.shape,self.conv4.weight.shape,)
        # g layers
        self.g_bn1 = nn.BatchNorm2d(ndf * 4)
        self.g_relu1 = nn.ReLU(True)
        self.g_bn2 = nn.BatchNorm2d(ndf * 2)
        self.g_relu2 = nn.ReLU(True)
        self.g_bn3 = nn.BatchNorm2d(ndf * 1)
        self.g_relu3 = nn.ReLU(True)
        self.g_tanh = nn.Tanh()

    def _forward(self, x, norm=True):

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.flatten(self.conv4(out3))
        if norm:
            out4 = F.normalize(out4)
        return out4

    def _inverse(self, z):
        out4 = z.reshape(len(z),-1,1,1)
        out3 = self.g_relu1(self.g_bn1(self.conv4.inverse(out4)))
        out2 = self.g_relu2(self.g_bn2(self.conv3.inverse(out3)))
        out1 = self.g_relu3(self.g_bn3(self.conv2.inverse(out2)))

        x_hat = self.g_tanh(self.conv1.inverse(out1))

        return x_hat

    def forward(self, x, norm=True):
        z = self._forward(x, norm)
        x_hat = self._inverse(z.reshape(len(z),-1,1,1))
        return z, x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

class InverseNet_LRELU(nn.Module):
    # leaky relu in g
    def __init__(self, nz, ndf, nc):
        super(InverseNet_LRELU, self).__init__()

        # f layers
        self.conv1 = DictConv2d(nc, ndf, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.conv3 = DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)
        self.conv4 = DictConv2d(ndf * 4, nz, 4, 1, 0, bias=False)
        self.flatten = nn.Flatten()

        # g layers
        self.g_bn1 = nn.BatchNorm2d(ndf * 4)
        self.g_relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn2 = nn.BatchNorm2d(ndf * 2)
        self.g_relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn3 = nn.BatchNorm2d(ndf * 1)
        self.g_relu3 = nn.LeakyReLU(0.2, inplace=False)
        self.g_tanh = nn.Tanh()

    def _forward(self, x, norm=True):

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.flatten(self.conv4(out3))
        if norm:
            out4 = F.normalize(out4)
        return out4

    def _inverse(self, z):
        out4 = z #z.view(len(z),-1,1,1)
        out3 = self.g_relu1(self.g_bn1(self.conv4.inverse(out4)))
        out2 = self.g_relu2(self.g_bn2(self.conv3.inverse(out3)))
        out1 = self.g_relu3(self.g_bn3(self.conv2.inverse(out2)))
        x_hat = self.g_tanh(self.conv1.inverse(out1))

        return x_hat

    def forward(self, x, norm=True):
        z = self._forward(x, norm)
        x_hat = self._inverse(z.reshape(len(z),-1,1,1))
        return z, x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

# class InverseNet2_realinv(nn.Module):
#     # leaky relu in g
#     def __init__(self, nz, ndf, nc):
#         super(InverseNet2_realinv, self).__init__()

#         # f layers
#         self.conv1 = DictConv2d(nc, ndf, 4, 2, 1, bias=False)
#         self.relu1 = LeakyReLU(0.2, inplace=False)
#         self.conv2 = DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
#         self.bn2 = BatchNorm(ndf * 2)
#         self.relu2 = LeakyReLU(0.2, inplace=False)
#         self.conv3 = DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
#         self.bn3 = BatchNorm(ndf * 4)
#         self.relu3 = LeakyReLU(0.2, inplace=False)
#         self.conv4 = DictConv2d(ndf * 4, nz, 4, 1, 0, bias=False)
#         self.flatten = nn.Flatten()

#         # g layers
#         # self.g_bn1 = nn.BatchNorm2d(ndf * 4)
#         # self.g_relu1 = nn.LeakyReLU(0.2, inplace=False)
#         # self.g_bn2 = nn.BatchNorm2d(ndf * 2)
#         # self.g_relu2 = nn.LeakyReLU(0.2, inplace=False)
#         # self.g_bn3 = nn.BatchNorm2d(ndf * 1)
#         # self.g_relu3 = nn.LeakyReLU(0.2, inplace=False)
#         # self.g_tanh = nn.Tanh()

#     def f_forward(self, x):

#         out1 = self.relu1(self.conv1(x))
#         out2 = self.relu2(self.bn2(self.conv2(out1)))
#         out3 = self.relu3(self.bn3(self.conv3(out2)))
#         out4 = self.flatten(self.conv4(out3))
#         # print(out3[0][0])
#         return out4

#     def g_forward(self, z):
#         out3 = self.conv4.inverse(z.view(len(z),-1,1,1))
#         out2 = self.conv3.inverse(self.bn3.inverse(self.relu3.inverse(out3)))
#         out1 = self.conv2.inverse(self.bn2.inverse(self.relu2.inverse(out2)))
#         x_hat = self.conv1.inverse((self.relu1.inverse(out1)))
#         # print(out3[0][0])
#         return x_hat

#     def update_stepsize(self):
#         for m in self.modules():
#             if isinstance(m, DictBlock):
#                 m.update_stepsize()

# class InverseNet3(nn.Module):
#     # no relu in g
#     def __init__(self, nz, ndf, nc):
#         super(InverseNet3, self).__init__()

#         # f layers
#         self.conv1 = DictConv2d(nc, ndf, 4, 2, 1, bias=False)
#         self.relu1 = nn.LeakyReLU(0.2, inplace=False)
#         self.conv2 = DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(ndf * 2)
#         self.relu2 = nn.LeakyReLU(0.2, inplace=False)
#         self.conv3 = DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(ndf * 4)
#         self.relu3 = nn.LeakyReLU(0.2, inplace=False)
#         self.conv4 = DictConv2d(ndf * 4, nz, 4, 1, 0, bias=False)
#         self.flatten = nn.Flatten()

#         # g layers
#         self.g_bn1 = nn.BatchNorm2d(ndf * 4)
#         # self.g_relu1 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_bn2 = nn.BatchNorm2d(ndf * 2)
#         # self.g_relu2 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_bn3 = nn.BatchNorm2d(ndf * 1)
#         # self.g_relu3 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_tanh = nn.Tanh()

#     def f_forward(self, x):

#         out1 = self.relu1(self.conv1(x))
#         out2 = self.relu2(self.bn2(self.conv2(out1)))
#         out3 = self.relu3(self.bn3(self.conv3(out2)))
#         out4 = self.flatten(self.conv4(out3))

#         return F.normalize(out4)

#     def g_forward(self, z):
#         out4 = z.reshape(len(z),-1,1,1)
#         out3 = self.g_bn1(self.conv4.inverse(out4))
#         out2 = self.g_bn2(self.conv3.inverse(out3))
#         out1 = self.g_bn3(self.conv2.inverse(out2))

#         x_hat = self.g_tanh(self.conv1.inverse(out1))

#         return x_hat

#     def update_stepsize(self):
#         for m in self.modules():
#             if isinstance(m, DictBlock):
#                 m.update_stepsize()

class InverseNet_8_layers(nn.Module):
    # more layers
    def __init__(self, nz, ndf, nc):
        super(InverseNet_8_layers, self).__init__()

        # f layers
        ### (3, 32, 32)
        self.conv1_1 = DictConv2d(nc, ndf, 4, 2, 1, bias=False)
        self.relu1_1 = nn.LeakyReLU(0.2, inplace=False)
        
        ### (64, 16, 16)
        self.conv1_2 = DictConv2d(ndf, ndf, 3, 1, 1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(ndf)
        self.relu1_2 = nn.LeakyReLU(0.2, inplace=False)

        ### (64, 16, 16)

        self.conv2_1 = DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(ndf * 2)
        self.relu2_1 = nn.LeakyReLU(0.2, inplace=False)

        self.conv2_2 = DictConv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(ndf * 2)
        self.relu2_2 = nn.LeakyReLU(0.2, inplace=False)

        self.conv3_1 = DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(ndf * 4)
        self.relu3_1 = nn.LeakyReLU(0.2, inplace=False)

        self.conv3_2 = DictConv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(ndf * 4)
        self.relu3_2 = nn.LeakyReLU(0.2, inplace=False)

        self.conv4_1 = DictConv2d(ndf * 4, ndf * 4, 4, 1, 0, bias=False)
        self.bn4_1 = nn.BatchNorm2d(ndf * 4)
        self.relu4_1 = nn.LeakyReLU(0.2, inplace=False)

        self.conv4_2 = DictConv2d(ndf * 4, nz, 1, 1, 0, bias=False)
        self.flatten = nn.Flatten()

        # g layers
        self.g_bn1_2 = nn.BatchNorm2d(ndf * 1)
        self.g_relu1_2 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn2_1 = nn.BatchNorm2d(ndf * 1)
        self.g_relu2_1 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn2_2 = nn.BatchNorm2d(ndf * 2)
        self.g_relu2_2 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn3_1 = nn.BatchNorm2d(ndf * 2)
        self.g_relu3_1 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn3_2 = nn.BatchNorm2d(ndf * 4)
        self.g_relu3_2 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn4_1 = nn.BatchNorm2d(ndf * 4)
        self.g_relu4_1 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn4_2 = nn.BatchNorm2d(ndf * 4)
        self.g_relu4_2 = nn.LeakyReLU(0.2, inplace=False)
        self.g_tanh = nn.Tanh()

    def _forward(self, x):
        # print("\tIn Model: input size", x.size())

        out1_1 = self.relu1_1(self.conv1_1(x))
        out1_2 = self.relu1_2(self.bn1_2(self.conv1_2(out1_1)))

        out2_1 = self.relu2_1(self.bn2_1(self.conv2_1(out1_2)))
        out2_2 = self.relu2_2(self.bn2_2(self.conv2_2(out2_1)))

        out3_1 = self.relu3_1(self.bn3_1(self.conv3_1(out2_2)))
        out3_2 = self.relu3_2(self.bn3_2(self.conv3_2(out3_1)))

        out4_1 = self.relu4_1(self.bn4_1(self.conv4_1(out3_2)))
        out4_2 = self.flatten(self.conv4_2(out4_1))

        return F.normalize(out4_2)

    def _inverse(self, z):
        out4_2 = z #z.view(len(z),-1,1,1)
        out4_1 = self.g_relu4_2(self.g_bn4_2(self.conv4_2.inverse(out4_2)))
        out3_2 = self.g_relu4_1(self.g_bn4_1(self.conv4_1.inverse(out4_1)))

        out3_1 = self.g_relu3_2(self.g_bn3_2(self.conv3_2.inverse(out3_2)))
        out2_2 = self.g_relu3_1(self.g_bn3_1(self.conv3_1.inverse(out3_1)))

        out2_1 = self.g_relu2_2(self.g_bn2_2(self.conv2_2.inverse(out2_2)))
        out1_2 = self.g_relu2_1(self.g_bn2_1(self.conv2_1.inverse(out2_1)))

        out1_1 = self.g_relu1_2(self.g_bn1_2(self.conv1_2.inverse(out1_2)))
        x_hat = self.g_tanh(self.conv1_1.inverse(out1_1))

        return x_hat

    def forward(self, x):
        z = self._forward(x)
        x_hat = self._inverse(z.reshape(len(z),-1,1,1))
        return z, x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

# class BasicBlock_2layers(nn.Module):
#     def __init__(self, in_channels, out_channels, n_layers=2, stride=2):
#         super(BasicBlock_2layers, self).__init__()

#         self.conv1 = DictConv2d(in_channels, out_channels, 
#                                 kernel_size=4, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu1 = nn.LeakyReLU(0.2, inplace=False)
#         self.conv2 = DictConv2d(out_channels, out_channels, 
#                                 kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu2 = nn.LeakyReLU(0.2, inplace=False)

#         # g layers
#         self.g_bn1 = nn.BatchNorm2d(out_channels)
#         self.g_relu1 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_bn2 = nn.BatchNorm2d(in_channels)
#         self.g_relu2 = nn.LeakyReLU(0.2, inplace=False)

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.relu2(self.bn2(self.conv2(out)))
#         return out

#     def inverse(self, z):
#         out = self.g_relu2(self.g_bn2(self.conv2.inverse(z)))
#         out = self.g_relu1(self.g_bn1(self.conv1.inverse(out)))
#         return out

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_layers=2, stride=2, padding=1, with_ACT=False, with_fBN=True, with_gBN=True, fACT_last=nn.LeakyReLU(negative_slope=0.2, inplace=False), gACT_last=nn.LeakyReLU(negative_slope=0.2, inplace=False)):
        super(BasicBlock, self).__init__()
        self.n_layers = n_layers

        strides = [stride] + [1]*(self.n_layers-1)
        in_channels = [in_channel] + [out_channel]*(self.n_layers-1)
        out_channels = [out_channel]*(self.n_layers)
        kernel_sizes = [4] + [3]*(self.n_layers-1)

        DC_layers = []
        fBN_layers = []
        gBN_layers = []
        fACT_funcs = []
        gACT_funcs = []
        for i in range(self.n_layers):
            DC_layers.append(DictConv2d(in_channels[i], out_channels[i], 
                                kernel_size=kernel_sizes[i], stride=strides[i], padding=padding, bias=False))
            ## add BN on f
            if with_fBN:
                fBN_layers.append(nn.BatchNorm2d(out_channels[i]))
            else:
                fBN_layers.append(nn.Identity())
            ## add BN on g
            if with_gBN:
                gBN_layers.append(nn.BatchNorm2d(in_channels[i]))
            else:
                gBN_layers.append(nn.Identity())
            ## add ACT function
            if i < self.n_layers-1:
                if with_ACT:
                    fACT_funcs.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
                    gACT_funcs.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
                else: 
                    fACT_funcs.append(nn.Identity())
                    gACT_funcs.append(nn.Identity())  
            elif i == self.n_layers-1:
                fACT_funcs.append(fACT_last)
                gACT_funcs.append(gACT_last)

        self.DCs = nn.ModuleList(DC_layers)
        self.fBNs = nn.ModuleList(fBN_layers)
        self.gBNs = nn.ModuleList(gBN_layers)
        self.fACT_funcs = nn.ModuleList(fACT_funcs)
        self.gACT_funcs = nn.ModuleList(gACT_funcs)

    def _forward(self, x):
        out = x
        for i in range(self.n_layers):
            # print(out.shape, self.DCs[i], self.fBNs[i], self.fACT_funcs[i])
            out = self.fACT_funcs[i](self.fBNs[i](self.DCs[i](out)))
        return out

    def _inverse(self, z):
        out = z
        for i in range(self.n_layers):
            # print(out.shape, self.DCs[self.n_layers-1-i], self.gBNs[self.n_layers-1-i], self.gACT_funcs[self.n_layers-1-i])
            out = self.gACT_funcs[self.n_layers-1-i](self.gBNs[self.n_layers-1-i](self.DCs[self.n_layers-1-i].inverse(out)))
        return out

    def forward(self, x):
        return self._forward(x)

# def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)


# class Block(nn.Sequential):
#     def forward(self, input):
#         for module in self:
#             input = module(input)
#         return input

#     def inverse(self, input):
#         for module in self:
#             try:
#                 input = module.inverse(input)
#                 print(module, 'inverse')
#             except:
#                 input = module(input)
#                 print(module, 'forward')
#             return input

class InverseNet_block(nn.Module):
    # more layers
    def __init__(self, nz=512, ndf=64, nc=3, n_layers=[1,1,1,1]):
        super(InverseNet_block, self).__init__()
        # f layers
        self.block1 = BasicBlock(nc, ndf, n_layers=n_layers[0], stride=2, with_fBN=False, with_gBN=False, gACT_last=nn.Tanh())
        self.block2 = BasicBlock(ndf, ndf*2, n_layers=n_layers[1], stride=2)
        self.block3 = BasicBlock(ndf*2, ndf*4, n_layers=n_layers[2], stride=2)
        self.block4 = BasicBlock(ndf*4, nz, n_layers=n_layers[3], stride=1, padding=0, with_fBN=False, fACT_last=nn.Flatten())

    def forward(self, x, norm=True):
        # print(x.shape)
        z = self._forward(x, norm=norm)
        x_hat = self._inverse(z)
        return z, x_hat

    def _forward(self, x, norm=True):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        z = self.block4(out3)
        if norm:
            z = F.normalize(z)
        return z

    def _inverse(self, z):
        z = z.view(len(z),-1,1,1)
        out4_inv = self.block4._inverse(z)
        out3_inv = self.block3._inverse(out4_inv)
        out2_inv = self.block2._inverse(out3_inv)
        out1_inv = self.block1._inverse(out2_inv)
        return out1_inv

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()