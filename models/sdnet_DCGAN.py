import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .dictnet import DictBlock
from config import config as cfg

class DictConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DictConv2d, self).__init__()
        assert stride <= 2
        self.in_stride = stride
        self.stride = 1 if cfg['MODEL']['POOLING'] else stride
        # print(cfg)
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

    def forward(self, input):
        return F.normalize(self.main(input))

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

    def forward(self, input):
        return F.normalize(self.main(input))

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

    def f_forward(self, x):

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.flatten(self.conv4(out3))

        return F.normalize(out4)

    def g_forward(self, z):
        out4 = z.reshape(len(z),-1,1,1)
        out3 = self.g_relu1(self.g_bn1(self.conv4.inverse(out4)))
        out2 = self.g_relu2(self.g_bn2(self.conv3.inverse(out3)))
        out1 = self.g_relu3(self.g_bn3(self.conv2.inverse(out2)))

        x_hat = self.g_tanh(self.conv1.inverse(out1))

        return x_hat

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

    def f_forward(self, x):

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.flatten(self.conv4(out3))

        return F.normalize(out4)

    def g_forward(self, z):
        out4 = z.reshape(len(z),-1,1,1)
        out3 = self.g_relu1(self.g_bn1(self.conv4.inverse(out4)))
        out2 = self.g_relu2(self.g_bn2(self.conv3.inverse(out3)))
        out1 = self.g_relu3(self.g_bn3(self.conv2.inverse(out2)))

        x_hat = self.g_tanh(self.conv1.inverse(out1))

        return x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

class InverseNet2(nn.Module):
    # leaky relu in g
    def __init__(self, nz, ndf, nc):
        super(InverseNet2, self).__init__()

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

    def f_forward(self, x):

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.flatten(self.conv4(out3))

        return F.normalize(out4)

    def g_forward(self, z):
        out4 = z #z.view(len(z),-1,1,1)
        out3 = self.g_relu1(self.g_bn1(self.conv4.inverse(out4)))
        out2 = self.g_relu2(self.g_bn2(self.conv3.inverse(out3)))
        out1 = self.g_relu3(self.g_bn3(self.conv2.inverse(out2)))
        x_hat = self.g_tanh(self.conv1.inverse(out1))

        return x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

class InverseNet2_realinv(nn.Module):
    # leaky relu in g
    def __init__(self, nz, ndf, nc):
        super(InverseNet2_realinv, self).__init__()

        # f layers
        self.conv1 = DictConv2d(nc, ndf, 4, 2, 1, bias=False)
        self.relu1 = LeakyReLU(0.2, inplace=False)
        self.conv2 = DictConv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = BatchNorm(ndf * 2)
        self.relu2 = LeakyReLU(0.2, inplace=False)
        self.conv3 = DictConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = BatchNorm(ndf * 4)
        self.relu3 = LeakyReLU(0.2, inplace=False)
        self.conv4 = DictConv2d(ndf * 4, nz, 4, 1, 0, bias=False)
        self.flatten = nn.Flatten()

        # g layers
        # self.g_bn1 = nn.BatchNorm2d(ndf * 4)
        # self.g_relu1 = nn.LeakyReLU(0.2, inplace=False)
        # self.g_bn2 = nn.BatchNorm2d(ndf * 2)
        # self.g_relu2 = nn.LeakyReLU(0.2, inplace=False)
        # self.g_bn3 = nn.BatchNorm2d(ndf * 1)
        # self.g_relu3 = nn.LeakyReLU(0.2, inplace=False)
        # self.g_tanh = nn.Tanh()

    def f_forward(self, x):

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.flatten(self.conv4(out3))
        # print(out3[0][0])
        return out4

    def g_forward(self, z):
        out3 = self.conv4.inverse(z.view(len(z),-1,1,1))
        out2 = self.conv3.inverse(self.bn3.inverse(self.relu3.inverse(out3)))
        out1 = self.conv2.inverse(self.bn2.inverse(self.relu2.inverse(out2)))
        x_hat = self.conv1.inverse((self.relu1.inverse(out1)))
        # print(out3[0][0])
        return x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

class InverseNet3(nn.Module):
    # no relu in g
    def __init__(self, nz, ndf, nc):
        super(InverseNet3, self).__init__()

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
        # self.g_relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn2 = nn.BatchNorm2d(ndf * 2)
        # self.g_relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.g_bn3 = nn.BatchNorm2d(ndf * 1)
        # self.g_relu3 = nn.LeakyReLU(0.2, inplace=False)
        self.g_tanh = nn.Tanh()

    def f_forward(self, x):

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.flatten(self.conv4(out3))

        return F.normalize(out4)

    def g_forward(self, z):
        out4 = z.reshape(len(z),-1,1,1)
        out3 = self.g_bn1(self.conv4.inverse(out4))
        out2 = self.g_bn2(self.conv3.inverse(out3))
        out1 = self.g_bn3(self.conv2.inverse(out2))

        x_hat = self.g_tanh(self.conv1.inverse(out1))

        return x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

class InverseNet4(nn.Module):
    # more layers
    def __init__(self, nz, ndf, nc):
        super(InverseNet4, self).__init__()

        # f layers
        self.conv1_1 = DictConv2d(nc, ndf, 4, 2, 1, bias=False)
        self.relu1_1 = nn.LeakyReLU(0.2, inplace=False)

        self.conv1_2 = DictConv2d(ndf, ndf, 3, 1, 1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(ndf)
        self.relu1_2 = nn.LeakyReLU(0.2, inplace=False)

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

    def f_forward(self, x):
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

    def g_forward(self, z):
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
        z = self.f_forward(x)
        x_hat = self.g_forward(z.reshape(len(z),-1,1,1))
        return z, x_hat

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

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

# class InverseNet2_block(nn.Module):
#     # more layers
#     def __init__(self, nz, ndf, nc):
#         super(InverseNet2_block, self).__init__()

#         # f layers
#         self.block0 = nn.Sequential(
#                 DictConv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
#                 #nn.BatchNorm2d(ndf),
#                 nn.LeakyReLU(0.2, inplace=False),
#             )
#         self.block1 = nn.Sequential(
#                 DictConv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(ndf*2),
#                 nn.LeakyReLU(0.2, inplace=False),
#             )
#         self.block2 = nn.Sequential(
#                 DictConv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(ndf*4),
#                 nn.LeakyReLU(0.2, inplace=False),
#             )
#         self.block3 = nn.Sequential(
#                 DictConv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
#                 nn.Flatten(),
#             )

#         # g layers
#         self.g_bn1_2 = nn.BatchNorm2d(ndf * 1)
#         self.g_relu1_2 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_bn2_1 = nn.BatchNorm2d(ndf * 1)
#         self.g_relu2_1 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_bn2_2 = nn.BatchNorm2d(ndf * 2)
#         self.g_relu2_2 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_bn3_1 = nn.BatchNorm2d(ndf * 2)
#         self.g_relu3_1 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_bn3_2 = nn.BatchNorm2d(ndf * 4)
#         self.g_relu3_2 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_bn4_1 = nn.BatchNorm2d(ndf * 4)
#         self.g_relu4_1 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_bn4_2 = nn.BatchNorm2d(ndf * 4)
#         self.g_relu4_2 = nn.LeakyReLU(0.2, inplace=False)
#         self.g_tanh = nn.Tanh()

#     def f_forward(self, x):
#         out0 = self.block0(x)
#         out1 = self.block1(out0)
#         out2 = self.block2(out1)
#         out3 = self.block3(out2)
#         return F.normalize(out3)

#     def g_forward(self, z):
#         out4_2 = z #z.view(len(z),-1,1,1)
#         out4_1 = self.g_relu4_2(self.g_bn4_2(self.conv4_2.inverse(out4_2)))
#         out3_2 = self.g_relu4_1(self.g_bn4_1(self.conv4_1.inverse(out4_1)))

#         out3_1 = self.g_relu3_2(self.g_bn3_2(self.conv3_2.inverse(out3_2)))
#         out2_2 = self.g_relu3_1(self.g_bn3_1(self.conv3_1.inverse(out3_1)))

#         out2_1 = self.g_relu2_2(self.g_bn2_2(self.conv2_2.inverse(out2_2)))
#         out1_2 = self.g_relu2_1(self.g_bn2_1(self.conv2_1.inverse(out2_1)))

#         out1_1 = self.g_relu1_2(self.g_bn1_2(self.conv1_2.inverse(out1_2)))
#         x_hat = self.g_tanh(self.conv1_1.inverse(out1_1))

#         return x_hat

#     def update_stepsize(self):
#         for m in self.modules():
#             if isinstance(m, DictBlock):
#                 m.update_stepsize()