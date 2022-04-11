import torch
import numpy as np
import torch.nn as nn
from utils import label_to_membership

class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=10):
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        Pi = label_to_membership(Y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        discrimn_loss_theo = discrimn_loss_empi
        compress_loss_theo = compress_loss_empi
 
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()])

class MCRGANloss(nn.Module):

    def __init__(self, gam1=1., gam2=1., eps=0.5, num_class=10, mode='multi'):
        super(MCRGANloss, self).__init__()

        self.criterion = MaximalCodingRateReduction(gam1=gam1, gam2=gam2, eps=eps)
        self.num_class = num_class
        self.train_mode = mode

    def forward(self, Z, Z_bar, real_label, weights=[1,1,1]):

        if self.train_mode == 'multi':
            loss_z, _, _ = self.criterion(Z, real_label)
            loss_h, _, _ = self.criterion(Z_bar, real_label)
            loss_class = 0 
            for i in np.arange(self.num_class):
                new_Z = torch.cat((Z[real_label == i], Z_bar[real_label == i]), dim=0)
                new_label = torch.cat((torch.zeros_like(real_label[real_label == i]), torch.ones_like(real_label[real_label == i])), dim=0)
                loss, _, _ = self.criterion(new_Z, new_label)
                loss_class += loss    
            errD = weights[0] * loss_z + weights[1] * loss_h + weights[2] * loss_class
            return errD, [loss_z, loss_h, loss_class]

        elif self.train_mode == 'max_multi':
            loss_z, _, _ = self.criterion(Z, real_label)
            loss_h, _, _ = self.criterion(Z_bar, real_label)
            loss_class = 0 
            for i in np.arange(self.num_class):
                new_Z = torch.cat((Z[real_label == i], Z_bar[real_label == i]), dim=0)
                new_label = torch.cat((torch.zeros_like(real_label[real_label == i]), torch.ones_like(real_label[real_label == i])), dim=0)
                loss, _, _ = self.criterion(new_Z, new_label)
                loss_class += loss    
            errD = loss_z + loss_h - loss_class
            return errD, [loss_z, loss_h, loss_class]

        elif self.train_mode == 'binary':
            new_Z = torch.cat((Z, Z_bar), 0)
            new_label = torch.cat((torch.zeros_like(real_label), torch.ones_like(real_label)))
            errD, _, _ = self.criterion(new_Z, new_label)
            return errD, [None, None, None]
        else:
            raise ValueError()
