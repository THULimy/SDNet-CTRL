import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as FF

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.
    Parameters:
        targets (np.ndarray): matrix with one hot labels
    Return:
        Pi: membership matrix, shape (num_classes, num_samples, num_samples)
    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi

### Build Model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def normalize_back(img, mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5)):
    mean = torch.tensor(mean).view(1,-1,1,1)
    std = torch.tensor(std).view(1,-1,1,1)
    return (img*std)+mean

def save_fig(imgs, epoch, iters, model_dir='./'):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(os.path.join(model_dir,'images',str(epoch)+'_'+str(iters)+".png"))

class MyAffineTransform:
    """Transform by one of the ways."""
    '''Choice:[angle, scale]'''

    def __init__(self, choices):
        self.choices = choices

    def __call__(self, x):    
        choice = random.choice(self.choices)
        x = FF.affine(x, angle=choice[0], scale=choice[1], translate=[0,0], shear=0)
        return x
