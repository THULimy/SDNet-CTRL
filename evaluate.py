import argparse
# from msilib.schema import Error
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from sklearn.manifold import TSNE
import random
from typing import List, Union, Tuple

# import cluster
# import train_func as tf
import utils
from scipy import linalg
from models.inceptionII import InceptionV3 as IC

# Set random seeds and deterministic pytorch for reproducibility
manualSeed = 100
random.seed(manualSeed)  # python random seed
torch.manual_seed(manualSeed)  # pytorch random seed
np.random.seed(manualSeed)  # numpy random seed
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_pca(args, features, labels, epoch, select_label=None):
    """Plot PCA of learned features. """
    ## create save folder
    pca_dir = os.path.join(args.model_dir, 'figures', 'pca')
    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)

    ## perform PCA on features
    n_comp = np.min([args.n_comp, features.shape[1]])
    num_classes = labels.numpy().max() + 1
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    pca = PCA(n_components=n_comp).fit(features.numpy())
    sig_vals = [pca.singular_values_]
    sig_vals_each_class = []
    components_each_class = []
    means_each_class = []
    for c in range(num_classes): 
        pca = PCA(n_components=n_comp).fit(features_sort[c])
        sig_vals.append((pca.singular_values_))
        sig_vals_each_class.append((pca.singular_values_))
        components_each_class.append((pca.components_))
        means_each_class.append((pca.mean_))
        #print(sig_vals_each_class, components_each_class, means_each_class)
    ## plot features
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=500)
    x_min = np.min([len(sig_val) for sig_val in sig_vals])
    ax.plot(np.arange(x_min), sig_vals[0][:x_min], '-p', markersize=3, markeredgecolor='black',
        linewidth=1.5, color='tomato')
    map_vir = plt.cm.get_cmap('Blues', 6)
    norm = plt.Normalize(-10, 10)
    class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    norm_class = norm(class_list)
    color = map_vir(norm_class)
    for c, sig_val in enumerate(sig_vals[1:]):
        if select_label is not None:
            color_c = 'green' if c==select_label else color[c]
        else:
            color_c = color[c]
        # color_c = 'green' if c<5 else color[c]
        ax.plot(np.arange(x_min), sig_val[:x_min], '-o', markersize=3, markeredgecolor='black',
                alpha=0.6, linewidth=1.0, color=color_c)
    ax.set_xticks(np.arange(0, x_min, 5))
    ax.set_yticks(np.arange(0, 81, 5))
    ax.set_xlabel("components", fontsize=14)
    ax.set_ylabel("singular values", fontsize=14)
    [tick.label.set_fontsize(12) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(12) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    
    # save statistics
    np.save(os.path.join(pca_dir, f"sig_vals_epo{epoch}.npy"), sig_vals)
    np.save(os.path.join(pca_dir, f"sig_vals_each_class_epo{epoch}.npy"), sig_vals_each_class)
    np.save(os.path.join(pca_dir, f"components_each_class_epo{epoch}.npy"), components_each_class)
    np.save(os.path.join(pca_dir, f"means_each_class_epo{epoch}.npy"), means_each_class)

    file_name = os.path.join(pca_dir, f"pca_classVclass_epoch{epoch}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(pca_dir, f"pca_classVclass_epoch{epoch}.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_hist(args, features, labels, epoch):
    """Plot histogram of class vs. class. """
    ## create save folder
    hist_folder = os.path.join(args.model_dir, 'figures', 'hist')
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)

    num_classes = labels.numpy().max() + 1
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(num_classes):
        for j in range(i, num_classes):
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=250)
            if i == j:
                sim_mat = features_sort[i] @ features_sort[j].T
                sim_mat = sim_mat[np.triu_indices(sim_mat.shape[0], k = 1)]
            else:
                sim_mat = (features_sort[i] @ features_sort[j].T).reshape(-1)
            ax.hist(sim_mat, bins=40, color='red', alpha=0.5)
            ax.set_xlabel("cosine similarity")
            ax.set_ylabel("count")
            ax.set_title(f"Class {i} vs. Class {j}")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            fig.tight_layout()

            file_name = os.path.join(hist_folder, f"hist_{i}v{j}")
            fig.savefig(file_name)
            plt.close()
            print("Plot saved to: {}".format(file_name))


def plot_nearest_component_supervised(args, features, labels, epoch, trainset):
    """Find corresponding images to the nearests component. """
    ## perform PCA on features
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=10, stack=False)
    data_sort, _ = utils.sort_dataset(trainset.data, labels.numpy(), 
                            num_classes=10, stack=False)
    nearest_data = []
    for c in range(10):
        pca = TruncatedSVD(n_components=10, random_state=10).fit(features_sort[c])
        proj = features_sort[c] @ pca.components_.T
        img_idx = np.argmax(np.abs(proj), axis=0)
        nearest_data.append(np.array(data_sort[c])[img_idx])
    
    fig, ax = plt.subplots(ncols=10, nrows=10, figsize=(10, 10))
    for r in range(10):
        for c in range(10):
            ax[r, c].imshow(nearest_data[r][c])
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            ax[r, c].spines['top'].set_visible(False)
            ax[r, c].spines['right'].set_visible(False)
            ax[r, c].spines['bottom'].set_linewidth(False)
            ax[r, c].spines['left'].set_linewidth(False)
            if c == 0:
                ax[r, c].set_ylabel(f"comp {r}")
    ## save
    save_dir = os.path.join(args.model_dir, 'figures', 'nearcomp_sup')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"nearest_data.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"nearest_data.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_nearest_component_unsupervised(args, features, labels, epoch, trainset):
    """Find corresponding images to the nearests component. """
    save_dir = os.path.join(args.model_dir, 'figures', 'nearcomp_unsup')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    feature_dim = features.shape[1]
    pca = TruncatedSVD(n_components=feature_dim-1, random_state=10).fit(features)
    for j, comp in enumerate(pca.components_):
        proj = (features @ comp.T).numpy()
        img_idx = np.argsort(np.abs(proj), axis=0)[::-1][:10]
        nearest_vals = proj[img_idx]
        print(img_idx, trainset.data.shape)
        nearest_data = trainset.data[img_idx.copy()]
        fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(5, 2))
        i = 0
        for r in range(2):
            for c in range(5):
                ax[r, c].imshow(nearest_data[i])
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].spines['top'].set_visible(False)
                ax[r, c].spines['right'].set_visible(False)
                ax[r, c].spines['bottom'].set_linewidth(False)
                ax[r, c].spines['left'].set_linewidth(False)
                i+= 1
        file_name = os.path.join(save_dir, f"nearest_comp{j}.png")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        plt.close()


def plot_nearest_component_class(args, features, labels, epoch, trainset):
    """Find corresponding images to the nearests component per class. """
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=10, stack=False)
    data_sort, _ = utils.sort_dataset(trainset.data, labels.numpy(), 
                            num_classes=10, stack=False)

    for class_ in range(10):
        nearest_data = []
        nearest_val = []
        pca = TruncatedSVD(n_components=10, random_state=10).fit(features_sort[class_])
        for j in range(8):
            proj = features_sort[class_] @ pca.components_.T[:, j]
            img_idx = np.argsort(np.abs(proj), axis=0)[::-1][:10]
            nearest_val.append(proj[img_idx])
            nearest_data.append(np.array(data_sort[class_])[img_idx])
        
        fig, ax = plt.subplots(ncols=10, nrows=8, figsize=(10, 10))
        for r in range(8):
            for c in range(10):
                ax[r, c].imshow(nearest_data[r][c], cmap='gray')
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].spines['top'].set_visible(False)
                ax[r, c].spines['right'].set_visible(False)
                ax[r, c].spines['bottom'].set_linewidth(False)
                ax[r, c].spines['left'].set_linewidth(False)
                # ax[r, c].set_xlabel(f"proj: {nearest_val[r][c]:.2f}")
                if c == 0:
                    ax[r, c].set_ylabel(f"comp {r}")
        fig.tight_layout()

        ## save
        save_dir = os.path.join(args.model_dir, 'figures', 'nearcomp_class')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"nearest_class{class_}.png")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        file_name = os.path.join(save_dir, f"nearest_class{class_}.pdf")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        plt.close()

def plot_heatmap(args, features, labels, epoch):
    """Plot heatmap of cosine simliarity for all features. """
    num_classes = labels.numpy().max() + 1
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(len(features_sort)):
        features_sort[i] = features_sort[i][:2000]
    features_sort_ = np.vstack(features_sort)
    print(features_sort_.shape)
    sim_mat = np.abs(features_sort_ @ features_sort_.T)

    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(sim_mat, cmap='Blues')
    # im = ax.imshow(sim_mat, cmap='bwr')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, len(sim_mat), 6))
    ax.set_yticks(np.linspace(0, len(sim_mat), 6))
    [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    
    save_dir = os.path.join(args.model_dir, 'figures', 'heatmaps')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"heatmat_epoch{epoch}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"heatmat_epoch{epoch}.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_heatmap_ZnZ_hat(args, features, features_rec, labels, epoch):
    """Plot heatmap of cosine simliarity for all features. """
    num_classes = labels.numpy().max() + 1
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    features_rec_sort, _ = utils.sort_dataset(features_rec.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(len(features_sort)):
        features_sort[i] = features_sort[i]#[:2000]
        features_rec_sort[i] = features_rec_sort[i]
    features_sort_ = np.vstack(features_sort)
    features_rec_sort_ = np.vstack(features_rec_sort)

    sim_mat = np.abs(features_sort_ @ features_rec_sort_.T)

    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(sim_mat, cmap='Blues')
    # im = ax.imshow(sim_mat, cmap='bwr')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, 50000, 6))
    ax.set_yticks(np.linspace(0, 50000, 6))
    [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    
    save_dir = os.path.join(args.model_dir, 'figures', 'heatmaps')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"ZnZ_hat_heatmat_epoch{epoch}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"ZnZ_hat_heatmat_epoch{epoch}.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_tsne(args, features, labels, epoch):
    """Plot tsne of features. """
    num_classes = labels.numpy().max() + 1
    features_sort, labels_sort = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(len(features_sort)):
        features_sort[i] = features_sort[i][:200]
        labels_sort[i] = labels_sort[i][:200]
    features_sort_ = np.vstack(features_sort)
    labels_sort_ = np.hstack(labels_sort)
    print(features_sort_.shape, labels_sort_.shape)

    print('TSNEing')
    feature_tsne=TSNE(n_components=2, init='pca').fit_transform(features_sort_)
    print('TSNE Finished')

    #plot
    color = plt.cm.rainbow(np.linspace(0, 1, 10))

    plt.figure(figsize=(9,7))
    for i in range(10):
        plt.scatter(feature_tsne[labels_sort_==i,0],feature_tsne[labels_sort_==i,1],marker='o',s=30,edgecolors='k',linewidths=1,c=color[i])

    plt.xticks([])
    plt.yticks([])

    save_dir = os.path.join(args.model_dir, 'figures', 'tsne')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"tsne_train_feature_epoch{epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_tsne_all(args, features, features_recon, labels, epoch):
    """Plot tsne of features. """
    num_classes = labels.numpy().max() + 1
    features_sort, labels_sort = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    features_recon_sort, labels_sort = utils.sort_dataset(features_recon.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(len(features_sort)):
        features_sort[i] = features_sort[i][:200]
        features_recon_sort[i] = features_recon_sort[i][:200]
        labels_sort[i] = labels_sort[i][:200]
    features_sort_ = np.vstack(features_sort)
    features_recon_sort_ = np.vstack(features_recon_sort)
    features_all_ = np.concatenate((features_sort_, features_recon_sort_),axis=0)
    labels_sort_ = np.hstack(labels_sort)
    print(features_sort_.shape, labels_sort_.shape, features_all_.shape)

    print('TSNEing')
    #feature_tsne=TSNE(n_components=2, init='pca', early_exaggeration=100, perplexity=100).fit_transform(features_sort_)
    feature_tsne_all=TSNE(n_components=2, init='pca').fit_transform(features_all_)
    print('TSNE Finished')

    feature_tsne = feature_tsne_all[:feature_tsne_all.shape[0]//2]
    feature_tsne_recon = feature_tsne_all[feature_tsne_all.shape[0]//2:]

    #plot
    color = plt.cm.rainbow(np.linspace(0, 1, 10))

    plt.figure(figsize=(9,7))
    for i in range(10):
        plt.scatter(feature_tsne[labels_sort_==i,0],feature_tsne[labels_sort_==i,1],marker='o',s=30,edgecolors='k',linewidths=1,c=color[i])
        plt.scatter(feature_tsne_recon[labels_sort_==i,0],feature_tsne_recon[labels_sort_==i,1],marker='s',s=30,edgecolors='k',linewidths=1,c=color[i])
    plt.xticks([])
    plt.yticks([])

    # fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    # im = ax.imshow(sim_mat, cmap='Blues')
    # fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    # ax.set_xticks(np.linspace(0, 20000, 6))
    # ax.set_yticks(np.linspace(0, 20000, 6))
    # [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    # [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    # fig.tight_layout()

    save_dir = os.path.join(args.model_dir, 'figures', 'tsne')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"tsne_train_feature_epoch{epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_random_gen(args, netG, ncomp=8, scale=0.5):
    '''
        ncomp: the number of components for random sampling
        scale: value to control sample range
    '''
    ### Generate random samples
    print('Load statistics from PCA results.')
    sig_vals_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/sig_vals_each_class_epo{args.epoch}.npy"))
    components_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/components_each_class_epo{args.epoch}.npy"))
    means_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/means_each_class_epo{args.epoch}.npy"))
    
    random_images = []
    for i in range(10):
        sig_vals = np.array(sig_vals_each_class[i][:ncomp])
        var_vals = np.sqrt(sig_vals / np.sum(sig_vals))
        
        random_samples = np.random.normal(size=(64, ncomp))
        Z_random = means_each_class[i] + np.dot((scale * var_vals * random_samples), components_each_class[i][:ncomp]) # can modify scale to lower value to get more clear results  
        Z_random = Z_random / np.linalg.norm(Z_random, axis=1).reshape(-1,1)
        print(np.linalg.norm(Z_random, axis=1))
        X_recon_random = netG(torch.tensor(Z_random, dtype=torch.float).view(64,nz,1,1).to(device)).cpu().detach()
        print(X_recon_random.shape)
        random_images.append(X_recon_random)
        
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Random Recon Images")
        plt.imshow(np.transpose(vutils.make_grid(X_recon_random[:64], padding=2, normalize=True).cpu(), (1,2,0)))

        save_dir = os.path.join(args.model_dir, 'figures', 'images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"random_recon_images_epoch{args.epoch}_scale{scale}_class{i}.png")
        plt.savefig(file_name)
        print("Plot saved to: {}".format(file_name))

    random_images = torch.cat(([random_images[i][:8] for i in range(10)]), dim=0)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Random Recon Images")
    plt.imshow(np.transpose(vutils.make_grid(random_images, padding=2, normalize=True).cpu(), (1,2,0)))

    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"random_recon_images_epoch{args.epoch}_scale{scale}_ncomp{ncomp}_all_{manualSeed}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))


def plot_recon_imgs(train_X, train_X_bar, test_X, test_X_bar):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Train Images")
    plt.imshow(np.transpose(vutils.make_grid(train_X[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    
    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"train_images_epoch{args.epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Recon Train Images")
    plt.imshow(np.transpose(vutils.make_grid(train_X_bar[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    
    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"train_recon_images_epoch{args.epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Test Images")
    plt.imshow(np.transpose(vutils.make_grid(test_X[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    
    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"test_images_epoch{args.epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Recon Test Images")
    plt.imshow(np.transpose(vutils.make_grid(test_X_bar[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    
    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"test_recon_images_epoch{args.epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

def plot_linear_gen(args, netG, range_value=1, lin_sample_num=8):
    '''
        range_value: linspace value range
        lin_sample_num: linspace sample number
    '''
    ### Generate linspace samples
    sig_vals_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/sig_vals_each_class_epo{args.epoch}.npy"))
    components_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/components_each_class_epo{args.epoch}.npy"))
    means_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/means_each_class_epo{args.epoch}.npy"))
    
    lin_gen_images = []
    for select_class in range(10):
        for j in range(4):
            sig_vals = np.array(sig_vals_each_class[select_class][j])
            var_vals = np.sqrt(sig_vals / np.sum(sig_vals))
            
            lin_samples = np.linspace(-range_value, range_value, lin_sample_num, endpoint=True)
            Z_lin = means_each_class[select_class] + np.dot(lin_samples.reshape(-1,1), components_each_class[select_class][j].reshape(1,-1)) # can modify 1 to lower value to get more clear results  
            Z_lin = Z_lin / np.linalg.norm(Z_lin, axis=1).reshape(-1,1)
            X_recon_lin = netG(torch.tensor(Z_lin, dtype=torch.float).view(lin_sample_num,nz,1,1).to(device)).cpu().detach()
            lin_gen_images.append(X_recon_lin)

            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Lin Recon Images")
            plt.imshow(np.transpose(vutils.make_grid(X_recon_lin[:lin_sample_num], padding=2, normalize=True).cpu(), (1,2,0)))

            save_dir = os.path.join(args.model_dir, 'figures', 'images')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = os.path.join(save_dir, f"lin_recon_images_epoch{args.epoch}_class{select_class}_comp{j}_range{range_value}.png")
            plt.savefig(file_name)
            print("Plot saved to: {}".format(file_name))
    
    lin_gen_images = torch.cat(lin_gen_images,dim=0)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Lin Recon Images")
    plt.imshow(np.transpose(vutils.make_grid(lin_gen_images, nrow=16, padding=2, normalize=True).cpu(), (1,2,0)))

    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"lin_recon_images_all_epoch{args.epoch}_range{range_value}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

def plot_nearest_component_class_X_hat(args, features, labels, epoch, train_X):
    """Find corresponding images to the nearests component per class. """
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=10, stack=False)
    data_sort, _ = utils.sort_dataset(train_X, labels.numpy(), 
                            num_classes=10, stack=False)

    for class_ in range(10):
        nearest_data = []
        nearest_val = []
        pca = TruncatedSVD(n_components=10, random_state=10).fit(features_sort[class_])
        for j in range(8):
            proj = features_sort[class_] @ pca.components_.T[:, j]
            img_idx = np.argsort(np.abs(proj), axis=0)[::-1][:10]
            nearest_val.append(proj[img_idx])
            nearest_data.append(np.array(data_sort[class_])[img_idx])
        
        fig, ax = plt.subplots(ncols=10, nrows=8, figsize=(10, 10))
        for r in range(8):
            for c in range(10):
                ax[r, c].imshow(np.moveaxis(nearest_data[r][c],0,-1), cmap='gray')
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].spines['top'].set_visible(False)
                ax[r, c].spines['right'].set_visible(False)
                ax[r, c].spines['bottom'].set_linewidth(False)
                ax[r, c].spines['left'].set_linewidth(False)
                # ax[r, c].set_xlabel(f"proj: {nearest_val[r][c]:.2f}")
                if c == 0:
                    ax[r, c].set_ylabel(f"comp {r}")
        fig.tight_layout()

        ## save
        save_dir = os.path.join(args.model_dir, 'figures', 'nearcomp_class')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"nearest_class{class_}_X_hat_{args.epoch}.png")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        file_name = os.path.join(save_dir, f"nearest_class{class_}_X_hat_{args.epoch}.pdf")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        plt.close()

def plot_samples_interpolation(args, features, labels, epoch, netG, netD, norm=True, lin_sample_num=10):
    """Find corresponding images to the nearests component. """
    save_dir = os.path.join(args.model_dir, 'figures', 'sample_interpolation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    manualSeed = 5
    random.seed(manualSeed)  # python random seed
    torch.manual_seed(manualSeed)  # pytorch random seed
    np.random.seed(manualSeed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # img_id_1 = 0
    img_id_1 = []
    for i in range(10):
        indices = np.where(labels==i)
        img_id_1.append(np.random.choice(indices[0],1))
    img_id_1 = np.array(img_id_1)

    img_id_2 = []
    for i in range(len(img_id_1)):
        index = np.random.randint(0, 60000, 1)
        while labels[img_id_1[i]] == labels[index]:
            index = np.random.randint(0, 60000, 1)
        img_id_2.append(index)
    img_id_2 = np.array(img_id_2)

    # feature_dim = features.shape[1]
    # sim_mat = np.abs(features[img_id_1] @ features.T).numpy()

    # img_ids_sim = np.argsort(sim_mat, axis=1)[:,::-1][:,:10]
    # img_ids_not_sim = np.argsort(sim_mat, axis=1)[:,:10]
    # # print(img_ids_sim, img_ids_not_sim)
    # #print(img_ids_sim.shape, img_ids_not_sim.shape)
    # img_id_2 = img_ids_not_sim[:,0] # 0 is most different image
    # print(img_id_1.shape, img_id_1)
    # print(img_id_2.shape, img_id_2)

    # original_images = [traindata[img_id_1], traindata[img_id_2]]

    z1_all = features[img_id_1,:]
    z2_all = features[img_id_2,:]
    print(z1_all.shape, z2_all.shape)

    lin_sample_images = []
    for i in range(len(z1_all)):
        z1 = z1_all[i].view(1,-1,1,1)
        z2 = z2_all[i].view(1,-1,1,1)
        z_lin_sample = [(z1 + (z2-z1)*i/lin_sample_num) for i in range(lin_sample_num+1)]
        z_lin_sample = torch.cat(z_lin_sample, dim=0)

        # normalization
        z_lin_sample = F.normalize(z_lin_sample,dim=1) # normalization

        x_lin_sample = netG(z_lin_sample.view(len(z_lin_sample),-1,1,1).to(device)).detach().cpu()
        lin_sample_images.append(x_lin_sample)
        
    lin_sample_images = torch.cat(lin_sample_images,dim=0)
    print(lin_sample_images.shape)
    plt.figure(figsize=(40,40))
    plt.axis("off")
    # plt.title("Lin Recon Images")
    plt.imshow(np.transpose(vutils.make_grid(lin_sample_images, nrow=lin_sample_num+1, padding=2, normalize=True).cpu(), (1,2,0)))

    file_name = os.path.join(save_dir, f"interpolation_images_each_class_epo{epoch}_norm{norm}_linsamplenum{lin_sample_num}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))


def plot_samples_interpolation_old(args, features, epoch, traindata, netG, netD, norm=False, lin_sample_num=10):
    """Find corresponding images to the nearests component. """
    save_dir = os.path.join(args.model_dir, 'figures', 'sample_interpolation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    manualSeed = 1
    random.seed(manualSeed)  # python random seed
    torch.manual_seed(manualSeed)  # pytorch random seed
    np.random.seed(manualSeed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # img_id_1 = 0
    sample_num = 20
    img_id_1 = np.random.randint(0, 60000, sample_num)

    feature_dim = features.shape[1]
    sim_mat = np.abs(features[img_id_1] @ features.T).numpy()

    img_ids_sim = np.argsort(sim_mat, axis=1)[:,::-1][:,:10]
    img_ids_not_sim = np.argsort(sim_mat, axis=1)[:,:10]
    # print(img_ids_sim, img_ids_not_sim)
    #print(img_ids_sim.shape, img_ids_not_sim.shape)
    img_id_2 = img_ids_not_sim[:,0] # 0 is most different image
    print(img_id_1.shape, img_id_1)
    print(img_id_2.shape, img_id_2)

    original_images = [traindata[img_id_1], traindata[img_id_2]]

    z1_all = netD(traindata[img_id_1].view(-1,1,32,32).to(device))
    z2_all = netD(traindata[img_id_2].view(-1,1,32,32).to(device))
    print(z1_all.shape, z2_all.shape)

    x_hat_1 = netG(z1_all.view(len(z1_all), -1, 1, 1))
    x_hat_2 = netG(z2_all.view(len(z2_all), -1, 1, 1))
    print(x_hat_1.shape, x_hat_2.shape)

    lin_sample_images = []
    for i in range(sample_num):
        z1 = z1_all[i].view(1,-1,1,1)
        z2 = z2_all[i].view(1,-1,1,1)
        z_lin_sample = [(z1 + (z2-z1)*i/lin_sample_num) for i in range(lin_sample_num+1)]
        z_lin_sample = torch.cat(z_lin_sample, dim=0)

        if norm: # normalization
            z_lin_sample = F.normalize(z_lin_sample,dim=1) # normalization

        x_lin_sample = netG(z_lin_sample.view(len(z_lin_sample),-1,1,1)).detach().cpu()
        lin_sample_images.append(x_lin_sample)
        
    lin_sample_images = torch.cat(lin_sample_images,dim=0)
    print(lin_sample_images.shape)
    plt.figure(figsize=(40,40))
    plt.axis("off")
    # plt.title("Lin Recon Images")
    plt.imshow(np.transpose(vutils.make_grid(lin_sample_images, nrow=lin_sample_num+1, padding=2, normalize=True).cpu(), (1,2,0)))

    file_name = os.path.join(save_dir, f"interpolation_images_epo{epoch}_norm{norm}_linsamplenum{lin_sample_num}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))


def svm(args, train_features, train_labels, test_features, test_labels):
    svm = LinearSVC(verbose=0, random_state=10)
    svm.fit(train_features, train_labels)
    acc_train = svm.score(train_features, train_labels)
    acc_test = svm.score(test_features, test_labels)
    print("SVM: {}".format(acc_test))
    return acc_train, acc_test


def knn(args, train_features, train_labels, test_features, test_labels):
    """Perform k-Nearest Neighbor classification using cosine similaristy as metric.

    Options:
        k (int): top k features for kNN
    
    """
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=args.k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = utils.compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("kNN: {}".format(acc))
    return acc


def nearsub(args, train_features, train_labels, test_features, test_labels):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1 # should be correct most of the time
    features_sort, _ = utils.sort_dataset(train_features.numpy(), train_labels.numpy(), 
                                          num_classes=num_classes, stack=False)
    fd = features_sort[0].shape[1]
    for j in range(num_classes):
        pca = PCA(n_components=args.n_comp).fit(features_sort[j]) 
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_components=args.n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ (test_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        
        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_pca = utils.compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = utils.compute_accuracy(test_predict_svd, test_labels.numpy())
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return acc_svd

def nearsub_save_comps(args, train_features, train_labels, test_features, test_labels):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1 # should be correct most of the time
    features_sort, _ = utils.sort_dataset(train_features.numpy(), train_labels.numpy(), 
                                          num_classes=num_classes, stack=False)
    fd = features_sort[0].shape[1]
    pca_sig_vals_each_class = []
    pca_components_each_class = []
    pca_means_each_class = []
    svd_sig_vals_each_class = []
    svd_components_each_class = []
    for j in range(num_classes):
        pca = PCA(n_components=args.n_comp).fit(features_sort[j]) 
        pca_subspace = pca.components_.T
        print('PCA', 'class', j, 'explained_variance', pca.explained_variance_.sum())
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)
        pca_sig_vals_each_class.append((pca.singular_values_))
        pca_components_each_class.append((pca.components_))
        pca_means_each_class.append((pca.mean_))


        svd = TruncatedSVD(n_components=args.n_comp).fit(features_sort[j])
        print('SVD', 'class', j, 'explained_variance', svd.explained_variance_.sum())
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ (test_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        svd_sig_vals_each_class.append((svd.singular_values_))
        svd_components_each_class.append((svd.components_))
        
        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)

    # save components
    np.save(os.path.join(args.model_dir, f"pca_sig_vals_each_class_epo{args.epoch}.npy"), pca_sig_vals_each_class)
    np.save(os.path.join(args.model_dir, f"pca_components_each_class_epo{args.epoch}.npy"), pca_components_each_class)
    np.save(os.path.join(args.model_dir, f"pca_means_each_class_epo{args.epoch}.npy"), pca_means_each_class)
    np.save(os.path.join(args.model_dir, f"svd_sig_vals_each_class_epo{args.epoch}.npy"), svd_sig_vals_each_class)
    np.save(os.path.join(args.model_dir, f"svd_components_each_class_epo{args.epoch}.npy"), svd_components_each_class)

    # print(scores_pca)
    print('Error Distance:')
    print("PCA:", scores_pca[int(test_labels[0].data)][:10], scores_pca[int(test_labels[0].data)][:10].mean(), scores_pca[9][:10], scores_pca[9][:10].mean())
    print("SVD:", scores_svd[int(test_labels[0].data)][:10], scores_svd[int(test_labels[0].data)][:10].mean(), scores_svd[9][:10], scores_svd[9][:10].mean())

    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)

    print(test_predict_pca[:10], test_predict_svd[:10])
    acc_pca = utils.compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = utils.compute_accuracy(test_predict_svd, test_labels.numpy())
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))

    return acc_svd

def nearsub_for_adv(train_features, train_labels, test_features, test_labels, n_comp=12):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1 # should be correct most of the time
    features_sort, _ = utils.sort_dataset(train_features.numpy(), train_labels.numpy(), 
                                          num_classes=num_classes, stack=False)
    fd = features_sort[0].shape[1]
    for j in range(num_classes):
        pca = PCA(n_components=n_comp).fit(features_sort[j]) 
        pca_subspace = pca.components_.T
        # print('PCA', 'class', j, 'explained_variance', pca.explained_variance_.sum())
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_components=n_comp).fit(features_sort[j])
        # print('SVD', 'class', j, 'explained_variance', svd.explained_variance_.sum())
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ (test_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        
        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    # print(scores_pca)

    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)

    print(test_predict_pca[:10], test_predict_svd[:10])
    acc_pca = utils.compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = utils.compute_accuracy(test_predict_svd, test_labels.numpy())
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return scores_pca, scores_svd

def kmeans(args, train_features, train_labels):
    """Perform KMeans clustering. 
    
    Options:
        n (int): number of clusters used in KMeans.

    """
    return cluster.kmeans(args, train_features, train_labels)

def ensc(args, train_features, train_labels):
    """Perform Elastic Net Subspace Clustering.
    
    Options:
        gam (float): gamma parameter in EnSC
        tau (float): tau parameter in EnSC

    """
    return cluster.ensc(args, train_features, train_labels)

def get_data(trainloader, verbose=True):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    labels = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    for step, (batch_imgs, batch_lbls) in enumerate(train_bar):
        features.append(batch_imgs.view(-1,len(batch_imgs)).cpu().detach())
        labels.append(batch_lbls)
    return torch.cat(features), torch.cat(labels)

def get_features(model, trainloader, verbose=True):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    recons = []
    labels = []
    imgs = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    for step, (X, Y) in enumerate(train_bar):
        model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()

        Z = model.f_forward(X.cuda())
        X_hat = model.g_forward(Z.reshape(len(Z),-1,1,1))
        
        features.append(Z.view(-1, Z.shape[1]).cpu().detach())
        recons.append(X_hat.cpu().detach())
        labels.append(Y)
        imgs.append(X.cpu().detach())
    return torch.cat(features), torch.cat(labels), torch.cat(recons), torch.cat(imgs)

def get_features_gaussian_noise(model, trainloader, noise_sd=0.1, verbose=True):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    recons = []
    labels = []
    imgs = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    for step, (X, Y) in enumerate(train_bar):
        model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()

        X_noisy = X + noise_sd * torch.randn_like(X)
        Z = model.f_forward(X_noisy.cuda())
        X_hat = model.g_forward(Z.reshape(len(Z),-1,1,1))
        
        features.append(Z.view(-1, Z.shape[1]).cpu().detach())
        recons.append(X_hat.cpu().detach())
        labels.append(Y)
        imgs.append(X_noisy.cpu().detach())
    return torch.cat(features), torch.cat(labels), torch.cat(recons), torch.cat(imgs)

def train_clf_and_adv_attack_on_subspace(model, trainloader, testloader, ckpt_dir): 
    '''
        22.3.1
        Try to attack to the target subspace.
    '''
    ## set save_dir
    save_dir = os.path.join(ckpt_dir, 'figures', 'adv_images_subspace_proj')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.set_printoptions(precision=2, suppress=True)
    torch.set_printoptions(precision=2, sci_mode=False)
    ## Train classifier
    clf = nn.Linear(nz, 10).to(device)
    optimizer = optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)
    train_clf_epoch = 10
    feat_nat_train_all = []
    label_nat_train_all = []
    clf_path = os.path.join(ckpt_dir, 'checkpoints', 'model-clf-epoch{}.pt'.format(train_clf_epoch))
    if os.path.exists(clf_path):
        print('##### LOADING CLF from {} #####'.format(clf_path))
        clf.load_state_dict(torch.load(clf_path, map_location=device))
        clf.eval()
    else:
        print('##### START TRAINING CLF #####')
        for epoch in range(train_clf_epoch):
            correct = 0
            total = 0
            model.eval()
            clf.train()
            for i, (data, labels) in enumerate(trainloader):
                model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()

                feat = model.f_forward(data.to(device))
                pred = clf(feat)

                loss = criterion(pred, labels.long().to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct += (torch.argmax(pred, dim=1) == labels.to(device)).sum()
                total += len(labels)
            print(epoch, 'train acc:', correct.float()/total)
        
        ## save clf model for later usage
        torch.save(clf.state_dict(), os.path.join(ckpt_dir, 'checkpoints', 'model-clf-epoch{}.pt'.format(train_clf_epoch)))
        print('##### FINISH TRAINING CLF #####')
    
    print('##### Extracting Training Data Features #####')        
    correct = 0
    total = 0
    model.eval()
    clf.eval()
    for i, (data, labels) in enumerate(trainloader):
        model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()

        if i >= 10:
            break
        feat = model.f_forward(data.to(device))
        pred = clf(feat)

        correct += (torch.argmax(pred, dim=1) == labels.to(device)).sum()
        total += len(labels)                

        feat_nat_train_all.append(feat)
        label_nat_train_all.append(labels)
    print('train acc:', correct.float()/total)
    
    correct = 0
    total = 0
    model.eval()
    clf.eval()
    for i, (data, labels) in enumerate(testloader):
        model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()

        feat = model.f_forward(data.to(device))
        pred = clf(feat)

        correct += (torch.argmax(pred, dim=1) == labels.to(device)).sum()
        # correct += (labels.to(DEVICE) == 9 ).sum()

        total += len(labels)
    print('test acc:', correct.float()/total)
    feat_nat_train_all = torch.cat(feat_nat_train_all, dim=0)
    label_nat_train_all = torch.cat(label_nat_train_all, dim=0)
        
    ## generate adversarial examples
    pgd_perturb_steps = 10
    pgd_epsilon = 0.6
    pgd_step_size = (pgd_epsilon + 0.2) / pgd_perturb_steps # make largest step a little bit larger than eps

    criterion_CE = nn.CrossEntropyLoss()
    print('##### GENERATE ADVERSARIAL EXAMPLES #####')

    correct=0
    total=0
    x_adv_final_all = []
    x_adv_final_recon_all = []
    feat_nat_all = []
    feat_recon_nat_all = []

    x_adv_all = [[] for _ in range(pgd_perturb_steps+1)]
    feat_adv_all = [[] for _ in range(pgd_perturb_steps+1)]
    x_recon_adv_all = [[] for _ in range(pgd_perturb_steps+1)]
    feat_recon_adv_all = [[] for _ in range(pgd_perturb_steps+1)]
    pred_adv_all = [[] for _ in range(pgd_perturb_steps+1)]

    labels_all = []

    load_epoch = 100
    # svd_sig_vals_each_class = np.load(os.path.join(ckpt_dir, f"svd_sig_vals_each_class_epo{load_epoch}.npy"))
    # svd_components_each_class = np.load(os.path.join(ckpt_dir, f"svd_components_each_class_epo{load_epoch}.npy"))
    # print('svd_comp_shape:',svd_components_each_class.shape)
    # pca_means_each_class = np.load(os.path.join(save_dir, f"pca_means_each_class_epo{load_epoch}.npy"))
    
    for i, (data, labels) in enumerate(testloader):
        if i > 10:
            break
        x_nat = data.to(device)
        x_adv = x_nat.detach().clone() + 0.001 * torch.randn(x_nat.shape).to(device).detach()
        x_adv_batch = []
        x_adv_batch.append(x_adv.clone())
        for _ in range(pgd_perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()

                feat = model.f_forward(x_adv)
                # svd_subspace = torch.from_numpy(svd_components_each_class[9].T).to(device) # select target class 9
                # error = (torch.eye(128).to(device) - svd_subspace @ svd_subspace.T) @ (feat).T
                # error_norm = torch.norm(error, p=2, dim=1)
                # loss = - error_norm.mean()

                pred = clf(feat)
                # print(pred.argmax(dim=1))
                # loss_CE = criterion_CE(pred, labels.to(device)) #untarget
                loss = criterion_CE(pred, labels.to(device)) - criterion_CE(pred, 9 * torch.ones_like(labels).to(device)) #target 9

            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + pgd_step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_nat - pgd_epsilon), x_nat + pgd_epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x_adv_batch.append(x_adv.clone()) 
            # print((x_adv-x_nat).abs().max(), loss_CE.item())

        # save nature features and labels
        model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()

        feat_nat = model.f_forward(x_nat.to(device)).detach()
        x_recon_nat = model.g_forward(feat_nat.view(-1, 128, 1, 1)).detach()
        feat_recon_nat = model.f_forward(x_recon_nat).detach()

        feat_nat_all.append(feat_nat)
        feat_recon_nat_all.append(feat_recon_nat)
        labels_all.append(labels)

        # calculate adv accuracy (of the last one)
        model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()

        feat_adv = model.f_forward(x_adv.to(device))
        pred = clf(feat_adv)
        correct += (pred.argmax(dim=1) == labels.to(device)).sum().item()
        total += len(labels)

        # get the features and recons of adversarial samples in the attacking progress
        for attack_step, x_adv in enumerate(x_adv_batch):
            model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()
        
            feat_adv = model.f_forward(x_adv.to(device)).detach()
            x_recon_adv = model.g_forward(feat_adv.view(-1, 128, 1, 1)).detach()
            feat_recon_adv = model.f_forward(x_recon_adv).detach()
            pred_adv = clf(feat_adv).detach()

            # print(x_adv.shape, feat_adv.shape, x_recon_adv.shape)
            x_adv_all[attack_step].append(x_adv.detach())
            feat_adv_all[attack_step].append(feat_adv)
            x_recon_adv_all[attack_step].append(x_recon_adv)
            feat_recon_adv_all[attack_step].append(feat_recon_adv)
            pred_adv_all[attack_step].append(pred_adv)
    
            # # plot x_adv for the first batch
            if i == 0: 
                # x_adv
                plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.title("Test Images")
                plt.imshow(np.transpose(vutils.make_grid(x_adv[:64]*0.5 + 0.5, padding=2, normalize=False).cpu(), (1,2,0)))
                
                file_name = os.path.join(save_dir, f"test_x_adv_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
                plt.savefig(file_name)
                print("Plot saved to: {}".format(file_name))
                plt.close()

                # x_recon_adv
                plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.title("Test Images")
                plt.imshow(np.transpose(vutils.make_grid(x_recon_adv[:64]*0.5 + 0.5, padding=2, normalize=False).cpu(), (1,2,0)))
                
                file_name = os.path.join(save_dir, f"test_x_recon_adv_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
                plt.savefig(file_name)
                print("Plot saved to: {}".format(file_name))
                plt.close()

        # feat_adv_all.append(feat_adv_batch)
        x_adv_final_all.append(x_adv) # the last one
        x_adv_final_recon_all.append(x_recon_adv)

    ## concatnate tensors
    x_adv_final_all = torch.cat(x_adv_final_all, dim=0)
    x_adv_final_recon_all = torch.cat(x_adv_final_recon_all, dim=0)
    feat_nat_all = torch.cat(feat_nat_all, dim=0)
    feat_recon_nat_all = torch.cat(feat_recon_nat_all, dim=0)
    x_adv_all = [torch.cat(x, dim=0) for x in x_adv_all]
    x_adv_all = torch.stack(x_adv_all)
    x_recon_adv_all = [torch.cat(x, dim=0) for x in x_recon_adv_all]
    x_recon_adv_all = torch.stack(x_recon_adv_all)
    feat_adv_all = [torch.cat(x, dim=0) for x in feat_adv_all]
    feat_adv_all = torch.stack(feat_adv_all)
    feat_recon_adv_all = [torch.cat(x, dim=0) for x in feat_recon_adv_all]
    feat_recon_adv_all = torch.stack(feat_recon_adv_all)
    pred_adv_all = [torch.cat(x, dim=0) for x in pred_adv_all]
    pred_adv_all = torch.stack(pred_adv_all).T.detach().cpu().numpy()
    labels_all = torch.cat(labels_all, dim=0)
    print(x_adv_final_all.shape, x_adv_final_recon_all.shape, feat_nat_all.shape, feat_adv_all.shape, x_adv_all.shape, x_recon_adv_all.shape, feat_recon_adv_all.shape, labels_all.shape)
    print('Adv_acc:{:.4f}'.format(correct / total))
    print((x_adv-x_nat).abs().max(), x_adv.max(), x_adv.min())

    ## EXAMPLE: take one sample of class 0, plot its changing process
    select_class = 0 
    x_adv_one_sample = x_adv_all[:, np.where(labels_all==select_class)[0][0], :, :]
    x_recon_adv_one_sample = x_recon_adv_all[:, np.where(labels_all==select_class)[0][0], :, :]
    feat_adv_one_sample = feat_adv_all[:, np.where(labels_all==select_class)[0][0], :]
    feat_recon_adv_one_sample = feat_recon_adv_all[:, np.where(labels_all==select_class)[0][0], :]
    pred_adv_one_sample = pred_adv_all[:, np.where(labels_all==select_class)[0][0], :]
    print('sample data shape:', x_recon_adv_one_sample.shape)
    
    scores_pca_one_sample_adv, scores_svd_one_sample_adv = nearsub_for_adv(feat_nat_train_all.cpu().detach(), label_nat_train_all, feat_adv_one_sample.cpu().detach(), select_class * torch.ones(len(feat_adv_one_sample)))
    scores_pca_one_sample_recon_adv, scores_svd_one_sample_recon_adv = nearsub_for_adv(feat_nat_train_all.cpu().detach(), label_nat_train_all, feat_recon_adv_one_sample.cpu().detach(), select_class * torch.ones(len(feat_recon_adv_one_sample)))

    print('sample data shape:', scores_svd_one_sample_adv[0].shape)


    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("X")
    plt.imshow(np.transpose(vutils.make_grid(x_adv_one_sample*0.5 + 0.5, padding=2, normalize=False, nrow=len(x_recon_adv_one_sample)).cpu(), (1,2,0)))
    file_name = os.path.join(save_dir, f"test_sample_x_adv_change_process_class{select_class}_x_adv_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("X_recon")
    plt.imshow(np.transpose(vutils.make_grid(x_recon_adv_one_sample*0.5 + 0.5, padding=2, normalize=False, nrow=len(x_recon_adv_one_sample)).cpu(), (1,2,0)))
    file_name = os.path.join(save_dir, f"test_sample_x_recon_adv_change_process_class{select_class}_x_adv_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

    plt.figure(figsize=(8,8))
    plt.title("Scores_SVD")
    x_axis_vals = range(len(x_recon_adv_one_sample))
    colors_names = ['b', 'g','r','c','m','y','k','darkorange','coral','purple']
    for i in range(10):
        plt.plot(x_axis_vals, scores_svd_one_sample_adv[i], label=str(i), color=colors_names[i])
    plt.legend()
    file_name = os.path.join(save_dir, f"test_sample_svd_scores_feat_adv_change_process_class{select_class}_x_adv_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

    plt.figure(figsize=(8,8))
    plt.title("Scores_SVD")
    x_axis_vals = range(len(x_recon_adv_one_sample))
    colors_names = ['b', 'g','r','c','m','y','k','darkorange','coral','purple']
    for i in range(10):
        plt.plot(x_axis_vals, scores_svd_one_sample_recon_adv[i], label=str(i), color=colors_names[i])
    plt.legend()
    file_name = os.path.join(save_dir, f"test_sample_svd_scores_feat_recon_adv_change_process_class{select_class}_x_adv_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

    plt.figure(figsize=(8,8))
    plt.title("Scores_SVD")
    x_axis_vals = range(len(x_recon_adv_one_sample))
    colors_names = ['b', 'g','r','c','m','y','k','darkorange','coral','purple']
    for i in range(10):
        plt.plot(x_axis_vals, pred_adv_one_sample[i], label=str(i), color=colors_names[i])
    plt.legend()
    file_name = os.path.join(save_dir, f"test_sample_logits_adv_change_process_class{select_class}_x_adv_clf_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

    ## sort results by class, samples in them are 1-to-1 matched
    num_classes = labels.numpy().max() + 1
    feat_nat_sort, _ = utils.sort_dataset(feat_nat_all.cpu().detach().numpy(), labels_all.numpy(), 
                            num_classes=num_classes, stack=False)
    feat_adv_sort, _ = utils.sort_dataset(feat_adv_all[-1].cpu().detach().numpy(), labels_all.numpy(), 
                            num_classes=num_classes, stack=False)
    feat_recon_nat_sort, _ = utils.sort_dataset(feat_recon_nat_all.cpu().detach().numpy(), labels_all.numpy(), 
                            num_classes=num_classes, stack=False)
    feat_recon_adv_sort, _ = utils.sort_dataset(feat_recon_adv_all[-1].cpu().detach().numpy(), labels_all.numpy(), 
                            num_classes=num_classes, stack=False)
    x_adv_final_sort, _ = utils.sort_dataset(x_adv_final_all.cpu().detach().numpy(), labels_all.numpy(), 
                            num_classes=num_classes, stack=False)
    x_adv_final_recon_sort, _ = utils.sort_dataset(x_adv_final_recon_all.cpu().detach().numpy(), labels_all.numpy(), 
                            num_classes=num_classes, stack=False)

    ## check norm of features is correct (=1)
    print('clean feat norm:', torch.norm(feat_nat_all, p=2, dim=1))
    print('adv feat norm:', torch.norm(feat_adv_all[-1], p=2, dim=1))

    ## check nearsub accuracy
    print(feat_nat_train_all.shape, feat_adv_all[-1].shape)
    print(f"-----Test Clean class {select_class}-----")
    scores_pca_clean_select_class, scores_svd_clean_select_class = nearsub_for_adv(feat_nat_train_all.cpu().detach(), label_nat_train_all, torch.from_numpy(feat_nat_sort[select_class]), select_class * torch.ones(len(feat_nat_sort[select_class])))
    print(f"-----Test Adv class {select_class}-----")
    scores_pca_adv_select_class, scores_svd_adv_select_class = nearsub_for_adv(feat_nat_train_all.cpu().detach(), label_nat_train_all, torch.from_numpy(feat_adv_sort[select_class]), select_class * torch.ones(len(feat_adv_sort[select_class])))
    print(f"-----Test Z_hat Clean class {select_class}-----")
    scores_pca_clean_Z_hat_select_class, scores_svd_clean_Z_hat_select_class = nearsub_for_adv(feat_nat_train_all.cpu().detach(), label_nat_train_all, torch.from_numpy(feat_recon_nat_sort[select_class]), select_class * torch.ones(len(feat_recon_nat_sort[select_class])))
    print(f"-----Test Z_hat Adv class {select_class}-----")
    scores_pca_adv_Z_hat_select_class, scores_svd_adv_Z_hat_select_class = nearsub_for_adv(feat_nat_train_all.cpu().detach(), label_nat_train_all, torch.from_numpy(feat_recon_adv_sort[select_class]), select_class * torch.ones(len(feat_recon_adv_sort[select_class])))

    print('Clean Z Error Distance:')
    mean_distance = []
    for i in range(10):
        mean_distance.append(scores_svd_clean_select_class[i].mean())
    print("SVD:", mean_distance)

    print('Adv Z Error Distance:')
    mean_distance = []
    for i in range(10):
        mean_distance.append(scores_svd_adv_select_class[i].mean())
    print("SVD:", mean_distance)
    
    print('Clean Z_hat Error Distance:')
    mean_distance = []
    for i in range(10):
        mean_distance.append(scores_svd_clean_Z_hat_select_class[i].mean())
    print("SVD:", mean_distance)
    
    print('Adv Z_hat Error Distance:')
    mean_distance = []
    for i in range(10):
        mean_distance.append(scores_svd_adv_Z_hat_select_class[i].mean())
    print("SVD:", mean_distance)

    print(f"-----Test Clean All-----")
    nearsub_for_adv(feat_nat_train_all.cpu().detach(), label_nat_train_all, feat_nat_all.cpu().detach(), labels_all)
    print(f"-----Test Adv All-----")
    nearsub_for_adv(feat_nat_train_all.cpu().detach(), label_nat_train_all, feat_adv_all[-1].cpu().detach(), labels_all)

    # # clauculate some l2 distance
    # feat_nat_sort_mean = []
    # feat_adv_sort_mean = []
    # for i in range(len(feat_nat_sort)):
    #     feat_nat_sort_mean_i = np.mean(feat_nat_sort[i], axis=0)
    #     feat_nat_sort_mean_i = feat_nat_sort_mean_i / np.sum(feat_nat_sort_mean_i**2)**(0.5)

    #     feat_adv_sort_mean_i = np.mean(feat_adv_sort[i], axis=0)
    #     feat_adv_sort_mean_i = feat_adv_sort_mean_i / np.sum(feat_adv_sort_mean_i**2)**(0.5)

    #     print(np.sum(feat_nat_sort_mean_i**2), np.sum(feat_adv_sort_mean_i**2))

    #     feat_nat_sort_mean.append(feat_nat_sort_mean_i)
    #     feat_adv_sort_mean.append(feat_adv_sort_mean_i)
    
    # feat_nat_sort_mean_9 = feat_nat_sort_mean[9]
    # feat_nat_sort_mean = np.vstack(feat_nat_sort_mean)
    # feat_adv_sort_mean = np.vstack(feat_adv_sort_mean)
    # diff_nat_to_nine = np.sum((feat_nat_sort_mean - feat_nat_sort_mean_9)**2, axis=1)
    # diff_adv_to_nine = np.sum((feat_adv_sort_mean - feat_nat_sort_mean_9)**2, axis=1)
    # print(diff_nat_to_nine, diff_adv_to_nine)
    # print(np.sum(feat_nat_sort_mean**2, axis=1))
    # print(feat_nat_sort_mean @ feat_nat_sort_mean.transpose())
    # print(feat_nat_sort_mean @ feat_adv_sort_mean.transpose())


    # print('mean of distance from samples to mean9')
    # for i in range(len(feat_nat_sort)):
    #     print('class:', i)
    #     print('nat to mean9:', np.mean(np.sum((feat_nat_sort[i] - feat_nat_sort_mean_9)**2, axis=1)))
    #     print('adv to mean9:', np.mean(np.sum((feat_adv_sort[i] - feat_nat_sort_mean_9)**2, axis=1)))


    # ## plot

    # # plot x_adv
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Test Images")
    # plt.imshow(np.transpose(vutils.make_grid(torch.from_numpy(x_adv_final_sort[test_class][:64])*0.5 + 0.5, padding=2, normalize=False).cpu(), (1,2,0)))
    
    # file_name = os.path.join(save_dir, f"test_class{test_class}_x_adv_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
    # plt.savefig(file_name)
    # print("Plot saved to: {}".format(file_name))

    # # plot x_recon_adv
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Test Images")
    # plt.imshow(np.transpose(vutils.make_grid(torch.from_numpy(x_adv_final_recon_sort[test_class][:64])*0.5 + 0.5, padding=2, normalize=False).cpu(), (1,2,0)))
    
    # file_name = os.path.join(save_dir, f"test_class{test_class}_x_recon_adv_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
    # plt.savefig(file_name)
    # print("Plot saved to: {}".format(file_name))

    # plot X_recon of linspace in Z_clean to Z_adv
    
    ## save components
    # nearsub_save_comps(feat_nat_train_all.cpu().detach(), label_nat_train_all, feat_adv_all[-1].cpu().detach(), labels_all, save_dir, load_epoch)

    # x_recon_adv_lin = []
    # lin_sample_num = 100
    # Z_start = torch.from_numpy(feat_nat_sort[0][1]) 
    # Z_end = torch.from_numpy(feat_adv_sort[0][1])

    # for i in range(lin_sample_num):
    #     feat_lin = Z_start + (Z_end - Z_start) * i / lin_sample_num
    #     feat_lin = feat_lin.view(1,128,1,1).to(DEVICE)
    #     x_recon_adv_lin_ = netG(feat_lin)
    #     x_recon_adv_lin.append(x_recon_adv_lin_)
    
    # x_recon_adv_lin = torch.cat(x_recon_adv_lin, dim=0)
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Test Images")
    # plt.imshow(np.transpose(vutils.make_grid(x_recon_adv_lin*0.5 + 0.5, padding=2, normalize=False).cpu(), (1,2,0)))
    
    # file_name = os.path.join(save_dir, f"X_recon_from_Znat_to_Zadv_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.png")
    # plt.savefig(file_name)
    # print("Plot saved to: {}".format(file_name))

    # # plot heatmaps
    # feat_nat_sort_ = np.vstack(feat_nat_sort)
    # feat_adv_sort_ = np.vstack(feat_adv_sort)

    # sim_mat_nat_adv = np.abs(feat_nat_sort_ @ feat_adv_sort_.T)
    # sim_mat_nat = np.abs(feat_nat_sort_ @ feat_nat_sort_.T)
    # sim_mat_adv = np.abs(feat_adv_sort_ @ feat_adv_sort_.T)

    # print(sim_mat_nat_adv.shape)
    # print("Z_clean X Z_clean:", feat_nat_sort_[:10] @ feat_nat_sort_[:10].T)
    # print("Z_clean X Z_adv:", feat_nat_sort_[:10] @ feat_adv_sort_[:10].T)
    # print("Z_adv X Z_adv:", feat_adv_sort_[:10] @ feat_adv_sort_[:10].T)

    # for i in range(10):
    #     for j in range(10):
    #         # if i != j:
    #         print(f'mean cos-sim of adv {i}&{j}:', np.abs((feat_adv_sort[i] @ feat_adv_sort[j].T)).mean())


    # plt.rc('text', usetex=False)
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']

    # fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    # im = ax.imshow(sim_mat_nat_adv, cmap='Blues')
    # # im = ax.imshow(sim_mat, cmap='bwr')
    # fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    # ax.set_xticks(np.linspace(0, len(feat_nat_sort_), 6))
    # ax.set_yticks(np.linspace(0, len(feat_nat_sort_), 6))
    # [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    # [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    # fig.tight_layout()

    # file_name = os.path.join(save_dir, f"Z_nat_Z_adv_heatmat_subspace_attack_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.pdf")
    # fig.savefig(file_name)
    # # print("Plot saved to: {}".format(file_name))
    # # file_name = os.path.join(save_dir, f"ZnZ_noise_heatmat_epoch{epoch}.pdf")
    # # fig.savefig(file_name)
    # print("Plot saved to: {}".format(file_name))
    # plt.close()

    # fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    # im = ax.imshow(sim_mat_nat, cmap='Blues')
    # # im = ax.imshow(sim_mat, cmap='bwr')
    # fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    # ax.set_xticks(np.linspace(0, len(feat_nat_sort_), 6))
    # ax.set_yticks(np.linspace(0, len(feat_nat_sort_), 6))
    # [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    # [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    # fig.tight_layout()

    # file_name = os.path.join(save_dir, f"Z_nat_Z_nat_subspace_attack_heatmat_epo{load_epoch}.pdf")
    # fig.savefig(file_name)
    # # print("Plot saved to: {}".format(file_name))
    # # file_name = os.path.join(save_dir, f"ZnZ_noise_heatmat_epoch{epoch}.pdf")
    # # fig.savefig(file_name)
    # print("Plot saved to: {}".format(file_name))
    # plt.close()

    # fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    # im = ax.imshow(sim_mat_adv, cmap='Blues')
    # # im = ax.imshow(sim_mat, cmap='bwr')
    # fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    # ax.set_xticks(np.linspace(0, len(feat_nat_sort_), 6))
    # ax.set_yticks(np.linspace(0, len(feat_nat_sort_), 6))
    # [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    # [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    # fig.tight_layout()

    # file_name = os.path.join(save_dir, f"Z_adv_Z_adv_subspace_attack_heatmat_step{attack_step}_epo{load_epoch}_pgd{pgd_perturb_steps}_eps{pgd_epsilon/2}.pdf")
    # fig.savefig(file_name)
    # # print("Plot saved to: {}".format(file_name))
    # # file_name = os.path.join(save_dir, f"ZnZ_noise_heatmat_epoch{epoch}.pdf")
    # # fig.savefig(file_name)
    # print("Plot saved to: {}".format(file_name))
    # plt.close()

def vis_sparse(args, train_features, test_features):
    mean_train_l1_norm = train_features.abs().sum(dim=1).mean()
    mean_test_l1_norm = test_features.abs().sum(dim=1).mean()
    print(mean_train_l1_norm, mean_test_l1_norm)

def get_inception_feature(
    images: Union[List[torch.FloatTensor], DataLoader],
    dims: List[int],
    batch_size: int = 50,
    use_torch: bool = False,
    verbose: bool = False,
    device: torch.device = torch.device('cuda:0'),
) -> Union[torch.FloatTensor, np.ndarray]:
    """Calculate Inception Score and FID.
    For each image, only a forward propagation is required to
    calculating features for FID and Inception Score.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
            must be float tensor of range [0, 1].
        dims: List of int, see InceptionV3.BLOCK_INDEX_BY_DIM for
            available dimension.
        batch_size: int, The batch size for calculating activations. If
            `images` is torch.utils.data.Dataloader, this argument is
            ignored.
        use_torch: bool. The default value is False and the backend is same as
            official implementation, i.e., numpy. If use_torch is enableb,
            the backend linalg is implemented by torch, the results are not
            guaranteed to be consistent with numpy, but the speed can be
            accelerated by GPU.
        verbose: Set verbose to False for disabling progress bar. Otherwise,
            the progress bar is showing when calculating activations.
        device: the torch device which is used to calculate inception feature
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    assert all(dim in IC.BLOCK_INDEX_BY_DIM for dim in dims)

    is_dataloader = isinstance(images, DataLoader)
    if is_dataloader:
        num_images = min(len(images.dataset), images.batch_size * len(images))
        batch_size = images.batch_size
    else:
        num_images = len(images)

    block_idxs = [IC.BLOCK_INDEX_BY_DIM[dim] for dim in dims]
    model = IC(block_idxs).to(device)
    model.eval()

    if use_torch:
        features = [torch.empty((num_images, dim)).to(device) for dim in dims]
    else:
        features = [np.empty((num_images, dim)) for dim in dims]

    pbar = tqdm(
        total=num_images, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_feature")
    looper = iter(images)
    start = 0
    while start < num_images:
        # get a batch of images from iterator
        if is_dataloader:
            batch_images = next(looper)
        else:
            batch_images = images[start: start + batch_size]
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.tensor(batch_images).to(device)
        with torch.no_grad():
            outputs = model(batch_images)
            for feature, output, dim in zip(features, outputs, dims):
                if use_torch:
                    feature[start: end] = output.view(-1, dim)
                else:
                    feature[start: end] = output.view(-1, dim).cpu().numpy()
        start = end
        pbar.update(len(batch_images))
    pbar.close()
    return features

def calculate_inception_score(
    probs: Union[torch.FloatTensor, np.ndarray],
    splits: int = 10,
    use_torch: bool = False,
) -> Tuple[float, float]:
    # Inception Score
    scores = []
    for i in range(splits):
        part = probs[
            (i * probs.shape[0] // splits):
            ((i + 1) * probs.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        inception_score = torch.mean(scores).cpu().item()
        std = torch.std(scores).cpu().item()
    else:
        inception_score, std = (np.mean(scores), np.std(scores))
    del probs, scores
    return inception_score, std


def get_activations(images, model, batch_size=64, dims=2048, device=None):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : torch.Device

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        #batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = images[start:end].type(torch.FloatTensor)
        
        if device is not None:
            batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            print(pred.shape)

        pred_arr[start:end] = pred.cpu().numpy().reshape(batch_size, -1)

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    # print(diff.dot(diff)) 
    # print(np.trace(sigma1))
    # print(np.trace(sigma2))
    # print(- 2 * tr_covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(images, model, batch_size=64, dims=2048, device=None):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--adv_attack', action='store_true', help='do pgd attack')
    parser.add_argument('--svm', help='evaluate using SVM', action='store_true')
    parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
    parser.add_argument('--nearsub', help='evaluate using Nearest Subspace', action='store_true')
    parser.add_argument('--kmeans', help='evaluate using KMeans', action='store_true')
    parser.add_argument('--vis_sparse', help='visualize sparsity', action='store_true')
    parser.add_argument('--ensc', help='evaluate using Elastic Net Subspace Clustering', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    parser.add_argument('--pca', help='plot PCA singular values of feautres', action='store_true')
    parser.add_argument('--tsne', help='plot tsne of feautres', action='store_true')
    parser.add_argument('--nearcomp_sup', help='plot nearest component', action='store_true')
    parser.add_argument('--nearcomp_unsup', help='plot nearest component', action='store_true')
    parser.add_argument('--heat', help='plot heatmap of cosine similarity between samples', action='store_true')
    parser.add_argument('--hist', help='plot histogram of cosine similarity of features', action='store_true')
    parser.add_argument('--IS', help='eval is', action='store_true')
    parser.add_argument('--FID', help='eval fid', action='store_true')
    
    parser.add_argument('--k', type=int, default=5, help='top k components for kNN')
    parser.add_argument('--n', type=int, default=10, help='number of clusters for cluster (default: 10)')
    parser.add_argument('--gam', type=int, default=300, 
                        help='gamma paramter for subspace clustering (default: 100)')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='tau paramter for subspace clustering (default: 1.0)')
    parser.add_argument('--n_comp', type=int, default=100, help='number of components for PCA (default: 30)')
    parser.add_argument('--save', action='store_true', help='save labels')
    parser.add_argument('--data_dir', default='./data/', help='path to dataset')
    args = parser.parse_args()

    ## load model
    if args.dataset == 'MNIST':
        nz = 128
        ngf = 64
        nc = 1
    elif args.dataset == 'CIFAR10':
        nz = 512
        ngf = 64
        nc = 3
    else:
        raise NameError('Unsupported dataset.')

    from models.sdnet_DCGAN import Generator, Discriminator, InverseNet2
    model = InverseNet2(nz, ngf, nc).to(device)

    with torch.no_grad():
        print("====================")
        inputx = torch.zeros([100, nc, 32, 32]).cuda()
        # print(inputx)
        _ = model.f_forward(inputx)
        print("====================")


    if args.epoch is None: # get last epoch
        ckpt_dir = os.path.join(args.model_dir, 'checkpoints')
        epochs = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
        epoch = np.sort(epochs)[-1]
    ckpt_path = os.path.join(args.model_dir, 'checkpoints', 'model-epoch{}.pt'.format(args.epoch))

    print('Loading checkpoint: {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    # model.eval()
    
    # set dataset
    if args.dataset == 'CIFAR10':
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, drop_last=False,
                                                shuffle=False, num_workers=2)
        testset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200, drop_last=False,
                                                shuffle=False, num_workers=2)
    
    elif args.dataset == 'MNIST':
        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])
        trainset = datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, drop_last=False,
                                                shuffle=False, num_workers=2)
        testset = datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200, drop_last=False,
                                                shuffle=False, num_workers=2)
    from config import config as _cfg
    cfg = _cfg
    noise_sd = 0.5
    lamda = cfg['MODEL']['LAMBDA'][0]
    # get train features and labels
    train_features, train_labels, train_X_hat, train_X = get_features(model, trainloader)
    # train_features, train_labels = get_data(trainloader)
    train_features_noisy, train_labels_noisy, train_X_hat_noisy, train_X_noisy  = get_features_gaussian_noise(model, trainloader, noise_sd=noise_sd)

    # get test features and labels
    test_features, test_labels, test_X_hat, test_X  = get_features(model, testloader)
    test_features_noisy, test_labels_noisy, test_X_hat_noisy, test_X_noisy  = get_features_gaussian_noise(model, testloader, noise_sd=noise_sd)

    # test_features, test_labels = get_data(testloader)
    plot_recon_imgs(train_X, train_X_hat, test_X, test_X_hat) # Figure 3,5,8,10 -- draw reconstruction results
    # print(train_X.shape)
    # utils.save_fig(vutils.make_grid(train_X[:64].detach().cpu(), padding=2, normalize=True), args.epoch, 0, model_dir=os.path.join(args.model_dir, 'figures'), tail='train_X')
    # utils.save_fig(vutils.make_grid(train_X_hat[:64].detach().cpu(), padding=2, normalize=True), args.epoch, 0, model_dir=os.path.join(args.model_dir, 'figures'), tail='train_X_hat')
    utils.save_fig(vutils.make_grid(test_X_noisy[:64].detach().cpu(), padding=2, normalize=True), args.epoch, 0, model_dir=os.path.join(args.model_dir, 'figures'), tail=f'test_X_noisy_{noise_sd}')
    utils.save_fig(vutils.make_grid(test_X_hat_noisy[:64].detach().cpu(), padding=2, normalize=True), args.epoch, 0, model_dir=os.path.join(args.model_dir, 'figures'), tail=f'test_X_hat_noisy{noise_sd}_lambda{lamda}')
    utils.save_fig(vutils.make_grid(train_X_noisy[:64].detach().cpu(), padding=2, normalize=True), args.epoch, 0, model_dir=os.path.join(args.model_dir, 'figures'), tail=f'train_X_noisy_{noise_sd}')
    utils.save_fig(vutils.make_grid(train_X_hat_noisy[:64].detach().cpu(), padding=2, normalize=True), args.epoch, 0, model_dir=os.path.join(args.model_dir, 'figures'), tail=f'train_X_hat_noisy{noise_sd}_lambda{lamda}')


    if args.svm:
        svm(args, train_features, train_labels, test_features, test_labels)
    if args.knn:
        knn(args, train_features, train_labels, test_features, test_labels)
    if args.nearsub:
        nearsub(args, train_features, train_labels, test_features, test_labels)
        nearsub_save_comps(args, train_features, train_labels, test_features, test_labels)

        # nearsub(args, train_X.view(len(train_X), -1), train_labels, test_X.view(len(test_X), -1), test_labels)
    if args.vis_sparse:
        vis_sparse(args, train_features, test_features)
    if args.kmeans:
        kmeans(args, train_features, train_labels)
    if args.ensc:
        ensc(args, train_features, train_labels)
    
    if args.pca:
        plot_pca(args, train_features, train_labels, args.epoch)
    if args.nearcomp_sup:
        plot_nearest_component_supervised(args, train_features, train_labels, args.epoch, trainset)
    if args.nearcomp_unsup:
        plot_nearest_component_unsupervised(args, train_features, train_labels, args.epoch, trainset)
    # if args.nearcomp_class:
    #     plot_nearest_component_class(args, train_features, train_labels, args.epoch, trainset)
    #     plot_nearest_component_class_X_hat(args, train_features, train_labels, args.epoch, train_X_hat) # Figure 6,11 -- draw reconstruction along components
    if args.hist:
        plot_hist(args, train_features, train_labels, args.epoch)
    if args.heat:
        plot_heatmap(args, train_features, train_labels, args.epoch)
        # plot_heatmap_ZnZ_hat(args, train_features, train_Z_bar, train_labels, args.epoch) # Figure 4 -- draw similarity heatmap of Z and Z_hat
    # if args.tsne:
        # plot_tsne_all(args, train_Z, train_Z_bar, train_labels, args.epoch)
        
    # ## run --pca first
    # if args.random_gen:
    #     plot_random_gen(args, netG) # Figure 9,12 -- draw random generated images
    # if args.lin_gen:
    #     plot_linear_gen(args, netG)

    # if args.inter_sample:
    #     plot_samples_interpolation(args, train_Z, train_labels, args.epoch, netG, netD)

    if args.adv_attack:
        train_clf_and_adv_attack_on_subspace(model, trainloader, testloader, args.model_dir)
    
    if args.IS:
        
        acts, probs = get_inception_feature(
            train_X_hat, dims=[2048, 1008], use_torch=True)
        inception_score, std = calculate_inception_score(probs, 10, True)
        print("Inception Score:", inception_score, "Inception STD:", std)
    
    if args.FID:
        from models.inception import InceptionV3
        inception_model = InceptionV3().to(device)
        # Can change device here
        mu_fake, sigma_fake = calculate_activation_statistics(
            train_X_hat, inception_model, 100, device="cuda"
        )
        mu_real, sigma_real = calculate_activation_statistics(
            train_X_hat, inception_model, 100, device="cuda"
        )
        fid_score = calculate_frechet_distance(
            mu_fake, sigma_fake, mu_real, sigma_real
        )
        print("FID is:", fid_score)
