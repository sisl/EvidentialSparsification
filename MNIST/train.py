# Code modified from: "https://github.com/timbmg/VAE-CVAE-MNIST/
# Discrete latent variable code modelled after: https://github.com/EmilienDupont/vae-concrete/

import os
import time

seed = 123

import numpy as np
np.random.seed(seed) # original: 123, 135
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from copy import deepcopy

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from collections import OrderedDict, defaultdict

from utils import to_var
from models import VAE
from sparsemax import Sparsemax

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def main(args):

    random = deepcopy(seed)

    ts = time.time()

    datasets = OrderedDict()
    datasets['train'] = MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    
    tracker_global_train = defaultdict(torch.cuda.FloatTensor)
    tracker_global_test = defaultdict(torch.cuda.FloatTensor)

    total = np.sum(datasets['train'].targets.data.cpu().numpy() >= 0)
    num_even = np.sum(datasets['train'].targets.data.cpu().numpy() % 2 == 0)
    num_odd = np.sum(datasets['train'].targets.data.cpu().numpy() % 2 == 1)
    
    print("train shape after", total, num_even + num_odd, num_even, num_odd)
    tracker_global_train['num_even'] = num_even
    tracker_global_train['num_odd'] = num_odd

    def loss_fn(recon_x, x, q_dist, p_dist):
        """
        Variational Auto Encoder loss.
        """
        x = x.view(-1)
        recon_x = recon_x.view(-1)
        rec_loss = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
        
        kl_disc_loss = torch.distributions.kl.kl_divergence(q_dist, p_dist) # kl_discrete(q_dist, p_dist)
        kl_disc_loss = torch.mean(kl_disc_loss, dim=0, keepdim=True)
        kl_disc_loss = torch.sum(kl_disc_loss)
        
        return rec_loss + kl_disc_loss, rec_loss, kl_disc_loss

    vae = VAE(
        encoder_layer_sizes_q=args.encoder_layer_sizes_q,
        encoder_layer_sizes_p=args.encoder_layer_sizes_p,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels= 2 if args.conditional else 0 # used to be 10
        )

    vae = vae.cuda()
    optimizer = torch.optim.SGD(vae.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for split, dataset in datasets.items():

            print("split", split, epoch)

            data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=split=='train')

            for iteration, (x, y) in enumerate(data_loader):

                x = to_var(x)

                # convert to even or odd
                y = to_var(y % 2)

                x = x.view(-1, 784)
                y = y.view(-1, 1).cuda()

                if args.conditional:
                    recon_x, alpha_q, alpha_p, alpha_p_sparsemax, alpha_q_lin, alpha_p_lin, z, features = vae(x, y)
                else:
                    recon_x, alpha_q, alpha_p, alpha_p_sparsemax, alpha_q_lin, alpha_p_lin, z, features = vae(x)

                for i, yi in enumerate(y.data):
                    id = len(tracker_epoch)
                    tracker_epoch[id]['x'] = z[i, 0].data
                    tracker_epoch[id]['y'] = z[i, 1].data
                    tracker_epoch[id]['label'] = yi[0]

                # Form distributions out of alpha_q and alpha_p
                q_dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=alpha_q)
                p_dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=alpha_p)

                loss, rec, kl = loss_fn(recon_x, x, q_dist, p_dist)

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                    

                tracker_global_train['loss'] = torch.cat((tracker_global_train['loss'], (loss.data/x.size(0)).unsqueeze(-1)))
                tracker_global_train['it'] = torch.cat((tracker_global_train['it'], torch.Tensor([epoch*len(data_loader)+iteration]).cuda()))

                # if ((iteration == len(data_loader)-1) and (epoch == args.epochs - 1)):
                if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                    print("Batch %04d/%i, Loss %9.4f"%(iteration, len(data_loader)-1, loss.data))
                    print("recon_x", torch.max(recon_x))
                    print("recon", rec, "kl", kl)
                    # print("check", alpha_p)

                    plt.figure()
                    plt.figure(figsize=(10,20))

                    if args.conditional:
                        # c= to_var(torch.arange(0,2).repeat(1,10).long().view(-1,1)) # to_var(torch.arange(0,10).long().view(-1,1))
                        c= to_var(torch.arange(0,2).long().view(-1,1)) # to_var(torch.arange(0,10).long().view(-1,1))

                        # c= to_var(torch.arange(0,2).long().view(-1,1)) # to_var(torch.arange(0,10).long().view(-1,1))
                        # x, alpha_p, linear_p, features, z = vae.inference(n=c.size(0), c=c)
                        x, alpha_p, alpha_p_sparsemax, linear_p, features, z = vae.inference(n=10, c=c)

                        if 'x' in tracker_global_train.keys():
                            tracker_global_train['z'] = torch.cat((tracker_global_train['z'], torch.unsqueeze(z, dim=-1)), dim=-1)
                            tracker_global_train['x'] = torch.cat((tracker_global_train['x'], torch.unsqueeze(x, dim=-1)), dim=-1)
                            tracker_global_train['alpha_p'] = torch.cat((tracker_global_train['alpha_p'], torch.unsqueeze(alpha_p, dim=-1)), dim=-1)
                            tracker_global_train['alpha_p_sparsemax'] = torch.cat((tracker_global_train['alpha_p_sparsemax'], torch.unsqueeze(alpha_p_sparsemax, dim=-1)), dim=-1)
                            tracker_global_train['weight'] = torch.cat((tracker_global_train['weight'], torch.unsqueeze(linear_p.weight, dim=-1)), dim=-1)
                            tracker_global_train['bias'] = torch.cat((tracker_global_train['bias'], torch.unsqueeze(linear_p.bias, dim=-1)), dim=-1)
                            tracker_global_train['features'] = torch.cat((tracker_global_train['features'], torch.unsqueeze(features, dim=-1)), dim=-1)
                            tracker_global_train['c'] = torch.cat((tracker_global_train['c'], torch.unsqueeze(c, dim=-1)), dim=-1)

                        else:
                            tracker_global_train['z'] = torch.unsqueeze(z, dim=-1)
                            tracker_global_train['x'] = torch.unsqueeze(x, dim=-1)
                            tracker_global_train['alpha_p'] = torch.unsqueeze(alpha_p, dim=-1)
                            tracker_global_train['alpha_p_sparsemax'] = torch.unsqueeze(alpha_p_sparsemax, dim=-1)
                            tracker_global_train['weight'] = torch.unsqueeze(linear_p.weight, dim=-1)
                            tracker_global_train['bias'] = torch.unsqueeze(linear_p.bias, dim=-1)
                            tracker_global_train['features'] = torch.unsqueeze(features, dim=-1) 
                            tracker_global_train['c'] = torch.unsqueeze(c, dim=-1)       

                    else:
                        x, alpha_p, alpha_p_sparsemax, linear_p, features = vae.inference(n=c.size(0))

                    folder_name = str(ts) + "_mnist_SGD_random_" + str(random)
                    
                    if not os.path.exists(os.path.join(args.fig_root, folder_name)):
                        if not(os.path.exists(os.path.join(args.fig_root))):
                            os.mkdir(os.path.join(args.fig_root))
                        os.mkdir(os.path.join(args.fig_root, folder_name))

                    z_folder = os.path.join(folder_name, "epoch_%i_iter_%i/"%(epoch, iteration))

                    if not os.path.exists(os.path.join(args.fig_root, z_folder)):
                        if not(os.path.exists(os.path.join(args.fig_root))):
                            os.mkdir(os.path.join(args.fig_root))
                        os.mkdir(os.path.join(args.fig_root, z_folder))
                    
                    # for p in range(c.size(0)):
                    for p in range(10):
                        plt.clf()
                        plt.close()
                        plt.imshow(x[p].view(28,28).data.cpu().numpy())
                        plt.axis('off')

                        plt.savefig(os.path.join(args.fig_root, z_folder, "%i.png"%(p)), dpi=300)
                        plt.clf()
                        plt.close()

    # Plot losses
    plt.plot(tracker_global_train['it'].data.cpu().numpy(), tracker_global_train['loss'].data.cpu().numpy())
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.ylim(0,550)
    plt.savefig(os.path.join(args.fig_root, folder_name, "loss.png"))
    plt.clf()
    plt.close()

    # Save data
    torch.save(tracker_global_train, 'tracker_mnist_SGD_random_' + str(random) + '.pt')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20) # used to be 10
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001) # used to be 0.001
    parser.add_argument("--encoder_layer_sizes_q", type=list, default=[784, 256])
    parser.add_argument("--encoder_layer_sizes_p", type=list, default=[2, 30])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=10) # number of latent categories
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)
