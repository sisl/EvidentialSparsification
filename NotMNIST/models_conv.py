# Code modified from: https://github.com/timbmg/VAE-CVAE-MNIST/

seed = 123

import numpy as np
np.random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import torch.nn as nn
from sparsemax import Sparsemax

from utils import to_var, idx2onehot, sample_q, sample_p

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes_q, encoder_layer_sizes_p, latent_size, decoder_layer_sizes, conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes_q) == list
        assert type(encoder_layer_sizes_p) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(encoder_layer_sizes_q, encoder_layer_sizes_p, latent_size, conditional, num_labels)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        batch_size = x.size(0)

        # Sample from latent distributions
        alpha_q, alpha_p, alpha_p_max, features = self.encoder(x, c)

        # logits go into the RelaxedOnehotCategorical
        z = sample_q(alpha_q, train=True, temperature=0.67)

        recon_x = self.decoder(z)

        return recon_x, alpha_q, alpha_p, alpha_p_max, self.encoder.linear_latent_q, self.encoder.linear_latent_p, z, features

    def inference(self, n=1, c=None):

        batch_size = n

        alpha_q, alpha_p, alpha_p_max, features = self.encoder(x=torch.empty((0,0)), c=c, train=False)

        z = torch.eye(batch_size).cuda()

        recon_x = self.decoder(z)

        return recon_x, alpha_p, alpha_p_max, self.encoder.linear_latent_p, features, z

class Encoder(nn.Module):

    def __init__(self, layer_sizes_q, layer_sizes_p, latent_size, conditional, num_labels):

        super().__init__()

        self.num_labels = num_labels

        self.conditional = conditional
        if self.conditional:
            layer_sizes_q[0] += num_labels

        self.MLP_q = nn.Sequential()
        
        # conv architecture inspired by: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cnn-cvae_no-out-concat.ipynb
        self.MLP_q.add_module(name="C1", module=nn.Conv2d(in_channels=1 + self.num_labels,
                                                               out_channels=32,
                                                               kernel_size=6,
                                                               stride=2,
                                                               padding=0))
        self.MLP_q.add_module(name="A1", module=nn.ReLU())
        self.MLP_q.add_module(name="C2", module=nn.Conv2d(in_channels=32,
                                                               out_channels=64,
                                                               kernel_size=4,
                                                               stride=2,
                                                               padding=1))
        self.MLP_q.add_module(name="A2", module=nn.ReLU())
        self.MLP_q.add_module(name="C3", module=nn.Conv2d(in_channels=64,
                                                               out_channels=128,
                                                               kernel_size=2,
                                                               stride=2,
                                                               padding=1))
        self.MLP_q.add_module(name="A3", module=nn.ReLU())
        self.MLP_q.add_module(name="F", module=View((-1, 128*4*4)))

        self.MLP_p = nn.Sequential()
        
        if self.conditional:

            layer_sizes_p[0] = num_labels

            for i, (in_size, out_size) in enumerate( zip(layer_sizes_p[:-1], layer_sizes_p[1:]) ):
                self.MLP_p.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
                self.MLP_p.add_module(name="A%i"%(i), module=nn.ReLU())

        self.linear_latent_q = nn.Linear(128*4*4, latent_size)
        self.softmax_q = nn.Softmax(dim=-1)

        self.linear_latent_p = nn.Linear(layer_sizes_p[-1], latent_size)
        self.softmax_p = nn.Softmax(dim=-1)

        self.sparsemax_p = Sparsemax(dim=-1)

    def forward(self, x=None, c=None, train=True):

        if train: 
            if self.conditional:
                c = idx2onehot(c, n=self.num_labels) # used to be 10

                # concatenate labels to the images as in:
                # https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cnn-cvae_no-out-concat.ipynb
                c_concat = c.view(-1, self.num_labels, 1, 1)
                c_concat = torch.ones(c.size()[0],
                                      self.num_labels, 
                                      x.size()[2], 
                                      x.size()[3], 
                                      dtype=x.dtype).cuda() * c_concat

                full_x = torch.cat((x, c_concat), dim=1)

            full_x = self.MLP_q(full_x)

            alpha_q_lin = self.linear_latent_q(full_x)
            alpha_q = self.softmax_q(alpha_q_lin)

        else:
            alpha_q_lin = None
            alpha_q = None
            if self.conditional:
                c = idx2onehot(c, n=2) 

        c = self.MLP_p(c)

        alpha_p_lin = self.linear_latent_p(c)
        alpha_p = self.softmax_p(alpha_p_lin)
        alpha_p_max = self.sparsemax_p(alpha_p_lin)

        return alpha_q, alpha_p, alpha_p_max, c


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size # + num_labels

        self.MLP.add_module(name="L1", module=nn.Linear(latent_size, 4*4*128))
        self.MLP.add_module(name="R", module=View((-1, 128, 4, 4)))

        self.MLP.add_module(name="C1", module=nn.ConvTranspose2d(in_channels=128,
                                                               out_channels=64,
                                                               kernel_size=2,
                                                               stride=2,
                                                               padding=1))
        self.MLP.add_module(name="A1", module=nn.ReLU())
        self.MLP.add_module(name="C2", module=nn.ConvTranspose2d(in_channels=64,
                                                               out_channels=32,
                                                               kernel_size=4,
                                                               stride=2,
                                                               padding=1))
        self.MLP.add_module(name="A2", module=nn.ReLU())
        self.MLP.add_module(name="C3", module=nn.ConvTranspose2d(in_channels=32,
                                                               out_channels=1,
                                                               kernel_size=6,
                                                               stride=2,
                                                               padding=0))
        self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)

        return x
