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
        alpha_q, alpha_p, alpha_p_sparsemax, features = self.encoder(x, c)

        # logits go into the RelaxedOnehotCategorical
        z = sample_q(alpha_q, train=True, temperature=0.67) # Lambda(self._sampling_concrete)(alpha)

        recon_x = self.decoder(z, c)

        return recon_x, alpha_q, alpha_p, alpha_p_sparsemax, self.encoder.linear_latent_q, self.encoder.linear_latent_p, z, features

    def inference(self, n=1, c=None):

        batch_size = n

        alpha_q, alpha_p, alpha_p_sparsemax, features = self.encoder(x=torch.empty((0,0)), c=c, train=False)

        # Set to normalized filtered distribution
        #alpha_p = torch.Tensor([[0., 0.25066507, 0.1732346, 0.19631243, 0.13850342, 0., 0., 0., 0., 0.24128452],\
        #    [0.13482085, 0., 0., 0., 0.18401708, 0.1873733, 0.23260461, 0.15368521, 0.10749889, 0.]]).cuda()

        # sample from the prior distribution
        # z = sample_p(alpha_p) # to_var(torch.randn([batch_size, self.latent_size]))

        z = torch.eye(batch_size).cuda()

        recon_x = self.decoder(z, c)

        return recon_x, alpha_p, alpha_p_sparsemax, self.encoder.linear_latent_p, features, z

class Encoder(nn.Module):

    def __init__(self, layer_sizes_q, layer_sizes_p, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes_q[0] += num_labels

        self.MLP_q = nn.Sequential()

        for i, (in_size, out_size) in enumerate( zip(layer_sizes_q[:-1], layer_sizes_q[1:]) ):
            self.MLP_q.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            self.MLP_q.add_module(name="A%i"%(i), module=nn.ReLU())

        self.MLP_p = nn.Sequential()
        
        if self.conditional:

            layer_sizes_p[0] = num_labels

            for i, (in_size, out_size) in enumerate( zip(layer_sizes_p[:-1], layer_sizes_p[1:]) ):
                self.MLP_p.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
                self.MLP_p.add_module(name="A%i"%(i), module=nn.ReLU())

        self.linear_latent_q = nn.Linear(layer_sizes_q[-1], latent_size)
        self.softmax_q = nn.Softmax(dim=-1)

        self.linear_latent_p = nn.Linear(layer_sizes_p[-1], latent_size)
        self.softmax_p = nn.Softmax(dim=-1)

        self.sparsemax_p = Sparsemax(dim=-1)

    def forward(self, x=None, c=None, train=True):

        if train: 
            if self.conditional:
                c = idx2onehot(c, n=2) # used to be 10
                full_x = torch.cat((x, c), dim=-1)
            else:
                full_x = x

            full_x = self.MLP_q(full_x)

            alpha_q_lin = self.linear_latent_q(full_x)
            alpha_q = self.softmax_q(alpha_q_lin)

        else:
            alpha_q_lin = None
            alpha_q = None
            if self.conditional:
                c = idx2onehot(c, n=2) # used to be 10

        c = self.MLP_p(c)

        alpha_p_lin = self.linear_latent_p(c)
        alpha_p = self.softmax_p(alpha_p_lin)
        alpha_p_sparsemax = self.sparsemax_p(alpha_p_lin)

        return alpha_q, alpha_p, alpha_p_sparsemax, c


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size # + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate( zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A%i"%(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):
        x = self.MLP(z)

        return x
