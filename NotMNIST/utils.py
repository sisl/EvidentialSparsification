# Code modified from: https://github.com/timbmg/VAE-CVAE-MNIST/

seed = 123

import numpy as np
np.random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def idx2onehot(idx, n):

    assert idx.size(1) == 1
    assert torch.max(idx).data < n

    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx.data, 1)
    onehot = to_var(onehot)
    
    return onehot

# Function from: https://github.com/EmilienDupont/vae-concrete/
def kl_discrete(alpha, num_classes = 10, epsilon=1e-8):
    """
    KL divergence between a uniform distribution over num_cat categories and
    dist.
    Parameters
    ----------
    alpha : Tensor - shape (None, num_categories)
    num_cat : int
    """
    alpha_sum = torch.sum(alpha, axis=1)  # Sum over columns, this now has size (batch_size,)
    alpha_neg_entropy = torch.sum(alpha * torch.log(alpha + epsilon), axis=1)
    return np.log(num_classes) + torch.mean(alpha_neg_entropy - alpha_sum)

def kl_q_p(q_dist, p_dist):
    kl_separated = td.kl_divergence(self.q_dist, self.p_dist)
    if len(kl_separated.size()) < 2:
        kl_separated = torch.unsqueeze(kl_separated, dim=0)
        
    kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)
    kl = torch.sum(kl_minibatch)
        
    return kl

def sample_q(alpha, train=True, temperature=0.67):
	
    if train:
        zdist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(torch.tensor([temperature]).cuda(), probs = alpha)
        sample = zdist.rsample()
    else:
        zdist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=alpha)
        sample = zdist.sample()

    return sample

def sample_p(alpha):
    zdist = torch.distributions.one_hot_categorical.OneHotCategorical(probs = alpha)
    return zdist.sample()
