''' Dempster-Shafer class for softmax parameters to be converted to masses.'''
import numpy as np
import itertools
from dst_utils import *

class DST:
    # output of the linear p layer as input
    # weights are of shape [in_features, out_features] -> [J, K]
    # bias is of shape [out_shape, 1] -> [K, 1] # [N, *, K]
    # features are of shape [N, *, in_features] -> [N, *, J]
    # y = xW^T + b
    def weights_from_linear_layer(self, weights, bias, features, mean):
        self.weights = weights
        self.bias = bias
        self.features = features

        self.J = self.weights.shape[0]
        self.K = self.weights.shape[1]
        self.N = self.features.shape[0]

        # center the rows: [J, K]
        self.beta = self.weights - np.repeat(np.mean(self.weights, axis = 1)[:,None], self.K, axis=-1)
        
        # centered bias: [K, 1]
        bias_centered = self.bias - np.mean(self.bias)

        # mean: [J,]
        # features: [N, *, J]
        # alpha: [J, K]
        self.alpha = 1. / self.J * (np.squeeze(bias_centered) + self.beta.T.dot(mean)) - self.beta * np.repeat(mean[:, None], self.K, axis=-1)

        self.evidential_weights = np.repeat(self.beta[None,:,:], self.N, axis=0) * np.repeat(self.features[:,:,None], self.K, axis=-1) + self.alpha

        # compute the positive and negative weights: [N, J, K]
        self.evidential_weights_pos_jk = np.maximum(0, self.evidential_weights)
        self.evidential_weights_neg_jk = np.maximum(0, -1 * self.evidential_weights)

        # compute the positive and negative weights: [N, K] -> sum over J
        self.evidential_weights_pos_k = np.sum(self.evidential_weights_pos_jk, axis = 1)
        self.evidential_weights_neg_k = np.sum(self.evidential_weights_neg_jk, axis = 1)

    def get_output_mass(self, num_classes = 10):

        # obtain all possible hypothesis sets (power set)
        omega = range(num_classes)

        self.powerset = list(powerset(omega))

        # initialize the output mass dictionary keys to the powerset
        # skip the singletons and the empty set (which is always zero by default)
        self.output_mass = {}
        self.output_mass = self.output_mass.fromkeys(self.powerset[(len(omega) + 1):], 0)

        # for singleton sets
        # implement equation 31 in Deneoux paper: store in numpy array [N, K, K]
        tiled_evidential_weights_neg_k = np.transpose(np.tile(self.evidential_weights_neg_k, (self.K, 1, 1)), (1, 0, 2))
        prod_term = 1. - np.exp(-tiled_evidential_weights_neg_k)

        row_idx, col_idx = np.diag_indices(self.K)
        prod_term[:, row_idx, col_idx] = 1
        
        self.output_mass_singletons = np.exp(-1. * self.evidential_weights_neg_k) * (np.exp(self.evidential_weights_pos_k) - 1. + np.product(prod_term, axis = 2))

        for key in self.output_mass.keys():

            # compute the masses for subsets of cardinality 2 or greater
            # Note: the empty set receives a mass of zero

            # check if every element of omega is in the key set
            mask = np.isin(omega, list(key), assume_unique=True)

            # implement equation 31 in Deneoux paper
            prod_in = np.product(np.exp(-1. * np.sum(self.evidential_weights_neg_k[:, mask])))
            prod_not_in = np.product(1. - np.exp(-1. * self.evidential_weights_neg_k[:, np.logical_not(mask)]))
            self.output_mass[key] = prod_in * prod_not_in

        # normalize the mass values
        norm_constant = (sum(self.output_mass.values()) + np.sum(self.output_mass_singletons))

        for key in self.output_mass.keys():
            self.output_mass[key] /= norm_constant

        self.output_mass_singletons /= norm_constant