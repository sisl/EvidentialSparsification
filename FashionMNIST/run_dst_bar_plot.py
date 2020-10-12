'''This file computes the qualitative observations for the bar plots.'''

import numpy as np
import scipy
import scipy.stats
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy

from tqdm import tqdm
import matplotlib2tikz

import torch

from dst_utils import *
from dst import *

# From: https://github.com/sebastianruder/learn-to-select-data
def bhattacharyya_distance(p, q):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""
    sim = -1.0*np.log(np.sum([np.sqrt(p*q)]))
    assert not np.isnan(sim), 'Error: Similarity is nan.'
    if np.isinf(sim):
        # the similarity is -inf if no term in the review is in the vocabulary
        return 0
    return sim

# Load the data and perform the DST transformation
def main():

    random = 2
        
    tracker_global_train = torch.load('tracker_fashion_Adam_random_' + str(random) + '.pt')

    # [K, J, M]
    print(tracker_global_train['weight'].data.shape)

    # [K, M]
    print(tracker_global_train['bias'].data.shape)

    # [N, K, M]
    print(tracker_global_train['alpha_p'].data.shape)

    # [N, K, M]
    print(tracker_global_train['alpha_p_sparsemax'].data.shape)
    
    # [N, J, M]
    print(tracker_global_train['features'].data.shape) 

    # [K, K, M]
    print(tracker_global_train['z'].data.shape)

    K = 10

    m_Z = []

    M = tracker_global_train['weight'].shape[2]

    probs_tops = tracker_global_train['alpha_p'].data.cpu().numpy()[range(0,1),:,-1]
    probs_bottoms = tracker_global_train['alpha_p'].data.cpu().numpy()[range(1,2),:,-1]

    num_tops = tracker_global_train['num_tops']
    num_bottoms = tracker_global_train['num_bottoms']

    features_tops = tracker_global_train['features'].data.cpu().numpy()[range(0,1),:,-1]
    features_bottoms = tracker_global_train['features'].data.cpu().numpy()[range(1,2),:,-1]

    for num in range(2):
        indices = range(num,num+1)

        iter = -1

        # [J, K]
        weights = tracker_global_train['weight'].data.cpu().numpy()[:,:,iter].T
        # [K, 1]
        bias = np.expand_dims(tracker_global_train['bias'].data.cpu().numpy()[:,iter], axis = -1)
        # [N, J]
        features = tracker_global_train['features'].data.cpu().numpy()[indices,:,iter]
        # [N, J]
        x = np.reshape(tracker_global_train['x'].data.cpu().numpy()[indices,:,iter], (28,28))
        # [N, 1]
        c = tracker_global_train['c'].data.cpu().numpy()[indices,:,iter]
        # [N, K]
        alpha_p = tracker_global_train['alpha_p'].data.cpu().numpy()[indices,:,iter]
        # [N, K]
        alpha_p_sparsemax = tracker_global_train['alpha_p_sparsemax'].data.cpu().numpy()[indices,:,iter]
        # [K, K]
        z_one_hot = tracker_global_train['z'].data.cpu().numpy()[indices,:,iter]
        z = np.argmax(z_one_hot, axis=1)

        mean = features.flatten()

        dst_obj = DST()
        dst_obj.weights_from_linear_layer(weights, bias, features, mean)
        dst_obj.get_output_mass(num_classes = K)

        m_Z.append(dst_obj.output_mass[tuple(range(K))])

        norm_singletons = deepcopy(alpha_p)
        norm_singletons[dst_obj.output_mass_singletons == 0.] = 0.
        norm_singletons = norm_singletons/np.sum(norm_singletons)

        print('sum of singletons', sum(dst_obj.output_mass_singletons.flatten()))
        if num == 1:

            plt.figure()
            width = 0.5
            p1 = plt.bar(np.arange(K), alpha_p.flatten(), width, color='blue', alpha = 0.5)
            p2 = plt.bar(np.arange(K), alpha_p_sparsemax.flatten(), width, color='orange', alpha = 0.5)
            p3 = plt.bar(np.arange(K), norm_singletons.flatten(), width, color='green', alpha = 0.5)

            plt.xlabel('Z')
            plt.ylabel('Values')
            plt.ylim(0,1.0)
            plt.title('Values for bottoms')
            plt.legend(['Softmax', 'Sparsemax', 'Ours'])
            plt.savefig('bottoms_random_' + str(random) + '.png', dpi=600)  
            matplotlib2tikz.save('bottoms_random_' + str(random) + '.tex')  
            plt.close()

        if num == 0:

            plt.figure()
            width = 0.5
            p1 = plt.bar(np.arange(K), alpha_p.flatten(), width, color='blue', alpha = 0.5)
            p2 = plt.bar(np.arange(K), alpha_p_sparsemax.flatten(), width, color='orange', alpha = 0.5)
            p3 = plt.bar(np.arange(K), norm_singletons.flatten(), width, color='green', alpha = 0.5)

            plt.xlabel('Z')
            plt.ylabel('Values')
            plt.ylim(0,1.0)
            plt.title('Values for tops')
            plt.legend(['Softmax', 'Sparsemax', 'Ours'])
            plt.savefig('tops_random_' + str(random) + '.png', dpi=600)  
            matplotlib2tikz.save('tops_random_' + str(random) + '.tex')
            plt.close()

        print("labels", c)
        print("singletons", dst_obj.output_mass_singletons)
        print("probabilities", alpha_p)

if __name__== "__main__":
    main()
