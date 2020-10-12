'''This file computes the training evolution performance.'''

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

    norm_singletons_b_dist_even_total = []
    probs_b_dist_even_total = []
    probs_sparsemax_b_dist_even_total = []
    norm_singletons_b_dist_odd_total = []
    probs_b_dist_odd_total = []
    probs_sparsemax_b_dist_odd_total = []

    norm_singletons_w_dist_even_total = []
    probs_w_dist_even_total = []
    probs_sparsemax_w_dist_even_total = []
    norm_singletons_w_dist_odd_total = []
    probs_w_dist_odd_total = []
    probs_sparsemax_w_dist_odd_total = []

    seeds = [123] # 5, 135, 13579, 135791

    for random in seeds:
        
        tracker_global_train = torch.load('tracker_mnist_SGD_random_' + str(random) + '.pt')

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
        
        norm_singletons_b_dist_even = []
        probs_b_dist_even = []
        probs_sparsemax_b_dist_even = []
        norm_singletons_b_dist_odd = []
        probs_b_dist_odd = []
        probs_sparsemax_b_dist_odd = []


        norm_singletons_w_dist_even = []
        probs_w_dist_even = []
        probs_sparsemax_w_dist_even = []
        norm_singletons_w_dist_odd = []
        probs_w_dist_odd = []
        probs_sparsemax_w_dist_odd = []

        M = tracker_global_train['weight'].shape[2]

        probs_even = tracker_global_train['alpha_p'].data.cpu().numpy()[range(0,1),:,-1]
        probs_odd = tracker_global_train['alpha_p'].data.cpu().numpy()[range(1,2),:,-1]

        target_even = (probs_even >= probs_odd)*np.maximum(probs_even, probs_odd)
        target_even = target_even/1./np.sum(target_even)
        
        target_odd = (probs_odd >= probs_even)*np.maximum(probs_even, probs_odd)
        target_odd = target_odd/np.sum(target_odd)

        for iter in tqdm(range(M)):

            features_even = tracker_global_train['features'].data.cpu().numpy()[range(0,1),:,iter]
            features_odd = tracker_global_train['features'].data.cpu().numpy()[range(1,2),:,iter]

            for num in range(2):
                indices = range(num,num+1)

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
                # [N, K]
                z_one_hot = tracker_global_train['z'].data.cpu().numpy()[indices,:,iter]
                z = np.argmax(z_one_hot, axis=1)

                mean = deepcopy(features.flatten())

                dst_obj = DST()
                dst_obj.weights_from_linear_layer(weights, bias, features, mean)
                dst_obj.get_output_mass(num_classes = K)

                m_Z.append(dst_obj.output_mass[tuple(range(K))])

                norm_singletons = deepcopy(alpha_p)
                norm_singletons[dst_obj.output_mass_singletons == 0.] = 0.
                norm_singletons = norm_singletons/np.sum(norm_singletons)

                if num == 1:
                    bdist_mass = bhattacharyya_distance(norm_singletons, target_odd)
                    bdist_prob = bhattacharyya_distance(alpha_p, target_odd)
                    bdist_prob_sparsemax = bhattacharyya_distance(alpha_p_sparsemax, target_odd)
                    norm_singletons_b_dist_odd.append(bdist_mass)
                    probs_b_dist_odd.append(bdist_prob)
                    probs_sparsemax_b_dist_odd.append(bdist_prob_sparsemax)

                    wdist_mass = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=norm_singletons.flatten(), v_weights=target_odd.flatten())
                    wdist_prob = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=alpha_p.flatten(), v_weights=target_odd.flatten())
                    wdist_prob_sparsemax = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=alpha_p_sparsemax.flatten(), v_weights=target_odd.flatten())

                    norm_singletons_w_dist_odd.append(wdist_mass)
                    probs_w_dist_odd.append(wdist_prob)
                    probs_sparsemax_w_dist_odd.append(wdist_prob_sparsemax)

                if num == 0:
                    bdist_mass = bhattacharyya_distance(norm_singletons, target_even)
                    bdist_prob = bhattacharyya_distance(alpha_p, target_even)
                    bdist_prob_sparsemax = bhattacharyya_distance(alpha_p_sparsemax, target_even)
                    norm_singletons_b_dist_even.append(bdist_mass)
                    probs_b_dist_even.append(bdist_prob)
                    probs_sparsemax_b_dist_even.append(bdist_prob_sparsemax)

                    wdist_mass = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=norm_singletons.flatten(), v_weights=target_even.flatten())
                    wdist_prob = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=alpha_p.flatten(), v_weights=target_even.flatten())
                    wdist_prob_sparsemax = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=alpha_p_sparsemax.flatten(), v_weights=target_even.flatten())

                    norm_singletons_w_dist_even.append(wdist_mass)
                    probs_w_dist_even.append(wdist_prob)
                    probs_sparsemax_w_dist_even.append(wdist_prob_sparsemax)

                    print("target_even", target_even.shape)
                    print("probs even", alpha_p.shape)
                    print("probs sparsemax even", alpha_p_sparsemax.shape)
                    print("masses even", norm_singletons.shape)

        norm_singletons_b_dist_even_total.append(norm_singletons_b_dist_even)
        probs_b_dist_even_total.append(probs_b_dist_even)
        probs_sparsemax_b_dist_even_total.append(probs_sparsemax_b_dist_even)
        norm_singletons_b_dist_odd_total.append(norm_singletons_b_dist_odd)
        probs_b_dist_odd_total.append(probs_b_dist_odd)
        probs_sparsemax_b_dist_odd_total.append(probs_sparsemax_b_dist_odd)

        norm_singletons_w_dist_even_total.append(norm_singletons_w_dist_even)
        probs_w_dist_even_total.append(probs_w_dist_even)
        probs_sparsemax_w_dist_even_total.append(probs_sparsemax_w_dist_even)
        norm_singletons_w_dist_odd_total.append(norm_singletons_w_dist_odd)
        probs_w_dist_odd_total.append(probs_w_dist_odd)
        probs_sparsemax_w_dist_odd_total.append(probs_sparsemax_w_dist_odd)

    print("labels", c)
    print("singletons", dst_obj.output_mass_singletons)
    print("probabilities", alpha_p)
    print("probabilities", alpha_p_sparsemax)

    print("normalized singletons", norm_singletons)
    print("TARGET", target_even, target_odd)
    print("BDist", bdist_mass, bdist_prob, bdist_prob_sparsemax)

    # total

    norm_singletons_b_dist_even_mean = np.mean(np.array(norm_singletons_b_dist_even_total), axis=0)
    probs_b_dist_even_mean = np.mean(np.array(probs_b_dist_even_total), axis=0)
    probs_sparsemax_b_dist_even_mean = np.mean(np.array(probs_sparsemax_b_dist_even_total), axis=0)
    norm_singletons_b_dist_odd_mean = np.mean(np.array(norm_singletons_b_dist_odd_total), axis=0)
    probs_b_dist_odd_mean = np.mean(np.array(probs_b_dist_odd_total), axis=0)
    probs_sparsemax_b_dist_odd_mean = np.mean(np.array(probs_sparsemax_b_dist_odd_total), axis=0)

    norm_singletons_w_dist_even_mean = np.mean(np.array(norm_singletons_w_dist_even_total), axis=0)
    probs_w_dist_even_mean = np.mean(np.array(probs_w_dist_even_total), axis=0)
    probs_sparsemax_w_dist_even_mean = np.mean(np.array(probs_sparsemax_w_dist_even_total), axis=0)
    norm_singletons_w_dist_odd_mean = np.mean(np.array(norm_singletons_w_dist_odd_total), axis=0)
    probs_w_dist_odd_mean = np.mean(np.array(probs_w_dist_odd_total), axis=0)
    probs_sparsemax_w_dist_odd_mean = np.mean(np.array(probs_sparsemax_w_dist_odd_total), axis=0)

    N = np.array(probs_w_dist_odd_total).shape[0]
    print("Total number of samples: ", N)

    norm_singletons_b_dist_even_std = np.std(np.array(norm_singletons_b_dist_even_total), axis=0)/np.sqrt(N)
    probs_b_dist_even_std = np.std(np.array(probs_b_dist_even_total), axis=0)/np.sqrt(N)
    probs_sparsemax_b_dist_even_std = np.std(np.array(probs_sparsemax_b_dist_even_total), axis=0)/np.sqrt(N)
    norm_singletons_b_dist_odd_std = np.std(np.array(norm_singletons_b_dist_odd_total), axis=0)/np.sqrt(N)
    probs_b_dist_odd_std = np.std(np.array(probs_b_dist_odd_total), axis=0)/np.sqrt(N)
    probs_sparsemax_b_dist_odd_std = np.std(np.array(probs_sparsemax_b_dist_odd_total), axis=0)/np.sqrt(N)

    norm_singletons_w_dist_even_std = np.std(np.array(norm_singletons_w_dist_even_total), axis=0)/np.sqrt(N)
    probs_w_dist_even_std = np.std(np.array(probs_w_dist_even_total), axis=0)/np.sqrt(N)
    probs_sparsemax_w_dist_even_std = np.std(np.array(probs_sparsemax_w_dist_even_total), axis=0)/np.sqrt(N)
    norm_singletons_w_dist_odd_std = np.std(np.array(norm_singletons_w_dist_odd_total), axis=0)/np.sqrt(N)
    probs_w_dist_odd_std = np.std(np.array(probs_w_dist_odd_total), axis=0)/np.sqrt(N)
    probs_sparsemax_w_dist_odd_std = np.std(np.array(probs_sparsemax_w_dist_odd_total), axis=0)/np.sqrt(N)

    plt.figure()
    x = 100*np.arange(M)
    plt.plot(x, probs_b_dist_even_mean, label="Softmax")
    plt.fill_between(x, probs_b_dist_even_mean-probs_b_dist_even_std, probs_b_dist_even_mean+probs_b_dist_even_std, alpha=0.5)
    plt.plot(x, probs_sparsemax_b_dist_even_mean, label="Sparsemax")
    plt.fill_between(x, probs_sparsemax_b_dist_even_mean-probs_sparsemax_b_dist_even_std, probs_sparsemax_b_dist_even_mean+probs_sparsemax_b_dist_even_std, alpha=0.5)
    plt.plot(x, norm_singletons_b_dist_even_mean, label="Ours")
    plt.fill_between(x, norm_singletons_b_dist_even_mean-norm_singletons_b_dist_even_std, norm_singletons_b_dist_even_mean+norm_singletons_b_dist_even_std, alpha=0.5)
    plt.legend()
    matplotlib2tikz.save("even_b_dist.tex")
    plt.savefig("even_b_dist.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(x, probs_b_dist_odd_mean, label="Softmax")
    plt.fill_between(x, probs_b_dist_odd_mean-probs_b_dist_odd_std, probs_b_dist_odd_mean+probs_b_dist_odd_std, alpha=0.5)
    plt.plot(x, probs_sparsemax_b_dist_odd_mean, label="Sparsemax")
    plt.fill_between(x, probs_sparsemax_b_dist_odd_mean-probs_sparsemax_b_dist_odd_std, probs_sparsemax_b_dist_odd_mean+probs_sparsemax_b_dist_odd_std, alpha=0.5)
    plt.plot(x, norm_singletons_b_dist_odd_mean, label="Ours")
    plt.fill_between(x, norm_singletons_b_dist_odd_mean-norm_singletons_b_dist_odd_std, norm_singletons_b_dist_odd_mean+norm_singletons_b_dist_odd_std, alpha=0.5)
    plt.legend()
    matplotlib2tikz.save("odd_b_dist.tex")
    plt.savefig("odd_b_dist.png")
    plt.show()
    plt.close()

    plt.figure()
    x = 100*np.arange(M)
    plt.plot(x, probs_w_dist_even_mean, label="Softmax")
    plt.fill_between(x, probs_w_dist_even_mean-probs_w_dist_even_std, probs_w_dist_even_mean+probs_w_dist_even_std, alpha=0.5)
    plt.plot(x, probs_sparsemax_w_dist_even_mean, label="Sparsemax")
    plt.fill_between(x, probs_sparsemax_w_dist_even_mean-probs_sparsemax_w_dist_even_std, probs_sparsemax_w_dist_even_mean+probs_sparsemax_w_dist_even_std, alpha=0.5)
    plt.plot(x, norm_singletons_w_dist_even_mean, label="Ours")
    plt.fill_between(x, norm_singletons_w_dist_even_mean-norm_singletons_w_dist_even_std, norm_singletons_w_dist_even_mean+norm_singletons_w_dist_even_std, alpha=0.5)
    plt.legend()
    matplotlib2tikz.save("even_w_dist.tex")
    plt.savefig("even_w_dist.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(x, probs_w_dist_odd_mean, label="Softmax")
    plt.fill_between(x, probs_w_dist_odd_mean-probs_w_dist_odd_std, probs_w_dist_odd_mean+probs_w_dist_odd_std, alpha=0.5)
    plt.plot(x, probs_sparsemax_w_dist_odd_mean, label="Sparsemax")
    plt.fill_between(x, probs_sparsemax_w_dist_odd_mean-probs_sparsemax_w_dist_odd_std, probs_sparsemax_w_dist_odd_mean+probs_sparsemax_w_dist_odd_std, alpha=0.5)
    plt.plot(x, norm_singletons_w_dist_odd_mean, label="Ours")
    plt.fill_between(x, norm_singletons_w_dist_odd_mean-norm_singletons_w_dist_odd_std, norm_singletons_w_dist_odd_mean+norm_singletons_w_dist_odd_std, alpha=0.5)
    plt.legend()
    matplotlib2tikz.save("odd_w_dist.tex")
    plt.savefig("odd_w_dist.png")
    plt.show()
    plt.close()

if __name__== "__main__":
    main()
