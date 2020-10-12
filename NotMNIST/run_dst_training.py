'''This file computes the training evolution performance.'''

import numpy as np
import scipy
import scipy.stats
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy
import pdb

from tqdm import tqdm
import matplotlib2tikz

import torch

from dst_utils import *
from dst import *
import os

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

    norm_singletons_b_dist_middle_bar_total = []
    probs_b_dist_middle_bar_total = []
    probs_sparsemax_b_dist_middle_bar_total = []
    norm_singletons_b_dist_no_middle_bar_total = []
    probs_b_dist_no_middle_bar_total = []
    probs_sparsemax_b_dist_no_middle_bar_total = []

    norm_singletons_w_dist_middle_bar_total = []
    probs_w_dist_middle_bar_total = []
    probs_sparsemax_w_dist_middle_bar_total = []
    norm_singletons_w_dist_no_middle_bar_total = []
    probs_w_dist_no_middle_bar_total = []
    probs_sparsemax_w_dist_no_middle_bar_total = []

    seeds = [123] # 5, 135, 13579, 135791

    for index in range(len(seeds)):

        random = seeds[index]
        time_number = time_numbers[index]

        num_iters = 0

        K = 10

        m_Z = []
        
        norm_singletons_b_dist_middle_bar = []
        probs_b_dist_middle_bar = []
        probs_sparsemax_b_dist_middle_bar = []
        norm_singletons_b_dist_no_middle_bar = []
        probs_b_dist_no_middle_bar = []
        probs_sparsemax_b_dist_no_middle_bar = []

        norm_singletons_w_dist_middle_bar = []
        probs_w_dist_middle_bar = []
        probs_sparsemax_w_dist_middle_bar = []
        norm_singletons_w_dist_no_middle_bar = []
        probs_w_dist_no_middle_bar = []
        probs_sparsemax_w_dist_no_middle_bar = []

        tracker_global_train = torch.load('tracker_mnist_SGD_random_' + str(random) + '.pt')
        
        # [K, J, M]
        print(tracker_global_train['weight'].data.shape)

        # [K, M]
        print(tracker_global_train['bias'].data.shape)

        # [N, K, M]
        print(tracker_global_train['alpha_p'].data.shape)
        
        # [N, J, M]
        print(tracker_global_train['features'].data.shape) 

        # [N, J, M]
        print(tracker_global_train['z'].data.shape) 

        M = tracker_global_train['weight'].shape[2]

        # compute how many epochs recorded
        num_iters += M
        print("num_iters", num_iters)

        for iter in tqdm(range(M)):

            probs_middle_bar = tracker_global_train['alpha_p'].data.cpu().numpy()[range(0,1),:,iter]
            probs_no_middle_bar = tracker_global_train['alpha_p'].data.cpu().numpy()[range(1,2),:,iter]

            # target_middle_bar = (probs_middle_bar >= probs_no_middle_bar)/1./np.sum(probs_middle_bar >= probs_no_middle_bar)
            # target_no_middle_bar = (probs_no_middle_bar >= probs_middle_bar)/1./np.sum(probs_no_middle_bar >= probs_middle_bar)
            
            target_middle_bar = (probs_middle_bar >= probs_no_middle_bar)*np.maximum(probs_middle_bar, probs_no_middle_bar)
            target_middle_bar = target_middle_bar/1./np.sum(target_middle_bar)
            
            target_no_middle_bar = (probs_no_middle_bar >= probs_middle_bar)*np.maximum(probs_middle_bar, probs_no_middle_bar)
            target_no_middle_bar = target_no_middle_bar/np.sum(target_no_middle_bar)

            features_middle_bar = tracker_global_train['features'].data.cpu().numpy()[range(0,1),:,iter]
            features_no_middle_bar = tracker_global_train['features'].data.cpu().numpy()[range(1,2),:,iter]

            for num in range(2):
                indices = range(num,num+1)
                # pdb.set_trace()

                # [J, K]
                weights = tracker_global_train['weight'].data.cpu().numpy()[:,:,iter].T
                # [K, 1]
                bias = np.expand_dims(tracker_global_train['bias'].data.cpu().numpy()[:,iter], axis = -1)
                # [N, J]
                features = tracker_global_train['features'].data.cpu().numpy()[indices,:,iter]
                # [N, J]
                x = np.reshape(tracker_global_train['x'].data.cpu().numpy()[:,:,:,:,iter], (K,28,28))
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
                    bdist_mass = bhattacharyya_distance(norm_singletons, target_no_middle_bar)
                    bdist_prob = bhattacharyya_distance(alpha_p, target_no_middle_bar)
                    bdist_prob_sparsemax = bhattacharyya_distance(alpha_p_sparsemax, target_no_middle_bar)
                    norm_singletons_b_dist_no_middle_bar.append(bdist_mass)
                    probs_b_dist_no_middle_bar.append(bdist_prob)
                    probs_sparsemax_b_dist_no_middle_bar.append(bdist_prob_sparsemax)

                    wdist_mass = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=norm_singletons.flatten(), v_weights=target_no_middle_bar.flatten())
                    wdist_prob = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=alpha_p.flatten(), v_weights=target_no_middle_bar.flatten())
                    wdist_prob_sparsemax = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=alpha_p_sparsemax.flatten(), v_weights=target_no_middle_bar.flatten())

                    norm_singletons_w_dist_no_middle_bar.append(wdist_mass)
                    probs_w_dist_no_middle_bar.append(wdist_prob)
                    probs_sparsemax_w_dist_no_middle_bar.append(wdist_prob_sparsemax)

                if num == 0:
                    bdist_mass = bhattacharyya_distance(norm_singletons, target_middle_bar)
                    bdist_prob = bhattacharyya_distance(alpha_p, target_middle_bar)
                    bdist_prob_sparsemax = bhattacharyya_distance(alpha_p_sparsemax, target_middle_bar)
                    norm_singletons_b_dist_middle_bar.append(bdist_mass)
                    probs_b_dist_middle_bar.append(bdist_prob)
                    probs_sparsemax_b_dist_middle_bar.append(bdist_prob_sparsemax)

                    wdist_mass = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=norm_singletons.flatten(), v_weights=target_middle_bar.flatten())
                    wdist_prob = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=alpha_p.flatten(), v_weights=target_middle_bar.flatten())
                    wdist_prob_sparsemax = scipy.stats.wasserstein_distance(u_values=np.arange(10)/9., v_values=np.arange(10)/9., u_weights=alpha_p_sparsemax.flatten(), v_weights=target_middle_bar.flatten())

                    norm_singletons_w_dist_middle_bar.append(wdist_mass)
                    probs_w_dist_middle_bar.append(wdist_prob)
                    probs_sparsemax_w_dist_middle_bar.append(wdist_prob_sparsemax)

        # add new random seed data
        norm_singletons_b_dist_middle_bar_total.append(norm_singletons_b_dist_middle_bar)
        probs_b_dist_middle_bar_total.append(probs_b_dist_middle_bar)
        probs_sparsemax_b_dist_middle_bar_total.append(probs_sparsemax_b_dist_middle_bar)
        norm_singletons_b_dist_no_middle_bar_total.append(norm_singletons_b_dist_no_middle_bar)
        probs_b_dist_no_middle_bar_total.append(probs_b_dist_no_middle_bar)
        probs_sparsemax_b_dist_no_middle_bar_total.append(probs_sparsemax_b_dist_no_middle_bar)

        norm_singletons_w_dist_middle_bar_total.append(norm_singletons_w_dist_middle_bar)
        probs_w_dist_middle_bar_total.append(probs_w_dist_middle_bar)
        probs_sparsemax_w_dist_middle_bar_total.append(probs_sparsemax_w_dist_middle_bar)
        norm_singletons_w_dist_no_middle_bar_total.append(norm_singletons_w_dist_no_middle_bar)
        probs_w_dist_no_middle_bar_total.append(probs_w_dist_no_middle_bar)
        probs_sparsemax_w_dist_no_middle_bar_total.append(probs_sparsemax_w_dist_no_middle_bar)

    print("labels", c)
    print("singletons", dst_obj.output_mass_singletons)
    print("probabilities", alpha_p)
    print("probabilities max", alpha_p_sparsemax)

    print("normalized singletons", norm_singletons)
    print("TARGET", target_middle_bar, target_no_middle_bar)
    print("BDist", bdist_mass, bdist_prob, bdist_prob_sparsemax)

    print("size of final data", len(norm_singletons_b_dist_middle_bar_total))

    # total

    norm_singletons_b_dist_middle_bar_mean = np.mean(np.array(norm_singletons_b_dist_middle_bar_total), axis=0)
    probs_b_dist_middle_bar_mean = np.mean(np.array(probs_b_dist_middle_bar_total), axis=0)
    probs_sparsemax_b_dist_middle_bar_mean = np.mean(np.array(probs_sparsemax_b_dist_middle_bar_total), axis=0)
    norm_singletons_b_dist_no_middle_bar_mean = np.mean(np.array(norm_singletons_b_dist_no_middle_bar_total), axis=0)
    probs_b_dist_no_middle_bar_mean = np.mean(np.array(probs_b_dist_no_middle_bar_total), axis=0)
    probs_sparsemax_b_dist_no_middle_bar_mean = np.mean(np.array(probs_sparsemax_b_dist_no_middle_bar_total), axis=0)

    norm_singletons_w_dist_middle_bar_mean = np.mean(np.array(norm_singletons_w_dist_middle_bar_total), axis=0)
    probs_w_dist_middle_bar_mean = np.mean(np.array(probs_w_dist_middle_bar_total), axis=0)
    probs_sparsemax_w_dist_middle_bar_mean = np.mean(np.array(probs_sparsemax_w_dist_middle_bar_total), axis=0)
    norm_singletons_w_dist_no_middle_bar_mean = np.mean(np.array(norm_singletons_w_dist_no_middle_bar_total), axis=0)
    probs_w_dist_no_middle_bar_mean = np.mean(np.array(probs_w_dist_no_middle_bar_total), axis=0)
    probs_sparsemax_w_dist_no_middle_bar_mean = np.mean(np.array(probs_sparsemax_w_dist_no_middle_bar_total), axis=0)

    N = np.array(probs_w_dist_no_middle_bar_total).shape[0]
    print("Total number of samples: ", N)

    norm_singletons_b_dist_middle_bar_std = np.std(np.array(norm_singletons_b_dist_middle_bar_total), axis=0)/np.sqrt(N)
    probs_b_dist_middle_bar_std = np.std(np.array(probs_b_dist_middle_bar_total), axis=0)/np.sqrt(N)
    probs_sparsemax_b_dist_middle_bar_std = np.std(np.array(probs_sparsemax_b_dist_middle_bar_total), axis=0)/np.sqrt(N)
    norm_singletons_b_dist_no_middle_bar_std = np.std(np.array(norm_singletons_b_dist_no_middle_bar_total), axis=0)/np.sqrt(N)
    probs_b_dist_no_middle_bar_std = np.std(np.array(probs_b_dist_no_middle_bar_total), axis=0)/np.sqrt(N)
    probs_sparsemax_b_dist_no_middle_bar_std = np.std(np.array(probs_sparsemax_b_dist_no_middle_bar_total), axis=0)/np.sqrt(N)

    norm_singletons_w_dist_middle_bar_std = np.std(np.array(norm_singletons_w_dist_middle_bar_total), axis=0)/np.sqrt(N)
    probs_w_dist_middle_bar_std = np.std(np.array(probs_w_dist_middle_bar_total), axis=0)/np.sqrt(N)
    probs_sparsemax_w_dist_middle_bar_std = np.std(np.array(probs_sparsemax_w_dist_middle_bar_total), axis=0)/np.sqrt(N)
    norm_singletons_w_dist_no_middle_bar_std = np.std(np.array(norm_singletons_w_dist_no_middle_bar_total), axis=0)/np.sqrt(N)
    probs_w_dist_no_middle_bar_std = np.std(np.array(probs_w_dist_no_middle_bar_total), axis=0)/np.sqrt(N)
    probs_sparsemax_w_dist_no_middle_bar_std = np.std(np.array(probs_sparsemax_w_dist_no_middle_bar_total), axis=0)/np.sqrt(N)

    plt.figure()
    x = 200*np.arange(probs_b_dist_middle_bar_mean.shape[0])

    # subsample
    subsample = 10
    x = x[::subsample]
    
    norm_singletons_b_dist_middle_bar_mean = norm_singletons_b_dist_middle_bar_mean[::subsample]
    probs_b_dist_middle_bar_mean = probs_b_dist_middle_bar_mean[::subsample]
    probs_sparsemax_b_dist_middle_bar_mean = probs_sparsemax_b_dist_middle_bar_mean[::subsample]
    norm_singletons_b_dist_no_middle_bar_mean = norm_singletons_b_dist_no_middle_bar_mean[::subsample]
    probs_b_dist_no_middle_bar_mean = probs_b_dist_no_middle_bar_mean[::subsample]
    probs_sparsemax_b_dist_no_middle_bar_mean = probs_sparsemax_b_dist_no_middle_bar_mean[::subsample]

    norm_singletons_w_dist_middle_bar_mean = norm_singletons_w_dist_middle_bar_mean[::subsample]
    probs_w_dist_middle_bar_mean = probs_w_dist_middle_bar_mean[::subsample]
    probs_sparsemax_w_dist_middle_bar_mean = probs_sparsemax_w_dist_middle_bar_mean[::subsample]
    norm_singletons_w_dist_no_middle_bar_mean = norm_singletons_w_dist_no_middle_bar_mean[::subsample]
    probs_w_dist_no_middle_bar_mean = probs_w_dist_no_middle_bar_mean[::subsample]
    probs_sparsemax_w_dist_no_middle_bar_mean = probs_sparsemax_w_dist_no_middle_bar_mean[::subsample]

    norm_singletons_b_dist_middle_bar_std = norm_singletons_b_dist_middle_bar_std[::subsample]
    probs_b_dist_middle_bar_std = probs_b_dist_middle_bar_std[::subsample]
    probs_sparsemax_b_dist_middle_bar_std = probs_sparsemax_b_dist_middle_bar_std[::subsample]
    norm_singletons_b_dist_no_middle_bar_std = norm_singletons_b_dist_no_middle_bar_std[::subsample]
    probs_b_dist_no_middle_bar_std = probs_b_dist_no_middle_bar_std[::subsample]
    probs_sparsemax_b_dist_no_middle_bar_std = probs_sparsemax_b_dist_no_middle_bar_std[::subsample]

    norm_singletons_w_dist_middle_bar_std = norm_singletons_w_dist_middle_bar_std[::subsample]
    probs_w_dist_middle_bar_std = probs_w_dist_middle_bar_std[::subsample]
    probs_sparsemax_w_dist_middle_bar_std = probs_sparsemax_w_dist_middle_bar_std[::subsample]
    norm_singletons_w_dist_no_middle_bar_std = norm_singletons_w_dist_no_middle_bar_std[::subsample]
    probs_w_dist_no_middle_bar_std = probs_w_dist_no_middle_bar_std[::subsample]
    probs_sparsemax_w_dist_no_middle_bar_std = probs_sparsemax_w_dist_no_middle_bar_std[::subsample]

    plt.plot(x, probs_b_dist_middle_bar_mean, label="Softmax", color="blue")
    plt.fill_between(x, probs_b_dist_middle_bar_mean-probs_b_dist_middle_bar_std, probs_b_dist_middle_bar_mean+probs_b_dist_middle_bar_std, alpha=0.5, color="blue")
    plt.plot(x, probs_sparsemax_b_dist_middle_bar_mean, label="Sparsemax", color="orange")
    plt.fill_between(x, probs_sparsemax_b_dist_middle_bar_mean-probs_sparsemax_b_dist_middle_bar_std, probs_sparsemax_b_dist_middle_bar_mean+probs_sparsemax_b_dist_middle_bar_std, alpha=0.5, color="orange")
    plt.plot(x, norm_singletons_b_dist_middle_bar_mean, label="Ours", color="green")
    plt.fill_between(x, norm_singletons_b_dist_middle_bar_mean-norm_singletons_b_dist_middle_bar_std, norm_singletons_b_dist_middle_bar_mean+norm_singletons_b_dist_middle_bar_std, alpha=0.5, color="green")
    plt.legend()
    matplotlib2tikz.save("middle_bar_b_dist_sparsemax_subsample_" + str(subsample) + ".tex")
    plt.savefig("middle_bar_b_dist_sparsemax_subsample_" + str(subsample) + ".png")
    # plt.show()
    plt.close()

    plt.figure()
    plt.plot(x, probs_b_dist_no_middle_bar_mean, label="Softmax", color="blue")
    plt.fill_between(x, probs_b_dist_no_middle_bar_mean-probs_b_dist_no_middle_bar_std, probs_b_dist_no_middle_bar_mean+probs_b_dist_no_middle_bar_std, alpha=0.5, color="blue")
    plt.plot(x, probs_sparsemax_b_dist_no_middle_bar_mean, label="Sparsemax", color="orange")
    plt.fill_between(x, probs_sparsemax_b_dist_no_middle_bar_mean-probs_sparsemax_b_dist_no_middle_bar_std, probs_sparsemax_b_dist_no_middle_bar_mean+probs_sparsemax_b_dist_no_middle_bar_std, alpha=0.5, color="orange")
    plt.plot(x, norm_singletons_b_dist_no_middle_bar_mean, label="Ours", color="green")
    plt.fill_between(x, norm_singletons_b_dist_no_middle_bar_mean-norm_singletons_b_dist_no_middle_bar_std, norm_singletons_b_dist_no_middle_bar_mean+norm_singletons_b_dist_no_middle_bar_std, alpha=0.5, color="green")
    plt.legend()
    matplotlib2tikz.save("no_middle_bar_b_dist_sparsemax_subsample_" + str(subsample) + ".tex")
    plt.savefig("no_middle_bar_b_dist_sparsemax_subsample_" + str(subsample) + ".png")
    # plt.show()
    plt.close()

    plt.figure()
    plt.plot(x, probs_w_dist_middle_bar_mean, label="Softmax", color="blue")
    plt.fill_between(x, probs_w_dist_middle_bar_mean-probs_w_dist_middle_bar_std, probs_w_dist_middle_bar_mean+probs_w_dist_middle_bar_std, alpha=0.5, color="blue")
    plt.plot(x, probs_sparsemax_w_dist_middle_bar_mean, label="Sparsemax", color="orange")
    plt.fill_between(x, probs_sparsemax_w_dist_middle_bar_mean-probs_sparsemax_w_dist_middle_bar_std, probs_sparsemax_w_dist_middle_bar_mean+probs_sparsemax_w_dist_middle_bar_std, alpha=0.5, color="orange")
    plt.plot(x, norm_singletons_w_dist_middle_bar_mean, label="Ours", color="green")
    plt.fill_between(x, norm_singletons_w_dist_middle_bar_mean-norm_singletons_w_dist_middle_bar_std, norm_singletons_w_dist_middle_bar_mean+norm_singletons_w_dist_middle_bar_std, alpha=0.5, color="green")
    plt.legend()
    matplotlib2tikz.save("middle_bar_w_dist_sparsemax_subsample_" + str(subsample) + ".tex")
    plt.savefig("middle_bar_w_dist_sparsemax_subsample_" + str(subsample) + ".png")
    # plt.show()
    plt.close()

    plt.figure()
    plt.plot(x, probs_w_dist_no_middle_bar_mean, label="Softmax", color="blue")
    plt.fill_between(x, probs_w_dist_no_middle_bar_mean-probs_w_dist_no_middle_bar_std, probs_w_dist_no_middle_bar_mean+probs_w_dist_no_middle_bar_std, alpha=0.5, color="blue")
    plt.plot(x, probs_sparsemax_w_dist_no_middle_bar_mean, label="Sparsemax", color="orange")
    plt.fill_between(x, probs_sparsemax_w_dist_no_middle_bar_mean-probs_sparsemax_w_dist_no_middle_bar_std, probs_sparsemax_w_dist_no_middle_bar_mean+probs_sparsemax_w_dist_no_middle_bar_std, alpha=0.5, color="orange")
    plt.plot(x, norm_singletons_w_dist_no_middle_bar_mean, label="Ours", color="green")
    plt.fill_between(x, norm_singletons_w_dist_no_middle_bar_mean-norm_singletons_w_dist_no_middle_bar_std, norm_singletons_w_dist_no_middle_bar_mean+norm_singletons_w_dist_no_middle_bar_std, alpha=0.5, color="green")
    plt.legend()
    matplotlib2tikz.save("no_middle_bar_w_dist_subsample_" + str(subsample) + ".tex")
    plt.savefig("no_middle_bar_w_dist_subsample_" + str(subsample) + ".png")
    # plt.show()
    plt.close()

if __name__== "__main__":
    main()
