import numpy as np
import scipy
import scipy.stats
import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from copy import deepcopy
import pickle as pkl
import os

from tqdm import tqdm
import matplotlib2tikz

import torch
from sparsemax import Sparsemax

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

    random = 123

    dict = pkl.load(open('../../dst_test.pkl', "rb"))
    print("weight", dict['weight'].shape)
    print("z", dict['z_one_hot'].shape)
    print("x", len(dict['x']), dict['x'][0].shape)
    print("alpha_p", dict['alpha_p'].data.shape)
    print("features", dict['features'].shape)
    print("bias", dict['bias'])
    print("c", len(dict['c']), dict['c'][0][0].shape, dict['c'][1][0].shape, len(dict['c'][1][1]))
    print("gt", len(dict['gt']), dict['gt'][0].shape, dict['gt'][1].shape)
    batch_size = dict['features'].shape[0]

    K = 5

    m_Z = []
    keep_nums = {}

    if not os.path.exists('results/'):
        os.makedirs('results/')

    # initialize filtered alpha p
    filtered_alpha_p = np.zeros((batch_size, 2, 5))
    filtered_alpha_p_sparsemax = np.zeros((batch_size, 2, 5))

    for batch in tqdm(range(batch_size)):

        # [N, J]
        features = np.expand_dims(dict['features'].data.cpu().numpy()[batch,:], axis=0)
        # [25,12,2]
        x = dict['x'][batch]
        # [9,6]
        c = dict['c'][batch][0]
        # [12, 2]
        gt = dict['gt'][batch]

        num_predictions = gt.shape[0]

        filtered_mask = np.zeros((x.shape[0]))
        filtered_mask_sparsemax = np.zeros((x.shape[0]))

        for num in range(2):

            indices = range(num,num+1)

            # [J, K]
            weights = np.reshape(dict['weight'], (32,2,5))[:,num,:]
            # [K, 1]
            bias = np.expand_dims(np.reshape(dict['bias'], (2,5))[num,:], axis = -1)
            # [N, K]
            alpha_p = dict['alpha_p'][batch,indices,:]
            # [25, N, K]
            z_one_hot = np.reshape(dict['z_one_hot'], (25,batch_size,2,5))[:,batch,indices,:]
            # [25, 1]
            z = np.argmax(z_one_hot, axis=-1)

            print("weights", weights.shape)
            print("bias", bias.shape)
            print("features", features.shape)
            print("x", x.shape)
            print("c", c.shape)
            print("alpha_p", alpha_p.shape)
            print("z_one_hot", z.shape)
            print("gt", gt.shape)

            # Compute the sparsemax baseline
            sparsemax = Sparsemax(dim=-1)
            sparsemax_logits = (weights.T.dot(features.T) + bias).flatten()
            filtered_alpha_p_sparsemax_torch = sparsemax(torch.from_numpy(sparsemax_logits).cuda())
            print('logits', sparsemax_logits)

            # Populate the filtered sparsemax distribution
            filtered_alpha_p_sparsemax[batch,num,:] = filtered_alpha_p_sparsemax_torch.data.cpu().numpy().flatten()
            print("filtered alpha p sparsemax", filtered_alpha_p_sparsemax[batch])

            indices_filtered_sparsemax = np.where(filtered_alpha_p_sparsemax[batch,num,:] == 0)
            print('filtered indices', indices_filtered_sparsemax[0].shape, filtered_mask_sparsemax.shape)
            for j in range(indices_filtered_sparsemax[0].shape[0]):
                filtered_mask_sparsemax[z.flatten() == indices_filtered_sparsemax[0][j]] = 1.

            # Compute DST filter
            mean = features.flatten()

            dst_obj = DST()
            dst_obj.weights_from_linear_layer(weights, bias, features, mean)
            dst_obj.get_output_mass(num_classes = K)

            m_Z.append(dst_obj.output_mass[tuple(range(K))])

            print('sum of singletons', sum(dst_obj.output_mass_singletons.flatten()))

            norm_singletons = deepcopy(alpha_p)
            norm_singletons[dst_obj.output_mass_singletons == 0.] = 0.
            norm_singletons = norm_singletons/np.sum(norm_singletons)

            indices_filtered = np.where(norm_singletons[0] == 0)
            print('filtered indices', indices_filtered[0].shape, z.shape, filtered_mask.shape)
            for j in range(indices_filtered[0].shape[0]):
                filtered_mask[z.flatten() == indices_filtered[0][j]] = 1.

            # save the filtered alpha_p
            filtered_alpha_p[batch,num,:] = norm_singletons.flatten()
            print("filtered alpha p", filtered_alpha_p[batch])

            if num == 1:

                plt.figure()
                width = 0.5
                p1 = plt.bar(np.arange(K), alpha_p.flatten(), width, color='blue', alpha = 0.5)
                p2 = plt.bar(np.arange(K), filtered_alpha_p_sparsemax[batch,num,:].flatten(), width, color='orange', alpha = 0.5)
                p3 = plt.bar(np.arange(K), norm_singletons.flatten(), width, color='green', alpha = 0.5) # /1./np.sum(dst_obj.output_mass_singletons.flatten())

                plt.xlabel('Z')
                plt.ylabel('Values')
                plt.ylim(0,1.0)
                plt.title('Values for Odd')
                plt.legend(['Probabilities', 'Singleton Masses'])
                plt.savefig('results/odd_dec_z_epochs_20_latent_10_p_30_filtered_prob_random_' + str(random) + '_old_batch_' + str(batch) + '.png', dpi=600) 
                matplotlib2tikz.save('results/odd_dec_z_epochs_20_latent_10_p_30_filtered_prob_random_' + str(random) + '_old_' + str(batch) + '.tex')  
                plt.close()

                print("odd evidential weights pos k", dst_obj.evidential_weights_pos_k)
                print("odd evidential weights neg k", dst_obj.evidential_weights_neg_k)

            if num == 0:

                plt.figure()
                width = 0.5
                p1 = plt.bar(np.arange(K), alpha_p.flatten(), width, color='blue', alpha = 0.5)
                p2 = plt.bar(np.arange(K), filtered_alpha_p_sparsemax[batch,num,:].flatten(), width, color='orange', alpha = 0.5)
                p3 = plt.bar(np.arange(K), norm_singletons.flatten(), width, color='green', alpha = 0.5) # /1./np.sum(dst_obj.output_mass_singletons.flatten())

                plt.xlabel('Z')
                plt.ylabel('Values')
                plt.ylim(0,1.0)
                plt.title('Values for Even')
                plt.legend(['Probabilities', 'Singleton Masses'])
                plt.savefig('results/even_dec_z_epochs_20_latent_10_p_30_filtered_prob_random_' + str(random) + '_old_' + str(batch) + '.png', dpi=600)  
                matplotlib2tikz.save('results/even_dec_z_epochs_20_latent_10_p_30_filtered_prob_random_' + str(random) + '_old_' + str(batch) + '.tex')  
                plt.close()

            # print("labels", c)
            print("singletons", dst_obj.output_mass_singletons)
            print("filtered probabilities", norm_singletons)
            print("probabilities", alpha_p)

        # plot the trajectories filtered out
        # [25, N, K]
        z_one_hot = np.reshape(dict['z_one_hot'], (25,batch_size,2,5))[:,batch,:,:]
        z = np.argmax(z_one_hot, axis=-1)
        print("z mask", filtered_mask, np.sum(filtered_mask))
        print("check", np.where(filtered_mask))

        keep_mask = 1 - filtered_mask
        keep_mask_sparsemax = 1 - filtered_mask_sparsemax

        color_index_blue = np.linspace(0,1,25)
        color_index_green = np.linspace(0,1,np.sum(keep_mask))

        counter = 0
        for i in range(x.shape[0]):
            plt.plot(x[i, :num_predictions, 0], x[i, :num_predictions, 1], color='blue', alpha=0.15)
            counter += 1

        counter = 0
        for i in np.where(keep_mask)[0]:
            plt.plot(x[i, :num_predictions, 0], x[i, :num_predictions, 1], color='green')
            counter += 1

        counter = 0
        for i in np.where(keep_mask_sparsemax)[0]:
            plt.plot(x[i, :num_predictions, 0], x[i, :num_predictions, 1], color='orange', alpha=0.5)
            counter += 1

        if not os.path.exists('figures_paper/'):
            os.makedirs('figures_paper/')

        plt.plot(gt[:,0], gt[:,1], color='black')
        plt.plot(c[:,0], c[:,1], '.', color='gray')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.savefig('figures_paper/keep_predictions_new_' + str(batch) + '.png')
        matplotlib2tikz.save('figures_paper/keep_predictions_new_tikz_' + str(batch) + '.tex')
        plt.close()

        # count the different filtered dimensions
        if np.sum(keep_mask) in keep_nums.keys():
            keep_nums[np.sum(keep_mask)] += 1
        else:
            keep_nums[np.sum(keep_mask)] = 1

    pkl.dump(filtered_alpha_p, open("filtered_alpha_p.pkl", "wb"))
    pkl.dump(filtered_alpha_p_sparsemax, open("filtered_alpha_p_sparsemax.pkl", "wb"))
    print("keep nums", keep_nums)

if __name__== "__main__":
    main()
