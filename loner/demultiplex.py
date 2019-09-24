#!/usr/bin/env python
import os
import json
from optparse import OptionParser

from scipy.stats import norm
from itertools import product
import anndata
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import calinski_harabasz_score


def _calculate_probabilities(z, total_samples):
    eps = 1e-15
    # probabilites for negative, singlet, doublets
    probabilities_for_each_hypothesis = np.zeros((z.shape[0], 3))

    all_indices = np.empty(z.shape[0])
    insilico_probs = []
    num_of_barcodes = z.shape[1]
    num_of_noise_barcodes = num_of_barcodes - 2

    z_arg = np.argsort(z, axis=1)

    noise_params_dict = {}
    signal_params_dict = {}

    # for each barcode get and empirical noise and signal distribution parameterization
    for x in np.arange(num_of_barcodes):
        sample_barcodes = z[:, x]
        sample_barcodes_noise_idx = np.where(z_arg[:, :num_of_noise_barcodes] == x)[0]
        sample_barcodes_signal_idx = np.where(z_arg[:, -1] == x)
        # get noise and signal counts
        noise_counts = np.log(sample_barcodes[sample_barcodes_noise_idx] + 1)
        signal_counts = np.log(sample_barcodes[sample_barcodes_signal_idx] + 1)

        # get parameters of distribution, assuming lognormal
        noise_param = (np.mean(noise_counts), np.std(noise_counts))
        signal_param = (np.mean(signal_counts), np.std(signal_counts))

        noise_params_dict[x] = noise_param
        signal_params_dict[x] = signal_param

    counter_to_barcode_combo = {}
    counter = 0

    # for each comibnation of noise and signal barcode calculate probiltiy of in silico and real cell hypotheses
    for noise_sample_idx, signal_sample_idx in product(np.arange(num_of_barcodes),
                                                       np.arange(num_of_barcodes)):
        signal_subset = (z_arg[:, -1] == signal_sample_idx)
        noise_subset = (z_arg[:, -2] == noise_sample_idx)
        subset = (signal_subset & noise_subset)
        if sum(subset) == 0:
            continue

        indices = np.where(subset)[0]
        barcode_combo = "_".join([str(noise_sample_idx),
                                  str(signal_sample_idx)])
        all_indices[np.where(subset)[0]] = counter
        counter_to_barcode_combo[counter] = barcode_combo
        counter += 1
        noise_params = noise_params_dict[noise_sample_idx]
        signal_params = signal_params_dict[signal_sample_idx]

        # create samples from noise and signal distribution
        noise_samples = norm.rvs(*noise_params[:-2], loc=noise_params[-2],
                                 scale=noise_params[-1], size=total_samples)
        signal_samples = norm.rvs(*signal_params[:-2], loc=signal_params[-2],
                                  scale=signal_params[-1], size=total_samples)

        # make in silico samples
        in_silico_samples = np.hstack([noise_samples, signal_samples])

        # get probabilities for each in silico cells
        insilico_signal_probs = norm.pdf(np.ravel(in_silico_samples),
                                         *signal_params[:-2],
                                         loc=signal_params[-2],
                                         scale=signal_params[-1])
        insilico_signal_probs = np.transpose(np.reshape(insilico_signal_probs,
                                                        (2, total_samples)))
        insilico_noise_probs = norm.pdf(np.ravel(in_silico_samples),
                                        *noise_params[:-2],
                                        loc=noise_params[-2],
                                        scale=noise_params[-1])
        insilico_noise_probs = np.transpose(np.reshape(insilico_noise_probs,
                                                       (2, total_samples)))
        log_insilico_signal_probs = np.log(insilico_signal_probs + eps)
        log_insilico_noise_probs = np.log(insilico_noise_probs + eps)

        # calculate probability of each hypothesis for in silico cells
        log_probs_of_insilico_negative = np.sum([log_insilico_noise_probs[:, -2], log_insilico_noise_probs[:, -1]], axis=0)
        log_probs_of_insilico_singlet = np.sum([log_insilico_noise_probs[:, -2], log_insilico_signal_probs[:, -1]], axis=0)
        log_probs_of_insilico_doublet = np.sum([log_insilico_signal_probs[:, -2], log_insilico_signal_probs[:, -1]], axis=0)
        log_probs_insilico_list = [log_probs_of_insilico_negative, log_probs_of_insilico_singlet, log_probs_of_insilico_doublet]

        # calculate probabilties for each hypothesis for each cell
        z_subset = np.log(z[subset] + 1)
        log_signal_signal_probs = np.log(norm.pdf(z_subset[:, signal_sample_idx], *signal_params[:-2], loc=signal_params[-2], scale=signal_params[-1])  + eps)
        signal_noise_params = signal_params_dict[noise_sample_idx]
        log_noise_signal_probs = np.log(norm.pdf(z_subset[:, noise_sample_idx], *signal_noise_params[:-2], loc=signal_noise_params[-2], scale=signal_noise_params[-1]) + eps)

        log_noise_noise_probs = np.log(norm.pdf(z_subset[:, noise_sample_idx], *noise_params[:-2], loc=noise_params[-2], scale=noise_params[-1])  + eps)
        log_signal_noise_probs = np.log(norm.pdf(z_subset[:, signal_sample_idx], *noise_params[:-2], loc=noise_params[-2], scale=noise_params[-1]) + eps )

        probs_of_negative = np.sum([log_noise_noise_probs, log_signal_noise_probs] , axis=0)
        probs_of_singlet = np.sum([log_noise_noise_probs, log_signal_signal_probs], axis=0)
        probs_of_doublet = np.sum([log_noise_signal_probs, log_signal_signal_probs], axis=0)
        log_probs_list = [probs_of_negative, probs_of_singlet, probs_of_doublet]

        # each cell and each hypothesis calculate a p-value threshold from
        # the in silico sample
        for prob_idx, log_prob in enumerate(log_probs_list):
            probabilities_for_each_hypothesis[indices, prob_idx] = log_prob
        insilico_probs.append(log_probs_insilico_list)
    return probabilities_for_each_hypothesis, all_indices, insilico_probs, counter_to_barcode_combo


def _calculate_llrts(data, doublet_llr_threshold, negative_llr_threshold, total_samples):
    '''
    '''
    probabilities_for_each_hypothesis, _, insilico_probs, _ = _calculate_probabilities(data, total_samples)
    dubs = np.zeros(probabilities_for_each_hypothesis.shape[0], dtype=bool)
    negs = np.zeros(probabilities_for_each_hypothesis.shape[0], dtype=bool)

    # get null distribution from all insilico examples
    all_insilico_llrt_d_s = np.ravel(np.array(insilico_probs)[:, 2]) - np.ravel(np.array(insilico_probs)[:, 1])
    all_insilico_llrt_n_s = np.ravel(np.array(insilico_probs)[:, 0]) - np.ravel(np.array(insilico_probs)[:, 1])
    dub_thresh = np.percentile(all_insilico_llrt_d_s, doublet_llr_threshold)
    neg_thresh = np.percentile(all_insilico_llrt_n_s, negative_llr_threshold)

    # do log lrt test for doublet over singlet
    llrt_d_s = probabilities_for_each_hypothesis[:,2] - probabilities_for_each_hypothesis[:,1]

    # do log lrt test for negative over singlet
    llrt_n_s = probabilities_for_each_hypothesis[:,0] - probabilities_for_each_hypothesis[:,1]

    # identify dubs and singlets
    dubs = llrt_d_s > dub_thresh
    negs = llrt_n_s > neg_thresh
    return {"llrt_d_s": llrt_d_s,
            "llrt_n_s": llrt_n_s,
            "dubs": dubs,
            "negs": negs,
            "dub_thresh": dub_thresh,
            "neg_thresh": neg_thresh}


def _get_clusters(clustering_data: anndata.AnnData,
                  resolutions: list):
    '''
    '''
    sc.pp.normalize_per_cell(clustering_data, counts_per_cell_after=1e4)
    sc.pp.log1p(clustering_data)
    sc.pp.highly_variable_genes(clustering_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    clustering_data = clustering_data[:, clustering_data.var['highly_variable']]
    sc.pp.scale(clustering_data, max_value=10)
    sc.tl.pca(clustering_data, svd_solver='arpack')
    sc.pp.neighbors(clustering_data, n_neighbors=10, n_pcs=40)
    sc.tl.umap(clustering_data)
    best_ch_score = -np.inf

    for resolution in resolutions:
        sc.tl.leiden(clustering_data, resolution=resolution)

        ch_score = calinski_harabasz_score(clustering_data.X, clustering_data.obs['leiden'])
        if ch_score > best_ch_score:
            clustering_data.obs['best_leiden'] = clustering_data.obs['leiden'].values
            best_ch_score = ch_score
    return clustering_data.obs['best_leiden'].values


def demultiplex_cell_hashing(adata: anndata.AnnData,
                             negative_p_value: float = .00001,
                             doublet_p_value: float = .0025,
                             total_samples: int = 10000,
                             pre_existing_clusters: str = None,
                             inplace: bool = True,
                             resolutions: list = [.1, .25, .5, .75, .9, 1, 2, 5],
                             clustering_data: anndata.AnnData = None,):
    '''Demultiplex cell hashing dataset

    Attributes
    ----------
    adata : Anndata
        Anndata object filled only with hashing counts
    negative_p_value : float
        P-value threshold for calling a cell a negative
    doublet_p_value : float
        P-value threshold for calling a cell a doublet
    total_samples : int
        total  number of sample to generate an empirical distribution for testing
    on_clusters: bool, iterable
    inplace : bool
        To do operation in place
    '''

    if clustering_data is not None:
        print("This may take awhile we are running clustering at {} different resolutions".format(len(resolutions)))
        if not all(clustering_data.obs_names == adata.obs_names):
            raise ValueError("clustering_data and cell hashing adata must have same index")
        adata.obs["best_leiden"] = _get_clusters(clustering_data, resolutions)

    negative_llr_threshold = 100 - negative_p_value * 100
    doublet_llr_threshold = 100 - doublet_p_value * 100
    data = adata.X
    num_of_cells = adata.shape[0]
    results = pd.DataFrame(np.zeros((num_of_cells, 7)),
                           columns=['llrt_d_s',
                                    'llrt_n_s',
                                    'dubs',
                                    'negs',
                                    'dub_thresh',
                                    'neg_thresh',
                                    'cluster_feature'],
                           index=adata.obs_names)
    if clustering_data is not None or pre_existing_clusters is not None:
        cluster_features = "best_leiden" if pre_existing_clusters is None else pre_existing_clusters
        unique_cluster_features = adata.obs[cluster_features].drop_duplicates()
        for cluster_feature in unique_cluster_features:
            cluster_feature_bool_vector = adata.obs[cluster_features] == cluster_feature
            llrts_dict = _calculate_llrts(data[cluster_feature_bool_vector], doublet_llr_threshold, negative_llr_threshold, total_samples)
            for k, v in llrts_dict.items():
                results.loc[cluster_feature_bool_vector, k] = v
            results.loc[cluster_feature_bool_vector, 'cluster_feature'] = cluster_feature
    else:
        llrts_dict = _calculate_llrts(data, doublet_llr_threshold, negative_llr_threshold, total_samples)
        for k, v in llrts_dict.items():
            results.loc[:, k] = v
        results.loc[:, 'cluster_feature'] = 0
    adata.obs = pd.merge(adata.obs, results, left_index=True, right_index=True)
    adata.obs["Classification"] = None
    adata.obs.loc[results['dubs'], "Classification"] = "Doublet"

    # note this will overwrite the doublet classification should be rare
    adata.obs.loc[results['negs'], "Classification"] = "Negative"

    all_sings = ~(results['dubs'] | results['negs'])
    singlet_sample_index = np.argmax(data[all_sings], axis=1)
    adata.obs.loc[all_sings, "Classification"] = adata.var_names[singlet_sample_index]
    return adata if not inplace else None


def plot_qc_checks_cell_hashing(adata: anndata.AnnData,
                                alpha: float = .05,
                                fig_path: str = None):
    '''Plot demultiplex cell hashing results

    Attributes
    ----------
    adata : Anndata
        Anndata object filled only with hashing counts
    alpha : float
        Tranparency of scatterplot points
    fig_path : str
        Path to save figure
    '''

    cell_hashing_demultiplexing = adata.obs
    cell_hashing_demultiplexing['log_counts'] = np.log(np.sum(adata.X, axis=1))
    number_of_clusters = cell_hashing_demultiplexing["cluster_feature"].drop_duplicates().shape[0]
    fig, all_axes = plt.subplots(number_of_clusters, 3, figsize=(30, 10 * number_of_clusters))
    counter = 0
    for cluster_feature, group in cell_hashing_demultiplexing.groupby("cluster_feature"):
        if number_of_clusters > 1:
            axes = all_axes[counter]
        else:
            axes = all_axes
        ax = axes[0]
        ax.plot(group["log_counts"], group['llrt_n_s'], 'bo', alpha=alpha)
        ax.set_title("LLR n/s vs log hashing counts")
        ax.set_ylabel("LLR negative - singlet")
        ax.set_xlabel("Log hashing counts")
        ax.axhline(group['neg_thresh'][0], label="LLR negative threshold")
        ax.legend()

        ax = axes[1]
        ax.plot(group["log_counts"], group['llrt_d_s'], 'bo', alpha=alpha)
        ax.set_title("LLR d/s vs log hashing counts")
        ax.set_ylabel("LLR doublet - singlet")
        ax.set_xlabel("Log hashing counts")
        ax.axhline(group['dub_thresh'][0], label="LLR doublet threshold")
        ax.legend()

        ax = axes[2]
        group['Classification'].value_counts().plot.bar(ax=ax)
        ax.set_title("Count of each samples classification")
        counter += 1
    plt.show()
    if fig_path is not None:
        fig.savefig(fig_path, format='png')


def main():
    usage = 'usage: %prog [options] <model_json> <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_dir',
                      default='solo_demultiplex')
    parser.add_option('-c', dest='clustering_data',
                      default=None)
    parser.add_option('-p', dest='pre_existing_clusters',
                      default=None)
    parser.add_option('-q', dest='plot_name',
                      default="hashing_qc_plots.png")
    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide model json and data loom')
    else:
        model_json_file = args[0]
        data_file = args[1]

    # read parameters
    with open(model_json_file) as model_json_open:
        params = json.load(model_json_open)

    data_ext = os.path.splitext(data_file)[-1]
    if data_ext == '.h5ad':
        adata = anndata.read(data_file)
    else:
        print('Unrecognized file format')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    demultiplex_cell_hashing(adata,
                             pre_existing_clusters=options.pre_existing_clusters,
                             clustering_data=options.clustering_data,
                             **params)
    adata.write(os.path.join(options.out_dir, "hashing_demultiplexed.h5ad"))
    plot_qc_checks_cell_hashing(adata, os.path.join(options.out_dir, options.plot_name))

###############################################################################
# __main__
###############################################################################


if __name__ == '__main__':
    main()