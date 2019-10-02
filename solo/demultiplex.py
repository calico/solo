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


def _calculate_probabilities(z):
    '''
    '''
    def gaussian_updates(data, mu_o, std_o):
        # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        lam_o = 1/(std_o**2)
        n = len(data)
        lam = 1/np.var(data) if len(data) > 1 else lam_o
        lam_n = lam_o + n*lam
        mu_n = (np.mean(data)*n*lam + mu_o*lam_o)/lam_n if len(data) > 0 else mu_o
        return mu_n, (1 / (lam_n / (n + 1)))**(1/2)

    eps = 1e-15
    # probabilites for negative, singlet, doublets
    probabilities_for_each_hypothesis = np.zeros((z.shape[0], 3))

    all_indices = np.empty(z.shape[0])
    num_of_barcodes = z.shape[1]
    num_of_noise_barcodes = num_of_barcodes - 2

    # assume log normal
    z = np.log(z + 1)
    z_arg = np.argsort(z, axis=1)
    z_sort = np.sort(z, axis=1)

    # global signal and noise counts useful for when we have few cells
    global_signal_counts = np.ravel(z_sort[:, -1])
    global_noise_counts = np.ravel(z_sort[:, :-2])
    global_mu_signal_o, global_sigma_signal_o = np.mean(global_signal_counts), np.std(global_signal_counts)
    global_mu_noise_o, global_sigma_noise_o = np.mean(global_noise_counts), np.std(global_noise_counts)

    noise_params_dict = {}
    signal_params_dict = {}

    # for each barcode get  empirical noise and signal distribution parameterization
    for x in np.arange(num_of_barcodes):
        sample_barcodes = z[:, x]
        sample_barcodes_noise_idx = np.where(z_arg[:, :num_of_noise_barcodes] == x)[0]
        sample_barcodes_signal_idx = np.where(z_arg[:, -1] == x)

        # get noise and signal counts
        noise_counts = sample_barcodes[sample_barcodes_noise_idx]
        signal_counts = sample_barcodes[sample_barcodes_signal_idx]

        # get parameters of distribution, assuming lognormal do update from global values
        noise_param = gaussian_updates(noise_counts, global_mu_noise_o, global_sigma_noise_o)
        signal_param = gaussian_updates(signal_counts, global_mu_signal_o, global_sigma_signal_o)
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

        # calculate probabilties for each hypothesis for each cell
        z_subset = z[subset]
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
    return probabilities_for_each_hypothesis, all_indices, counter_to_barcode_combo


def _calculate_bayes_rule(data, priors):
    '''
    '''
    log_likelihoods_for_each_hypothesis, _, _ = _calculate_probabilities(data)
    probs_hypotheses = np.exp(log_likelihoods_for_each_hypothesis) * np.array(priors) / np.prod(np.multiply(np.exp(log_likelihoods_for_each_hypothesis), np.array(priors)), axis=1)[:, None]
    most_likeli_hypothesis = np.argmax(probs_hypotheses, axis=1)
    return {"most_likeli_hypothesis": most_likeli_hypothesis,
            "probs_hypotheses": probs_hypotheses}


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


def demultiplex_cell_hashing(cell_hashing_adata: anndata.AnnData,
                             priors: list = [.01, .5, .49],
                             pre_existing_clusters: str = None,
                             clustering_data: anndata.AnnData = None,
                             resolutions: list = [.1, .25, .5, .75, 1],
                             inplace: bool = True,
                             ):
    '''Demultiplex cell hashing dataset

    Attributes
    ----------
    cell_hashing_adata : anndata.AnnData
        Anndata object filled only with hashing counts
    priors : list,
        P-value threshold for calling a cell a negative
    pre_existing_clusters : str
        column in cell_hashing_adata for how to break up demultiplexing
    clustering_data : anndata.AnnData
        transcriptional data for clustering
    resolutions : list
        clustering resolutions for leiden
    inplace : bool
        To do operation in place
    '''

    if clustering_data is not None:
        print("This may take awhile we are running clustering at {} different resolutions".format(len(resolutions)))
        if not all(clustering_data.obs_names == cell_hashing_adata.obs_names):
            raise ValueError("clustering_data and cell hashing cell_hashing_adata must have same index")
        cell_hashing_adata.obs["best_leiden"] = _get_clusters(clustering_data, resolutions)

    data = cell_hashing_adata.X
    num_of_cells = cell_hashing_adata.shape[0]
    results = pd.DataFrame(np.zeros((num_of_cells, 6)),
                           columns=['most_likeli_hypothesis',
                                    'probs_hypotheses',
                                    'cluster_feature',
                                    'negative_hypothesis_probability',
                                    'singlet_hypothesis_probability',
                                    'doublet_hypothesis_probability',],
                           index=cell_hashing_adata.obs_names)

    # more lines than it needs to be...
    if clustering_data is not None or pre_existing_clusters is not None:
        cluster_features = "best_leiden" if pre_existing_clusters is None else pre_existing_clusters
        unique_cluster_features = cell_hashing_adata.obs[cluster_features].drop_duplicates()
        for cluster_feature in unique_cluster_features:
            cluster_feature_bool_vector = cell_hashing_adata.obs[cluster_features] == cluster_feature
            posterior_dict = _calculate_bayes_rule(data[cluster_feature_bool_vector], priors)
            results.loc[cluster_feature_bool_vector, 'most_likeli_hypothesis'] = posterior_dict['most_likeli_hypothesis']
            results.loc[cluster_feature_bool_vector, 'cluster_feature'] = cluster_feature
            results.loc[cluster_feature_bool_vector, 'negative_hypothesis_probability'] = posterior_dict["probs_hypotheses"][:, 0]
            results.loc[cluster_feature_bool_vector, 'singlet_hypothesis_probability'] = posterior_dict["probs_hypotheses"][:, 1]
            results.loc[cluster_feature_bool_vector, 'doublet_hypothesis_probability'] = posterior_dict["probs_hypotheses"][:, 2]
    else:
        posterior_dict = _calculate_bayes_rule(data, priors)
        results.loc[:, 'most_likeli_hypothesis'] = posterior_dict['most_likeli_hypothesis']
        results.loc[:, 'cluster_feature'] = 0
        results.loc[:, 'negative_hypothesis_probability'] = posterior_dict["probs_hypotheses"][:, 0]
        results.loc[:, 'singlet_hypothesis_probability'] = posterior_dict["probs_hypotheses"][:, 1]
        results.loc[:, 'doublet_hypothesis_probability'] = posterior_dict["probs_hypotheses"][:, 2]

    cell_hashing_adata.obs['most_likeli_hypothesis'] = results.loc[cell_hashing_adata.obs_names, 'most_likeli_hypothesis']
    cell_hashing_adata.obs['cluster_feature'] = results.loc[cell_hashing_adata.obs_names, 'cluster_feature']
    cell_hashing_adata.obs['negative_hypothesis_probability'] = results.loc[cell_hashing_adata.obs_names, 'negative_hypothesis_probability']
    cell_hashing_adata.obs['singlet_hypothesis_probability'] = results.loc[cell_hashing_adata.obs_names, 'singlet_hypothesis_probability']
    cell_hashing_adata.obs['doublet_hypothesis_probability'] = results.loc[cell_hashing_adata.obs_names, 'doublet_hypothesis_probability']

    cell_hashing_adata.obs["Classification"] = None
    cell_hashing_adata.obs.loc[cell_hashing_adata.obs["most_likeli_hypothesis"] == 2, "Classification"] = "Doublet"
    cell_hashing_adata.obs.loc[cell_hashing_adata.obs["most_likeli_hypothesis"] == 0, "Classification"] = "Negative"
    all_sings = cell_hashing_adata.obs["most_likeli_hypothesis"] == 1
    singlet_sample_index = np.argmax(cell_hashing_adata.X[all_sings], axis=1)
    cell_hashing_adata.obs.loc[all_sings, "Classification"] = cell_hashing_adata.var_names[singlet_sample_index]

    return cell_hashing_adata if not inplace else None


def plot_qc_checks_cell_hashing(cell_hashing_adata: anndata.AnnData,
                                alpha: float = .05,
                                fig_path: str = None):
    '''Plot demultiplex cell hashing results

    Attributes
    ----------
    cell_hashing_adata : Anndata
        Anndata object filled only with hashing counts
    alpha : float
        Tranparency of scatterplot points
    fig_path : str
        Path to save figure
    '''

    cell_hashing_demultiplexing = cell_hashing_adata.obs
    cell_hashing_demultiplexing['log_counts'] = np.log(np.sum(cell_hashing_adata.X, axis=1))
    number_of_clusters = cell_hashing_demultiplexing["cluster_feature"].drop_duplicates().shape[0]
    fig, all_axes = plt.subplots(number_of_clusters, 4, figsize=(40, 10 * number_of_clusters))
    counter = 0
    for cluster_feature, group in cell_hashing_demultiplexing.groupby("cluster_feature"):
        if number_of_clusters > 1:
            axes = all_axes[counter]
        else:
            axes = all_axes

        ax = axes[0]
        ax.plot(group["log_counts"], group['negative_hypothesis_probability'], 'bo', alpha=alpha)
        ax.set_title("Probability of negative hypothesis vs log hashing counts")
        ax.set_ylabel("Probability of negative hypothesis")
        ax.set_xlabel("Log hashing counts")

        ax = axes[1]
        ax.plot(group["log_counts"],  group['singlet_hypothesis_probability'], 'bo', alpha=alpha)
        ax.set_title("Probability of singlet hypothesis vs log hashing counts")
        ax.set_ylabel("Probability of singlet hypothesis")
        ax.set_xlabel("Log hashing counts")

        ax = axes[2]
        ax.plot(group["log_counts"],  group['doublet_hypothesis_probability'], 'bo', alpha=alpha)
        ax.set_title("Probability of doublet hypothesis vs log hashing counts")
        ax.set_ylabel("Probability of doublet hypothesis")
        ax.set_xlabel("Log hashing counts")

        ax = axes[3]
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
        cell_hashing_adata = anndata.read(data_file)
    else:
        print('Unrecognized file format')

    if options.clustering_data is not None:
        clustering_data_file = options.clustering_data
        clustering_data_ext = os.path.splitext(clustering_data_file)[-1]
        if clustering_data_ext == '.h5ad':
            clustering_data = anndata.read(clustering_data_file)
        else:
            print('Unrecognized file format for clustering data')
    else:
        clustering_data = None

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    demultiplex_cell_hashing(cell_hashing_adata,
                             pre_existing_clusters=options.pre_existing_clusters,
                             clustering_data=clustering_data,
                             **params)
    cell_hashing_adata.write(os.path.join(options.out_dir, "hashing_demultiplexed.h5ad"))
    plot_qc_checks_cell_hashing(cell_hashing_adata, fig_path=os.path.join(options.out_dir, options.plot_name))

###############################################################################
# __main__
###############################################################################


if __name__ == '__main__':
    main()
