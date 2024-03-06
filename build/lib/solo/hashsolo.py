#!/usr/bin/env python
import os
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from scipy.stats import norm
from itertools import product
import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from scipy.sparse import issparse
from sklearn.metrics import calinski_harabasz_score

"""
HashSolo script provides a probabilistic cell hashing demultiplexing method
which generates a noise distribution and signal distribution for
each hashing barcode from empirically observed counts. These distributions
are updates from the global signal and noise barcode distributions, which
helps in the setting where not many cells are observed. Signal distributions
for a hashing barcode are estimated from samples where that hashing barcode
has the highest count. Noise distributions for a hashing barcode are estimated
from samples where that hashing barcode is one the k-2 lowest barcodes, where
k is the number of barcodes. A doublet should then have its two highest
barcode counts most likely coming from a signal distribution for those barcodes.
A singlet should have its highest barcode from a signal distribution, and its
second highest barcode from a noise distribution. A negative two highest
barcodes should come from noise distributions. We test each of these
hypotheses in a bayesian fashion, and select the most probable hypothesis.
"""


def _calculate_log_likelihoods(data, number_of_noise_barcodes):
    """Calculate log likelihoods for each hypothesis, negative, singlet, doublet

    Parameters
    ----------
    data : np.ndarray
        cells by hashing counts matrix
    number_of_noise_barcodes : int,
        number of barcodes to used to calculated noise distribution
    Returns
    -------
    log_likelihoods_for_each_hypothesis : np.ndarray
        a 2d np.array log likelihood of each hypothesis
    all_indices
    counter_to_barcode_combo
    """

    def gaussian_updates(data, mu_o, std_o):
        """Update parameters of your gaussian
        https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        Parameters
        ----------
        data : np.array
            1-d array of counts
        mu_o : float,
            global mean for hashing count distribution
        std_o : float,
            global std for hashing count distribution
        Returns
        -------
        float
            mean of gaussian
        float
            std of gaussian
        """
        lam_o = 1 / (std_o ** 2)
        n = len(data)
        lam = 1 / np.var(data) if len(data) > 1 else lam_o
        lam_n = lam_o + n * lam
        mu_n = (
            (np.mean(data) * n * lam + mu_o * lam_o) / lam_n if len(data) > 0 else mu_o
        )
        return mu_n, (1 / (lam_n / (n + 1))) ** (1 / 2)

    eps = 1e-15
    # probabilites for negative, singlet, doublets
    log_likelihoods_for_each_hypothesis = np.zeros((data.shape[0], 3))

    all_indices = np.empty(data.shape[0])
    num_of_barcodes = data.shape[1]
    number_of_non_noise_barcodes = (
        num_of_barcodes - number_of_noise_barcodes
        if number_of_noise_barcodes is not None
        else 2
    )
    num_of_noise_barcodes = num_of_barcodes - number_of_non_noise_barcodes

    # assume log normal
    data = np.log(data + 1)
    data_arg = np.argsort(data, axis=1)
    data_sort = np.sort(data, axis=1)

    # global signal and noise counts useful for when we have few cells
    # barcodes with the highest number of counts are assumed to be a true signal
    # barcodes with rank < k are considered to be noise
    global_signal_counts = np.ravel(data_sort[:, -1])
    global_noise_counts = np.ravel(data_sort[:, :-number_of_non_noise_barcodes])
    global_mu_signal_o, global_sigma_signal_o = np.mean(global_signal_counts), np.std(
        global_signal_counts
    )
    global_mu_noise_o, global_sigma_noise_o = np.mean(global_noise_counts), np.std(
        global_noise_counts
    )

    noise_params_dict = {}
    signal_params_dict = {}

    # for each barcode get  empirical noise and signal distribution parameterization
    for x in np.arange(num_of_barcodes):
        sample_barcodes = data[:, x]
        sample_barcodes_noise_idx = np.where(data_arg[:, :num_of_noise_barcodes] == x)[
            0
        ]
        sample_barcodes_signal_idx = np.where(data_arg[:, -1] == x)

        # get noise and signal counts
        noise_counts = sample_barcodes[sample_barcodes_noise_idx]
        signal_counts = sample_barcodes[sample_barcodes_signal_idx]

        # get parameters of distribution, assuming lognormal do update from global values
        noise_param = gaussian_updates(
            noise_counts, global_mu_noise_o, global_sigma_noise_o
        )
        signal_param = gaussian_updates(
            signal_counts, global_mu_signal_o, global_sigma_signal_o
        )
        noise_params_dict[x] = noise_param
        signal_params_dict[x] = signal_param

    counter_to_barcode_combo = {}
    counter = 0

    # for each combination of noise and signal barcode calculate probiltiy of in silico and real cell hypotheses
    for noise_sample_idx, signal_sample_idx in product(
        np.arange(num_of_barcodes), np.arange(num_of_barcodes)
    ):
        signal_subset = data_arg[:, -1] == signal_sample_idx
        noise_subset = data_arg[:, -2] == noise_sample_idx
        subset = signal_subset & noise_subset
        if sum(subset) == 0:
            continue

        indices = np.where(subset)[0]
        barcode_combo = "_".join([str(noise_sample_idx), str(signal_sample_idx)])
        all_indices[np.where(subset)[0]] = counter
        counter_to_barcode_combo[counter] = barcode_combo
        counter += 1
        noise_params = noise_params_dict[noise_sample_idx]
        signal_params = signal_params_dict[signal_sample_idx]

        # calculate probabilties for each hypothesis for each cell
        data_subset = data[subset]
        log_signal_signal_probs = np.log(
            norm.pdf(
                data_subset[:, signal_sample_idx],
                *signal_params[:-2],
                loc=signal_params[-2],
                scale=signal_params[-1]
            )
            + eps
        )
        signal_noise_params = signal_params_dict[noise_sample_idx]
        log_noise_signal_probs = np.log(
            norm.pdf(
                data_subset[:, noise_sample_idx],
                *signal_noise_params[:-2],
                loc=signal_noise_params[-2],
                scale=signal_noise_params[-1]
            )
            + eps
        )

        log_noise_noise_probs = np.log(
            norm.pdf(
                data_subset[:, noise_sample_idx],
                *noise_params[:-2],
                loc=noise_params[-2],
                scale=noise_params[-1]
            )
            + eps
        )
        log_signal_noise_probs = np.log(
            norm.pdf(
                data_subset[:, signal_sample_idx],
                *noise_params[:-2],
                loc=noise_params[-2],
                scale=noise_params[-1]
            )
            + eps
        )

        probs_of_negative = np.sum(
            [log_noise_noise_probs, log_signal_noise_probs], axis=0
        )
        probs_of_singlet = np.sum(
            [log_noise_noise_probs, log_signal_signal_probs], axis=0
        )
        probs_of_doublet = np.sum(
            [log_noise_signal_probs, log_signal_signal_probs], axis=0
        )
        log_probs_list = [probs_of_negative, probs_of_singlet, probs_of_doublet]

        # each cell and each hypothesis probability
        for prob_idx, log_prob in enumerate(log_probs_list):
            log_likelihoods_for_each_hypothesis[indices, prob_idx] = log_prob
    return log_likelihoods_for_each_hypothesis, all_indices, counter_to_barcode_combo


def _calculate_bayes_rule(data, priors, number_of_noise_barcodes):
    """
    Calculate bayes rule from log likelihoods

    Parameters
    ----------
    data : np.array
        Anndata object filled only with hashing counts
    priors : list,
        a list of your prior for each hypothesis
        first element is your prior for the negative hypothesis
        second element is your prior for the singlet hypothesis
        third element is your prior for the doublet hypothesis
        We use [0.01, 0.8, 0.19] by default because we assume the barcodes
        in your cell hashing matrix are those cells which have passed QC
        in the transcriptome space, e.g. UMI counts, pct mito reads, etc.
    number_of_noise_barcodes : int
        number of barcodes to used to calculated noise distribution
    Returns
    -------
    bayes_dict_results : dict
        'most_likely_hypothesis' key is a 1d np.array of the most likely hypothesis
        'probs_hypotheses' key is a 2d np.array probability of each hypothesis
        'log_likelihoods_for_each_hypothesis' key is a 2d np.array log likelihood of each hypothesis
    """
    priors = np.array(priors)
    log_likelihoods_for_each_hypothesis, _, _ = _calculate_log_likelihoods(
        data, number_of_noise_barcodes
    )
    probs_hypotheses = (
        np.exp(log_likelihoods_for_each_hypothesis)
        * priors
        / np.sum(
            np.multiply(np.exp(log_likelihoods_for_each_hypothesis), priors), axis=1
        )[:, None]
    )
    most_likely_hypothesis = np.argmax(probs_hypotheses, axis=1)
    return {
        "most_likely_hypothesis": most_likely_hypothesis,
        "probs_hypotheses": probs_hypotheses,
        "log_likelihoods_for_each_hypothesis": log_likelihoods_for_each_hypothesis,
    }


def _get_clusters(clustering_data: anndata.AnnData, resolutions: list):
    """
    Principled cell clustering
    Parameters
    ----------
    cell_hashing_adata : anndata.AnnData
        Anndata object filled only with hashing counts
    resolutions : list
        clustering resolutions for leiden
    Returns
    -------
    np.ndarray
        leiden clustering results for each cell
    """
    sc.pp.normalize_per_cell(clustering_data, counts_per_cell_after=1e4)
    sc.pp.log1p(clustering_data)
    sc.pp.highly_variable_genes(
        clustering_data, min_mean=0.0125, max_mean=3, min_disp=0.5
    )
    clustering_data = clustering_data[:, clustering_data.var["highly_variable"]]
    sc.pp.scale(clustering_data, max_value=10)
    sc.tl.pca(clustering_data, svd_solver="arpack")
    sc.pp.neighbors(clustering_data, n_neighbors=10, n_pcs=40)
    sc.tl.umap(clustering_data)
    best_ch_score = -np.inf

    for resolution in resolutions:
        sc.tl.leiden(clustering_data, resolution=resolution)

        ch_score = calinski_harabasz_score(
            clustering_data.X, clustering_data.obs["leiden"]
        )

        if ch_score > best_ch_score:
            clustering_data.obs["best_leiden"] = clustering_data.obs["leiden"].values
            best_ch_score = ch_score
    return clustering_data.obs["best_leiden"].values


def hashsolo(
    cell_hashing_adata: anndata.AnnData,
    priors: list = [0.01, 0.8, 0.19],
    pre_existing_clusters: str = None,
    clustering_data: anndata.AnnData = None,
    resolutions: list = [0.1, 0.25, 0.5, 0.75, 1],
    number_of_noise_barcodes: int = None,
    inplace: bool = True,
):
    """Demultiplex cell hashing dataset using HashSolo method

    Parameters
    ----------
    cell_hashing_adata : anndata.AnnData
        Anndata object filled only with hashing counts
    priors : list,
        a list of your prior for each hypothesis
        first element is your prior for the negative hypothesis
        second element is your prior for the singlet hypothesis
        third element is your prior for the doublet hypothesis
        We use [0.01, 0.8, 0.19] by default because we assume the barcodes
        in your cell hashing matrix are those cells which have passed QC
        in the transcriptome space, e.g. UMI counts, pct mito reads, etc.
    clustering_data : anndata.AnnData
        transcriptional data for clustering
    resolutions : list
        clustering resolutions for leiden
    pre_existing_clusters : str
        column in cell_hashing_adata.obs for how to break up demultiplexing
    inplace : bool
        To do operation in place

    Returns
    -------
    cell_hashing_adata : AnnData
        if inplace is False returns AnnData with demultiplexing results
        in .obs attribute otherwise does is in place
    """
    if issparse(cell_hashing_adata.X):
        cell_hashing_adata.X = np.array(cell_hashing_adata.X.todense())

    if clustering_data is not None:
        print(
            "This may take awhile we are running clustering at {} different resolutions".format(
                len(resolutions)
            )
        )
        if not all(clustering_data.obs_names == cell_hashing_adata.obs_names):
            raise ValueError(
                "clustering_data and cell hashing cell_hashing_adata must have same index"
            )
        cell_hashing_adata.obs["best_leiden"] = _get_clusters(
            clustering_data, resolutions
        )

    data = cell_hashing_adata.X
    num_of_cells = cell_hashing_adata.shape[0]
    results = pd.DataFrame(
        np.zeros((num_of_cells, 6)),
        columns=[
            "most_likely_hypothesis",
            "probs_hypotheses",
            "cluster_feature",
            "negative_hypothesis_probability",
            "singlet_hypothesis_probability",
            "doublet_hypothesis_probability",
        ],
        index=cell_hashing_adata.obs_names,
    )
    if clustering_data is not None or pre_existing_clusters is not None:
        cluster_features = (
            "best_leiden" if pre_existing_clusters is None else pre_existing_clusters
        )
        unique_cluster_features = np.unique(cell_hashing_adata.obs[cluster_features])
        for cluster_feature in unique_cluster_features:
            cluster_feature_bool_vector = (
                cell_hashing_adata.obs[cluster_features] == cluster_feature
            )
            posterior_dict = _calculate_bayes_rule(
                data[cluster_feature_bool_vector], priors, number_of_noise_barcodes
            )
            results.loc[
                cluster_feature_bool_vector, "most_likely_hypothesis"
            ] = posterior_dict["most_likely_hypothesis"]
            results.loc[
                cluster_feature_bool_vector, "cluster_feature"
            ] = cluster_feature
            results.loc[
                cluster_feature_bool_vector, "negative_hypothesis_probability"
            ] = posterior_dict["probs_hypotheses"][:, 0]
            results.loc[
                cluster_feature_bool_vector, "singlet_hypothesis_probability"
            ] = posterior_dict["probs_hypotheses"][:, 1]
            results.loc[
                cluster_feature_bool_vector, "doublet_hypothesis_probability"
            ] = posterior_dict["probs_hypotheses"][:, 2]
    else:
        posterior_dict = _calculate_bayes_rule(data, priors, number_of_noise_barcodes)
        results.loc[:, "most_likely_hypothesis"] = posterior_dict[
            "most_likely_hypothesis"
        ]
        results.loc[:, "cluster_feature"] = 0
        results.loc[:, "negative_hypothesis_probability"] = posterior_dict[
            "probs_hypotheses"
        ][:, 0]
        results.loc[:, "singlet_hypothesis_probability"] = posterior_dict[
            "probs_hypotheses"
        ][:, 1]
        results.loc[:, "doublet_hypothesis_probability"] = posterior_dict[
            "probs_hypotheses"
        ][:, 2]

    cell_hashing_adata.obs["most_likely_hypothesis"] = results.loc[
        cell_hashing_adata.obs_names, "most_likely_hypothesis"
    ]
    cell_hashing_adata.obs["cluster_feature"] = results.loc[
        cell_hashing_adata.obs_names, "cluster_feature"
    ]
    cell_hashing_adata.obs["negative_hypothesis_probability"] = results.loc[
        cell_hashing_adata.obs_names, "negative_hypothesis_probability"
    ]
    cell_hashing_adata.obs["singlet_hypothesis_probability"] = results.loc[
        cell_hashing_adata.obs_names, "singlet_hypothesis_probability"
    ]
    cell_hashing_adata.obs["doublet_hypothesis_probability"] = results.loc[
        cell_hashing_adata.obs_names, "doublet_hypothesis_probability"
    ]

    cell_hashing_adata.obs["Classification"] = None
    cell_hashing_adata.obs.loc[
        cell_hashing_adata.obs["most_likely_hypothesis"] == 2, "Classification"
    ] = "Doublet"
    cell_hashing_adata.obs.loc[
        cell_hashing_adata.obs["most_likely_hypothesis"] == 0, "Classification"
    ] = "Negative"
    all_sings = cell_hashing_adata.obs["most_likely_hypothesis"] == 1
    singlet_sample_index = np.argmax(cell_hashing_adata.X[all_sings], axis=1)
    cell_hashing_adata.obs.loc[
        all_sings, "Classification"
    ] = cell_hashing_adata.var_names[singlet_sample_index]

    return cell_hashing_adata if not inplace else None


def plot_qc_checks_cell_hashing(
    cell_hashing_adata: anndata.AnnData, alpha: float = 0.05, fig_path: str = None
):
    """Plot HashSolo demultiplexing results

    Parameters
    ----------
    cell_hashing_adata : Anndata
        Anndata object filled only with hashing counts
    alpha : float
        Tranparency of scatterplot points
    fig_path : str
        Path to save figure
    Returns
    -------
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_hashing_demultiplexing = cell_hashing_adata.obs
    cell_hashing_demultiplexing["log_counts"] = np.log(
        np.sum(cell_hashing_adata.X, axis=1)
    )
    number_of_clusters = (
        cell_hashing_demultiplexing["cluster_feature"].drop_duplicates().shape[0]
    )
    fig, all_axes = plt.subplots(
        number_of_clusters, 4, figsize=(40, 10 * number_of_clusters)
    )
    counter = 0
    for cluster_feature, group in cell_hashing_demultiplexing.groupby(
        "cluster_feature"
    ):
        if number_of_clusters > 1:
            axes = all_axes[counter]
        else:
            axes = all_axes

        ax = axes[0]
        ax.plot(
            group["log_counts"],
            group["negative_hypothesis_probability"],
            "bo",
            alpha=alpha,
        )
        ax.set_title("Probability of negative hypothesis vs log hashing counts")
        ax.set_ylabel("Probability of negative hypothesis")
        ax.set_xlabel("Log hashing counts")

        ax = axes[1]
        ax.plot(
            group["log_counts"],
            group["singlet_hypothesis_probability"],
            "bo",
            alpha=alpha,
        )
        ax.set_title("Probability of singlet hypothesis vs log hashing counts")
        ax.set_ylabel("Probability of singlet hypothesis")
        ax.set_xlabel("Log hashing counts")

        ax = axes[2]
        ax.plot(
            group["log_counts"],
            group["doublet_hypothesis_probability"],
            "bo",
            alpha=alpha,
        )
        ax.set_title("Probability of doublet hypothesis vs log hashing counts")
        ax.set_ylabel("Probability of doublet hypothesis")
        ax.set_xlabel("Log hashing counts")

        ax = axes[3]
        group["Classification"].value_counts().plot.bar(ax=ax)
        ax.set_title("Count of each samples classification")
        counter += 1
    plt.show()
    if fig_path is not None:
        fig.savefig(fig_path, dpi=300, format="pdf")


def main():
    usage = "hashsolo"
    parser = ArgumentParser(usage, formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        dest="data_file", help="h5ad file containing cell hashing counts"
    )
    parser.add_argument(
        "-j",
        dest="model_json_file",
        default=None,
        help="json file to pass optional arguments",
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        default="hashsolo_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "-c",
        dest="clustering_data",
        default=None,
        help="h5ad file with count transcriptional data to\
                    perform clustering on",
    )
    parser.add_argument(
        "-p",
        dest="pre_existing_clusters",
        default=None,
        help="column in cell_hashing_data_file.obs to \
                        specifying different cell types or clusters",
    )
    parser.add_argument(
        "-q",
        dest="plot_name",
        default="hashing_qc_plots.pdf",
        help="name of plot to output",
    )
    parser.add_argument(
        "-n",
        dest="number_of_noise_barcodes",
        default=None,
        help="Number of barcodes to use to create noise \
                        distribution",
    )

    args = parser.parse_args()

    model_json_file = args.model_json_file
    if model_json_file is not None:
        # read parameters
        with open(model_json_file) as model_json_open:
            params = json.load(model_json_open)
    else:
        params = {}
    data_file = args.data_file
    data_ext = os.path.splitext(data_file)[-1]
    if data_ext == ".h5ad":
        cell_hashing_adata = anndata.read(data_file)
    else:
        print("Unrecognized file format")

    if args.clustering_data is not None:
        clustering_data_file = args.clustering_data
        clustering_data_ext = os.path.splitext(clustering_data_file)[-1]
        if clustering_data_ext == ".h5ad":
            clustering_data = anndata.read(clustering_data_file)
        else:
            print("Unrecognized file format for clustering data")
    else:
        clustering_data = None

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    hashsolo(
        cell_hashing_adata,
        pre_existing_clusters=args.pre_existing_clusters,
        number_of_noise_barcodes=args.number_of_noise_barcodes,
        clustering_data=clustering_data,
        **params
    )
    cell_hashing_adata.write(os.path.join(args.out_dir, "hashsoloed.h5ad"))
    plot_qc_checks_cell_hashing(
        cell_hashing_adata, fig_path=os.path.join(args.out_dir, args.plot_name)
    )


###############################################################################
# __main__
###############################################################################


if __name__ == "__main__":
    main()
