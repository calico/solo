#!/usr/bin/env python
import json
import os
import umap
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pkg_resources

import numpy as np
from sklearn.metrics import *
from scipy.special import softmax
from scanpy import read_10x_mtx

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import scvi
from scvi.data import read_h5ad, read_loom, setup_anndata
from scvi.model import SCVI
from scvi.external import SOLO

from .utils import knn_smooth_pred_class

"""
solo.py

Simulate doublets, train a VAE, and then a classifier on top.
"""


###############################################################################
# main
###############################################################################


def main():
    usage = "solo"
    parser = ArgumentParser(usage, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-j",
        dest="model_json_file",
        help="json file to pass VAE parameters",
        required="--version" not in sys.argv,
    )
    parser.add_argument(
        "-d",
        dest="data_path",
        help="path to h5ad, loom, or 10x mtx dir cell by genes counts",
        required="--version" not in sys.argv,
    )
    parser.add_argument(
        "--set-reproducible-seed",
        dest="reproducible_seed",
        default=None,
        type=int,
        help="Reproducible seed, give an int to set seed",
    )
    parser.add_argument(
        "--doublet-depth",
        dest="doublet_depth",
        default=2.0,
        type=float,
        help="Depth multiplier for a doublet relative to the \
                        average of its constituents",
    )
    parser.add_argument(
        "-g", dest="gpu", default=True, action="store_true", help="Run on GPU"
    )
    parser.add_argument(
        "-a",
        dest="anndata_output",
        default=False,
        action="store_true",
        help="output modified anndata object with solo scores \
                        Only works for anndata",
    )
    parser.add_argument("-o", dest="out_dir", default="solo_out")
    parser.add_argument(
        "-r",
        dest="doublet_ratio",
        default=2,
        type=int,
        help="Ratio of doublets to true \
                        cells",
    )
    parser.add_argument(
        "-s",
        dest="seed",
        default=None,
        help="Path to previous solo output  \
                        directory. Seed VAE models with previously \
                        trained solo model. Directory structure is assumed to \
                        be the same as solo output directory structure. \
                        should at least have a vae.pt a pickled object of \
                        vae weights and a latent.npy an np.ndarray of the \
                        latents of your cells.",
    )
    parser.add_argument(
        "-e",
        dest="expected_number_of_doublets",
        help="Experimentally expected number of doublets",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-p",
        dest="plot",
        default=False,
        action="store_true",
        help="Plot outputs for solo",
    )
    parser.add_argument(
        "-recalibrate_scores",
        dest="recalibrate_scores",
        default=False,
        action="store_true",
        help="Recalibrate doublet scores (not recommended anymore)",
    )
    parser.add_argument(
        "--version",
        dest="version",
        default=False,
        action="store_true",
        help="Get version of solo-sc",
    )
    args = parser.parse_args()

    if args.version:
        version = pkg_resources.require("solo-sc")[0].version
        print(f"Current version of solo-sc is {version}")
        if args.model_json_file is None or args.data_path is None:
            print("Json or data path not give exiting solo")
            sys.exit()

    model_json_file = args.model_json_file
    data_path = args.data_path
    if args.gpu and not torch.cuda.is_available():
        args.gpu = torch.cuda.is_available()
        print("Cuda is not available, switching to cpu running!")

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.reproducible_seed is not None:
        scvi.settings.seed = args.reproducible_seed
    else:
        scvi.settings.seed = np.random.randint(10000)

    ##################################################
    # data

    # read loom/anndata
    data_ext = os.path.splitext(data_path)[-1]
    if data_ext == ".loom":
        scvi_data = read_loom(data_path)
    elif data_ext == ".h5ad":
        scvi_data = read_h5ad(data_path)
    elif os.path.isdir(data_path):
        scvi_data = read_10x_mtx(path=data_path)
        cell_umi_depth = scvi_data.X.sum(axis=1)
        fifth, ninetyfifth = np.percentile(cell_umi_depth, [5, 95])
        min_cell_umi_depth = np.min(cell_umi_depth)
        max_cell_umi_depth = np.max(cell_umi_depth)
        if fifth * 10 < ninetyfifth:
            print(
                """WARNING YOUR DATA HAS A WIDE RANGE OF CELL DEPTHS.
            PLEASE MANUALLY REVIEW YOUR DATA"""
            )
        print(
            f"Min cell depth: {min_cell_umi_depth}, Max cell depth: {max_cell_umi_depth}"
        )
    else:
        msg = f"{data_path} is not a recognized format.\n"
        msg += "must be one of {h5ad, loom, 10x mtx dir}"
        raise TypeError(msg)

    num_cells, num_genes = scvi_data.X.shape

    # check for parameters
    if not os.path.exists(model_json_file):
        raise FileNotFoundError(f"{model_json_file} does not exist.")
    # read parameters
    with open(model_json_file, "r") as model_json_open:
        params = json.load(model_json_open)

    # set VAE params
    vae_params = {}
    for par in ["n_hidden", "n_latent", "n_layers", "dropout_rate", "ignore_batch"]:
        if par in params:
            vae_params[par] = params[par]

    # training parameters
    batch_key = params.get("batch_key", None)
    batch_size = params.get("batch_size", 128)
    valid_pct = params.get("valid_pct", 0.1)
    learning_rate = params.get("learning_rate", 1e-3)
    stopping_params = {"patience": params.get("patience", 8), "min_delta": 0}

    # protect against single example batch
    while num_cells % batch_size == 1:
        batch_size = int(np.round(1.25 * batch_size))
        print("Increasing batch_size to %d to avoid single example batch." % batch_size)

    scvi.settings.batch_size = batch_size
    ##################################################
    # SCVI
    setup_anndata(scvi_data, batch_key=batch_key)
    vae = SCVI(
        scvi_data,
        gene_likelihood="nb",
        log_variational=True,
        **vae_params,
        use_observed_lib_size=False,
    )

    if args.seed:
        vae = vae.load(os.path.join(args.seed, "vae"), use_gpu=args.gpu)
    else:
        scvi_callbacks = []
        scvi_callbacks += [
            EarlyStopping(
                monitor="reconstruction_loss_validation", mode="min", **stopping_params
            )
        ]
        plan_kwargs = {
            "reduce_lr_on_plateau": True,
            "lr_factor": 0.1,
            "lr": 1e-2,
            "lr_patience": 10,
            "lr_threshold": 0,
            "lr_min": 1e-4,
            "lr_scheduler_metric": "reconstruction_loss_validation",
        }

        vae.train(
            max_epochs=2000,
            validation_size=valid_pct,
            check_val_every_n_epoch=5,
            plan_kwargs=plan_kwargs,
            callbacks=scvi_callbacks,
        )
        # save VAE
        vae.save(os.path.join(args.out_dir, "vae"))

    latent = vae.get_latent_representation()
    # save latent representation
    np.save(os.path.join(args.out_dir, "latent.npy"), latent.astype("float32"))

    ##################################################
    # classifier

    # model
    # todo add doublet ratio
    solo = SOLO.from_scvi_model(vae, doublet_ratio=args.doublet_ratio)
    solo.train(
        2000,
        lr=learning_rate,
        train_size=0.9,
        check_val_every_n_epoch=5,
        early_stopping_patience=6,
    )
    solo.train(
        2000,
        lr=learning_rate * 0.1,
        train_size=0.9,
        check_val_every_n_epoch=1,
        early_stopping_patience=30,
        callbacks=[],
    )
    solo.save(os.path.join(args.out_dir, "classifier"))

    logit_predictions = solo.predict(include_simulated_doublets=True)

    is_doublet_known = solo.adata.obs._solo_doub_sim == "doublet"
    is_doublet_pred = logit_predictions.idxmin(axis=1) == "singlet"

    validation_is_doublet_known = is_doublet_known[solo.validation_indices]
    validation_is_doublet_pred = is_doublet_pred[solo.validation_indices]
    training_is_doublet_known = is_doublet_known[solo.train_indices]
    training_is_doublet_pred = is_doublet_pred[solo.train_indices]

    valid_as = accuracy_score(validation_is_doublet_known, validation_is_doublet_pred)
    valid_roc = roc_auc_score(validation_is_doublet_known, validation_is_doublet_pred)
    valid_ap = average_precision_score(
        validation_is_doublet_known, validation_is_doublet_pred
    )

    train_as = accuracy_score(training_is_doublet_known, training_is_doublet_pred)
    train_roc = roc_auc_score(training_is_doublet_known, training_is_doublet_pred)
    train_ap = average_precision_score(
        training_is_doublet_known, training_is_doublet_pred
    )

    print(f"Training results")
    print(f"AUROC: {train_roc}, Accuracy: {train_as}, Average precision: {train_ap}")

    print(f"Validation results")
    print(f"AUROC: {valid_roc}, Accuracy: {valid_as}, Average precision: {valid_ap}")

    # write predictions
    # softmax predictions
    softmax_predictions = softmax(logit_predictions, axis=1)
    doublet_score = softmax_predictions.loc[:, "doublet"]

    np.save(
        os.path.join(args.out_dir, "no_updates_softmax_scores.npy"),
        doublet_score[:num_cells],
    )
    np.savetxt(
        os.path.join(args.out_dir, "no_updates_softmax_scores.csv"),
        doublet_score[:num_cells],
        delimiter=",",
    )
    np.save(
        os.path.join(args.out_dir, "no_updates_softmax_scores_sim.npy"),
        doublet_score[num_cells:],
    )

    # logit predictions
    logit_doublet_score = logit_predictions.loc[:, "doublet"]
    np.save(
        os.path.join(args.out_dir, "logit_scores.npy"), logit_doublet_score[:num_cells]
    )
    np.savetxt(
        os.path.join(args.out_dir, "logit_scores.csv"),
        logit_doublet_score[:num_cells],
        delimiter=",",
    )
    np.save(
        os.path.join(args.out_dir, "logit_scores_sim.npy"),
        logit_doublet_score[num_cells:],
    )

    # update threshold as a function of Solo's estimate of the number of
    # doublets
    # essentially a log odds update
    # TODO put in a function
    # currently overshrinking softmaxes
    diff = np.inf
    counter_update = 0
    solo_scores = doublet_score[:num_cells]
    logit_scores = logit_doublet_score[:num_cells]
    d_s = args.doublet_ratio / (args.doublet_ratio + 1)
    if args.recalibrate_scores:
        while (diff > 0.01) | (counter_update < 5):

            # calculate log odds calibration for logits
            d_o = np.mean(solo_scores)
            c = np.log(d_o / (1 - d_o)) - np.log(d_s / (1 - d_s))

            # update solo scores
            solo_scores = 1 / (1 + np.exp(-(logit_scores + c)))

            # update while conditions
            diff = np.abs(d_o - np.mean(solo_scores))
            counter_update += 1

    np.save(os.path.join(args.out_dir, "softmax_scores.npy"), solo_scores)
    np.savetxt(
        os.path.join(args.out_dir, "softmax_scores.csv"), solo_scores, delimiter=","
    )

    if args.expected_number_of_doublets is not None:
        k = len(solo_scores) - args.expected_number_of_doublets
        if args.expected_number_of_doublets / len(solo_scores) > 0.5:
            print(
                """Make sure you actually expect more than half your cells
                   to be doublets. If not change your
                   -e parameter value"""
            )
        assert k > 0
        idx = np.argpartition(solo_scores, k)
        threshold = np.max(solo_scores[idx[:k]])
        is_solo_doublet = solo_scores > threshold
    else:
        is_solo_doublet = solo_scores > 0.5

    np.save(os.path.join(args.out_dir, "is_doublet.npy"), is_solo_doublet[:num_cells])
    np.savetxt(
        os.path.join(args.out_dir, "is_doublet.csv"),
        is_solo_doublet[:num_cells],
        delimiter=",",
    )

    np.save(
        os.path.join(args.out_dir, "is_doublet_sim.npy"), is_solo_doublet[num_cells:]
    )

    np.save(os.path.join(args.out_dir, "preds.npy"), is_doublet_pred[:num_cells])
    np.savetxt(
        os.path.join(args.out_dir, "preds.csv"),
        is_doublet_pred[:num_cells],
        delimiter=",",
    )

    smoothed_preds = knn_smooth_pred_class(
        X=latent, pred_class=is_doublet_pred[:num_cells]
    )
    np.save(os.path.join(args.out_dir, "smoothed_preds.npy"), smoothed_preds)

    if args.anndata_output and data_ext == ".h5ad":
        scvi_data.obs["is_doublet"] = is_solo_doublet[:num_cells].values.astype(bool)
        scvi_data.obs["logit_scores"] = logit_doublet_score[:num_cells].values.astype(
            float
        )
        scvi_data.obs["softmax_scores"] = solo_scores[:num_cells].values.astype(float)
        scvi_data.write(os.path.join(args.out_dir, "soloed.h5ad"))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        train_solo_scores = doublet_score[solo.train_indices]
        validation_solo_scores = doublet_score[solo.validation_indices]

        train_fpr, train_tpr, _ = roc_curve(
            training_is_doublet_known, train_solo_scores
        )
        val_fpr, val_tpr, _ = roc_curve(
            validation_is_doublet_known, validation_solo_scores
        )

        # plot ROC
        plt.figure()
        plt.plot(train_fpr, train_tpr, label="Train")
        plt.plot(val_fpr, val_tpr, label="Validation")
        plt.gca().set_xlabel("False positive rate")
        plt.gca().set_ylabel("True positive rate")
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, "roc.pdf"))
        plt.close()

        train_precision, train_recall, _ = precision_recall_curve(
            training_is_doublet_known, train_solo_scores
        )
        val_precision, val_recall, _ = precision_recall_curve(
            validation_is_doublet_known, validation_solo_scores
        )
        # plot accuracy
        plt.figure()
        plt.plot(train_recall, train_precision, label="Train")
        plt.plot(val_recall, val_precision, label="Validation")
        plt.gca().set_xlabel("Recall")
        plt.gca().set_ylabel("pytPrecision")
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, "precision_recall.pdf"))
        plt.close()

        # plot distributions
        obs_indices = solo.validation_indices[solo.validation_indices < num_cells]
        sim_indices = solo.validation_indices[solo.validation_indices > num_cells]

        plt.figure()
        sns.displot(doublet_score[sim_indices], label="Simulated")
        sns.displot(doublet_score[obs_indices], label="Observed")
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, "sim_vs_obs_dist.pdf"))
        plt.close()

        plt.figure()
        sns.distplot(solo_scores[:num_cells], label="Observed (transformed)")
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, "real_cells_dist.pdf"))
        plt.close()

        scvi_umap = umap.UMAP(n_neighbors=16).fit_transform(latent)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(
            scvi_umap[:, 0],
            scvi_umap[:, 1],
            c=doublet_score[:num_cells],
            s=8,
            cmap="GnBu",
        )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.savefig(os.path.join(args.out_dir, "umap_solo_scores.pdf"))


###############################################################################
# __main__
###############################################################################


if __name__ == "__main__":
    main()
