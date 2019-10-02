#!/usr/bin/env python
from optparse import OptionParser
import json
import os
import shutil
import anndata

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.sparse import issparse
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from scvi.dataset import AnnDatasetFromAnnData, LoomDataset, \
                         GeneExpressionDataset
from scvi.models import Classifier, VAE
from scvi.inference import UnsupervisedTrainer, ClassifierTrainer
import torch

from .utils import create_average_doublet, create_summed_doublet, \
    create_multinomial_doublet, make_gene_expression_dataset

'''
solo.py

Simulate doublets, train a VAE, and then a classifier on top.
'''


###############################################################################
# main
###############################################################################


def main():
    usage = 'usage: %prog [options] <model_json> <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='doublet_depth',
                      default=2., type='float',
                      help='Depth multiplier for a doublet relative to the \
                      average of its constituents [Default: %default]')
    parser.add_option('-g', dest='gpu',
                      default=False, action='store_true',
                      help='Run on GPU [Default: %default]')
    parser.add_option('-o', dest='out_dir',
                      default='solo_out')
    parser.add_option('-r', dest='doublet_ratio',
                      default=2., type='float',
                      help='Ratio of doublets to true \
                      cells [Default: %default]')
    parser.add_option('-s', dest='seed',
                      default=None, help='Seed VAE model parameters')
    parser.add_option('-k', dest='known_doublets',
                      help='Experimentally defined doublets tsv file',
                      type=str)
    parser.add_option('-t', dest='doublet_type', help="Please enter \
                      multinomial, average, or sum [Default: %default]",
                      default="multinomial",
                      choices=['multinomial', 'average', 'sum'])
    parser.add_option('-e', dest='expected_number_of_doublets',
                      help='Experimentally expected number of doublets',
                      type=int, default=None)
    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide model json and data loom')
    else:
        model_json_file = args[0]
        data_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    ##################################################
    # data

    # read loom/anndata
    data_ext = os.path.splitext(data_file)[-1]
    if data_ext == '.loom':
        scvi_data = LoomDataset(data_file)
    elif data_ext == '.h5ad':
        scvi_data = AnnDatasetFromAnnData(anndata.read(data_file))
    else:
        print('Unrecognized file format')
    if issparse(scvi_data.X):
        scvi_data.X = scvi_data.X.todense()
    num_cells, num_genes = scvi_data.X.shape

    if options.known_doublets is not None:
        print("Removing known doublets for in silico doublet generation")
        print("Make sure known doublets are in the same order as your data")
        known_doublets = pd.read_csv(options.known_doublets,
                                     header=None)[0].values
        assert len(known_doublets) == scvi_data.X.shape[0]
        known_doublet_data = make_gene_expression_dataset(
                                    scvi_data.X[known_doublets],
                                    scvi_data.gene_names)
        known_doublet_data.labels = np.ones(known_doublet_data.X.shape[0])
        singlet_scvi_data = make_gene_expression_dataset(
                                                 scvi_data.X[~known_doublets],
                                                 scvi_data.gene_names)
        singlet_num_cells, _ = singlet_scvi_data.X.shape
    else:
        known_doublet_data = None
        singlet_num_cells = num_cells
        known_doublets = np.zeros(num_cells, dtype=bool)
        singlet_scvi_data = scvi_data
    singlet_scvi_data.labels = np.zeros(singlet_scvi_data.X.shape[0])
    scvi_data.labels = known_doublets.astype(int)
    ##################################################
    # parameters

    # read parameters
    with open(model_json_file) as model_json_open:
        params = json.load(model_json_open)

    # set VAE params
    vae_params = {}
    for par in ['n_hidden', 'n_latent', 'n_layers', 'dropout_rate',
                'ignore_batch']:
        if par in params:
            vae_params[par] = params[par]
    vae_params['n_batch'] = 0 if params.get(
        'ignore_batch', False) else scvi_data.n_batches

    # training parameters
    valid_pct = params.get('valid_pct', 0.1)
    learning_rate = params.get('learning_rate', 1e-3)
    stopping_params = {'patience': params.get('patience', 10), 'threshold': 0}

    ##################################################
    # VAE

    vae = VAE(n_input=singlet_scvi_data.nb_genes, n_labels=2,
              reconstruction_loss='nb',
              log_variational=True, **vae_params)

    if options.seed:
        if options.gpu:
            map_loc = None
        else:
            map_loc = 'cpu'
        vae.load_state_dict(torch.load(options.seed, map_location=map_loc))
        if options.gpu:
            vae.cuda()

        # copy latent representation
        latent_file = '%s/latent.npy' % os.path.split(options.seed)[0]
        if os.path.isfile(latent_file):
            shutil.copy(latent_file, '%s/latent.npy' % options.out_dir)

    else:
        stopping_params['early_stopping_metric'] = 'reconstruction_error'
        stopping_params['save_best_state_metric'] = 'reconstruction_error'

        # initialize unsupervised trainer
        utrainer = \
            UnsupervisedTrainer(vae, singlet_scvi_data,
                                train_size=(1. - valid_pct),
                                frequency=2,
                                metrics_to_monitor=['reconstruction_error'],
                                use_cuda=options.gpu,
                                early_stopping_kwargs=stopping_params)

        # initial epoch
        utrainer.train(n_epochs=2000, lr=learning_rate)

        # drop learning rate and continue
        utrainer.early_stopping.wait = 0
        utrainer.train(n_epochs=500, lr=0.1 * learning_rate)

        # save VAE
        torch.save(vae.state_dict(), '%s/vae.pt' % options.out_dir)

        # save latent representation
        full_posterior = utrainer.create_posterior(
                                    utrainer.model,
                                    singlet_scvi_data,
                                    indices=np.arange(len(singlet_scvi_data)))
        latent, _, _ = full_posterior.sequential().get_latent()
        np.save('%s/latent.npy' % options.out_dir, latent.astype('float32'))


    ##################################################
    # simulate doublets

    non_zero_indexes = np.where(singlet_scvi_data.X > 0)
    cells = non_zero_indexes[0]
    genes = non_zero_indexes[1]
    cells_ids = defaultdict(list)
    for cell_id, gene in zip(cells, genes):
        cells_ids[cell_id].append(gene)

    # choose doublets function type
    if options.doublet_type == "average":
        doublet_function = create_average_doublet
    elif options.doublet_type == "sum":
        doublet_function = create_summed_doublet
    else:
        doublet_function = create_multinomial_doublet

    cell_depths = singlet_scvi_data.X.sum(axis=1)
    num_doublets = int(options.doublet_ratio * singlet_num_cells)
    if known_doublet_data is not None:
        num_doublets -= known_doublet_data.X.shape[0]
        # make sure we are making a non negative amount of doublets
        assert num_doublets >= 0

    in_silico_doublets = np.zeros((num_doublets, num_genes), dtype='float32')
    # for desired # doublets
    for di in range(num_doublets):
        # sample two cells
        i, j = np.random.choice(singlet_num_cells, size=2)

        # generate doublets
        in_silico_doublets[di, :] = \
            doublet_function(singlet_scvi_data.X, i, j,
                             doublet_depth=options.doublet_depth,
                             cell_depths=cell_depths, cells_ids=cells_ids)


    # merge datasets
    # we can maybe up sample the known doublets
    # concatentate
    classifier_data = GeneExpressionDataset()
    classifier_data.populate_from_data(
                    X=np.vstack([scvi_data.X,
                                 in_silico_doublets]),
                    labels=np.hstack([np.ravel(scvi_data.labels),
                                      np.ones(in_silico_doublets.shape[0])]),
                    remap_attributes=False)

    assert(len(np.unique(classifier_data.labels.flatten())) == 2)

    ##################################################
    # classifier

    # model
    classifier = Classifier(n_input=(vae.n_latent + 1),
                            n_hidden=params['cl_hidden'],
                            n_layers=params['cl_layers'], n_labels=2,
                            dropout_rate=params['dropout_rate'])

    # trainer
    stopping_params['early_stopping_metric'] = 'accuracy'
    stopping_params['save_best_state_metric'] = 'accuracy'
    strainer = ClassifierTrainer(classifier, classifier_data,
                                 train_size=(1. - valid_pct),
                                 frequency=2, metrics_to_monitor=['accuracy'],
                                 use_cuda=options.gpu,
                                 sampling_model=vae, sampling_zl=True,
                                 early_stopping_kwargs=stopping_params)

    # initial
    strainer.train(n_epochs=1000, lr=learning_rate)

    # drop learning rate and continue
    strainer.early_stopping.wait = 0
    strainer.train(n_epochs=300, lr=0.1 * learning_rate)
    torch.save(classifier.state_dict(), '%s/classifier.pt' % options.out_dir)

    ##################################################
    # post-processing

    # models evaluation mode
    vae.eval()
    classifier.eval()

    print('Train accuracy: %.4f' % strainer.train_set.accuracy())
    print('Test accuracy:  %.4f' % strainer.test_set.accuracy())

    # compute my own predictions
    train_y, train_score = strainer.train_set.compute_predictions(soft=True)
    test_y, test_score = strainer.test_set.compute_predictions(soft=True)
    train_score = train_score[:, 1]
    train_y = train_y.astype('bool')
    test_score = test_score[:, 1]
    test_y = test_y.astype('bool')

    train_auroc = roc_auc_score(train_y, train_score)
    test_auroc = roc_auc_score(test_y, test_score)

    print('Train AUROC: %.4f' % train_auroc)
    print('Test AUROC:  %.4f' % test_auroc)

    # plot ROC
    train_fpr, train_tpr, train_t = roc_curve(train_y, train_score)
    test_fpr, test_tpr, test_t = roc_curve(test_y, test_score)
    train_t = np.minimum(train_t, 1 + 1e-9)
    test_t = np.minimum(test_t, 1 + 1e-9)
    plt.figure()
    plt.plot(train_fpr, train_tpr, label='Train')
    plt.plot(test_fpr, test_tpr, label='Test')
    plt.gca().set_xlabel('False positive rate')
    plt.gca().set_ylabel('True positive rate')
    plt.legend()
    plt.savefig('%s/roc.pdf' % options.out_dir)
    plt.close()

    # plot accuracy
    train_acc = np.zeros(len(train_t))
    for i in range(len(train_t)):
        train_acc[i] = np.mean(train_y == (train_score > train_t[i]))
    test_acc = np.zeros(len(test_t))
    for i in range(len(test_t)):
        test_acc[i] = np.mean(test_y == (test_score > test_t[i]))
    plt.figure()
    plt.plot(train_t, train_acc, label='Train')
    plt.plot(test_t, test_acc, label='Test')
    plt.axvline(0.5, color='black', linestyle='--')
    plt.gca().set_xlabel('Threshold')
    plt.gca().set_ylabel('Accuracy')
    plt.legend()
    plt.savefig('%s/accuracy.pdf' % options.out_dir)
    plt.close()

    # write predictions
    order_y, order_score = strainer.compute_predictions(soft=True)
    order_score = order_score[:, 1]
    np.save('%s/scores.npy' % options.out_dir, order_score[:num_cells])
    np.save('%s/scores_sim.npy' % options.out_dir, order_score[num_cells:])

    ## TODO: figure out this function
    if options.expected_number_of_doublets is not None:
        solo_scores = order_score[:num_cells]
        k = len(solo_scores) - options.expected_number_of_doublets
        if options.expected_number_of_doublets / len(solo_scores) > .5:
            print("""Make sure you actually expect more than half your cells
                   to be doublets. If not change your
                   -e parameter value""")
        assert k > 0
        idx = np.argpartition(solo_scores, k)
        threshold = np.max(solo_scores[idx[:k]])
    else:
        threshold = .5
    is_solo_doublet = order_score > threshold

    # plot distributions
    plt.figure()
    sns.distplot(test_score[test_y], label='Simulated')
    sns.distplot(test_score[~test_y], label='Observed')
    plt.axvline(x=threshold)
    plt.legend()
    plt.savefig('%s/train_v_test_dist.pdf' % options.out_dir)
    plt.close()

    plt.figure()
    sns.distplot(order_score[:num_cells], label='Simulated')
    plt.axvline(x=threshold)
    plt.legend()
    plt.savefig('%s/real_cells_dist.pdf' % options.out_dir)
    plt.close()

    is_doublet = known_doublets
    new_doublets_idx = np.where(~(is_doublet) & is_solo_doublet[:num_cells])[0]
    is_doublet[new_doublets_idx] = True

    np.save('%s/is_doublet.npy' % options.out_dir, is_doublet[:num_cells])
    np.save('%s/is_doublet_sim.npy' % options.out_dir, is_doublet[num_cells:])

    _, order_pred = strainer.compute_predictions()
    np.save('%s/preds.npy' % options.out_dir, order_pred[:num_cells])
    np.save('%s/preds_sim.npy' % options.out_dir, order_pred[num_cells:])


###############################################################################
# __main__
###############################################################################


if __name__ == '__main__':
    main()
