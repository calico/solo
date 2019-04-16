#!/usr/bin/env python
from optparse import OptionParser
import json
import os
import shutil

import numpy as np
import pandas as pd
from scipy.stats import multinomial
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns

from scvi.dataset import AnnDataset, LoomDataset, GeneExpressionDataset
from scvi.models import Classifier, VAE
from scvi.inference import UnsupervisedTrainer, ClassifierTrainer
import torch

'''
loner.py

Simulate doublets, train a VAE, and then a classifier on top.
'''


def read_tsv(path):
    return pd.read_csv(path, header=None)


def make_gene_expression_dataset(data, gene_names):
    means, var = GeneExpressionDataset.library_size(data)
    data_length = data.X.shape[0]
    batch = np.zeros((data_length, 1), dtype='uint32')
    labels = np.ones((data_length, 1), dtype='uint32')
    return GeneExpressionDataset(data, local_means=means, local_vars=var,
                                 batch_indices=batch, labels=labels,
                                 gene_names=gene_names)
###############################################################################
# main
###############################################################################


def main():
    usage = 'usage: %prog [options] <model_json> <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='doublet_depth',
                      default=1., type='float',
                      help='Depth multiplier for a doublet relative to the \
                      average of its constituents [Default: % default]')
    parser.add_option('-g', dest='gpu',
                      default=False, action='store_true',
                      help='Run on GPU [Default: %default]')
    parser.add_option('-o', dest='out_dir',
                      default='loner_out')
    parser.add_option('-r', dest='doublet_ratio',
                      default=1., type='float',
                      help='Ratio of doublets to true \
                      cells [Default: %default]')
    parser.add_option('-s', dest='seed',
                      default=None, help='Seed VAE model parameters')
    parser.add_option('-k', dest='known_doublets',
                      help='Experimentally defined doublets tsv file',
                      type=read_tsv)
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
        scvi_data = LoomDataset(data_file, save_path='./')
    elif data_ext == '.h5ad':
        scvi_data = AnnDataset(data_file)
    else:
        print('Unrecognized file format')

    if options.known_doublets is not None:
        print("Removing known doublets for in silico doublet generation")
        print("Make sure known doublets are in the same order as your data")
        known_doublets = options.known_doublets[0].values
        assert len(known_doublets) == scvi_data.X.shape[0]
        known_doublet_data = make_gene_expression_dataset(
                                    scvi_data.X[known_doublets],
                                    scvi_data.gene_names)
        scvi_data = make_gene_expression_dataset(scvi_data.X[~known_doublets],
                                                 scvi_data.gene_names)
    else:
        known_doublet_data = None

    num_cells, num_genes = scvi_data.X.shape

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

    vae = VAE(n_input=scvi_data.nb_genes, n_labels=2,
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
        stopping_params['early_stopping_metric'] = 'll'
        stopping_params['save_best_state_metric'] = 'll'

        # initialize unsupervised trainer
        utrainer = UnsupervisedTrainer(vae, scvi_data,
                                       train_size=(1. - valid_pct),
                                       frequency=2, metrics_to_monitor='ll',
                                       use_cuda=options.gpu, verbose=True,
                                       early_stopping_kwargs=stopping_params)

        # initial epoch
        utrainer.train(n_epochs=2000, lr=learning_rate)

        # drop learning rate and continue
        utrainer.early_stopping.wait = 0
        utrainer.train(n_epochs=500, lr=0.1 * learning_rate)

        # save VAE
        torch.save(vae.state_dict(), '%s/vae.pt' % options.out_dir)

        # save latent representation
        full_posterior = utrainer.create_posterior(utrainer.model, scvi_data,
                                                   indices=np.arange(len(scvi_data)))
        latent, _, _ = full_posterior.sequential().get_latent()
        np.save('%s/latent.npy' % options.out_dir, latent.astype('float32'))


    ##################################################
    # simulate doublets

    cell_depths = scvi_data.X.sum(axis=1)
    num_doublets = int(options.doublet_ratio * num_cells)

    if known_doublet_data is not None:
        num_doublets -= known_doublet_data.X.shape[0]
        # make sure we are making a non negative amount of doublets
        assert num_doublets >= 0
    X_doublets = np.zeros((num_doublets, num_genes), dtype='float32')
    # for desired # doublets
    for di in range(num_doublets):
        # sample two cells
        i, j = np.random.choice(num_cells, size=2)

        # add their counts
        dp = (scvi_data.X[i, :] + scvi_data.X[j, :]).astype('float64')

        # normalize
        dp /= dp.sum()

        # choose depth
        dd = int(options.doublet_depth * (cell_depths[i] + cell_depths[j]) / 2)

        # sample counts from multinomial
        X_doublets[di, :] = multinomial.rvs(n=dd, p=dp)

    # merge datasets
    # we can maybe up sample the known doublets
    if known_doublet_data is not None:
        X_doublets = np.vstack([known_doublet_data.X, X_doublets])

    doublet_data = make_gene_expression_dataset(X_doublets,
                                                scvi_data.gene_names)
    # manually set labels to 1
    doublet_data.labels += 1
    doublet_data.n_labels = 2
    scvi_data.n_labels = 2

    # concatentate
    scvi_data = GeneExpressionDataset.concat_datasets(scvi_data, doublet_data,
                                                      shared_labels=True,
                                                      shared_batches=True)
    assert(len(np.unique(scvi_data.labels.flatten())) == 2)

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
    strainer = ClassifierTrainer(classifier, scvi_data,
                                 train_size=(1. - valid_pct),
                                 frequency=2, metrics_to_monitor=['accuracy'],
                                 use_cuda=options.gpu, verbose=True,
                                 sampling_model=vae, sampling_zl=True,
                                 early_stopping_kwargs=stopping_params)

    # initial
    strainer.train(n_epochs=1000, lr=learning_rate)

    # drop learning rate and continue
    strainer.early_stopping.wait = 0
    strainer.train(n_epochs=300, lr=0.1 * learning_rate)

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

    # plot distributions
    plt.figure()
    sns.distplot(test_score[test_y], label='Simulated')
    sns.distplot(test_score[~test_y], label='Observed')
    plt.legend()
    plt.savefig('%s/dist.pdf' % options.out_dir)
    plt.close()

    # write predictions
    order_y, order_score = strainer.compute_predictions(soft=True)
    order_score = order_score[:, 1]
    np.save('%s/scores.npy' % options.out_dir, order_score[:num_cells])
    np.save('%s/scores_sim.npy' % options.out_dir, order_score[num_cells:])

    _, order_pred = strainer.compute_predictions()
    np.save('%s/preds.npy' % options.out_dir, order_pred[:num_cells])
    np.save('%s/preds_sim.npy' % options.out_dir, order_pred[num_cells:])


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

