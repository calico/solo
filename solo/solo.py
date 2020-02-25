#!/usr/bin/env python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import os
import shutil
import anndata

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.sparse import issparse
from collections import defaultdict

import scvi
from scvi.dataset import AnnDatasetFromAnnData, LoomDataset, \
    GeneExpressionDataset
from scvi.models import Classifier, VAE, SCANVI
from scvi.inference import UnsupervisedTrainer, ClassifierTrainer, JointSemiSupervisedTrainer, SemiSupervisedTrainer
import torch

from .utils import create_average_doublet, create_summed_doublet, \
    create_multinomial_doublet, make_gene_expression_dataset, \
    knn_smooth_pred_class

'''
solo.py

Simulate doublets, train a VAE, and then a classifier on top.
'''


###############################################################################
# main
###############################################################################


def main():
    usage = 'solo'
    parser = ArgumentParser(usage, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='model_json_file',
                        help='json file to pass VAE parameters')
    parser.add_argument(dest='data_file',
                        help='h5ad file containing cell by genes counts')
    parser.add_argument('-d', dest='doublet_depth',
                        default=2., type=float,
                        help='Depth multiplier for a doublet relative to the \
                        average of its constituents')
    parser.add_argument('-g', dest='gpu',
                        default=True, action='store_true',
                        help='Run on GPU')
    parser.add_argument('-a', dest='anndata_output',
                        default=False, action='store_true',
                        help='output modified anndata object with solo scores \
                        Only works for anndata')
    parser.add_argument('-o', dest='out_dir',
                        default='solo_out')
    parser.add_argument('-r', dest='doublet_ratio',
                        default=2., type=float,
                        help='Ratio of doublets to true \
                        cells')
    parser.add_argument('-s', dest='seed',
                        default=None, help='Path to previous solo output  \
                        directory. Seed VAE models with previously \
                        trained solo model. Directory structure is assumed to \
                        be the same as solo output directory structure. \
                        should at least have a vae.pt a pickled object of \
                        vae weights and a latent.npy an np.ndarray of the \
                        latents of your cells.')
    parser.add_argument('-k', dest='known_doublets',
                        help='Experimentally defined doublets tsv file. \
                        Should be a single column of True/False. True \
                        indicates the cell is a doublet. No header.',
                        type=str)
    parser.add_argument('-t', dest='doublet_type', help='Please enter \
                        multinomial, average, or sum',
                        default='multinomial',
                        choices=['multinomial', 'average', 'sum'])
    parser.add_argument('-e', dest='expected_number_of_doublets',
                        help='Experimentally expected number of doublets',
                        type=int, default=None)
    parser.add_argument('-p', dest='plot',
                        default=False, action='store_true',
                        help='Plot outputs for solo')
    parser.add_argument('-l', dest='normal_logging',
                        default=False, action='store_true',
                        help='Logging level set to normal (aka not debug)')
    parser.add_argument('--random-size', dest='randomize_doublet_size',
                        default=False,
                        action='store_true',
                        help='Sample depth multipliers from Unif(1, \
                        DoubletDepth) \
                        to provide a diversity of possible doublet depths.'
                        )
    parser.add_argument('--no-fix-vae-wieghts', dest='fix_vae_wieghts',
                        default=True,
                        action='store_false',
                        help='allow vae weights to change during classification \
                        Use Scanc.'
                        )
    args = parser.parse_args()

    if not args.normal_logging:
        scvi._settings.set_verbosity(10)

    model_json_file = args.model_json_file
    data_file = args.data_file
    if args.gpu and not torch.cuda.is_available():
        args.gpu = torch.cuda.is_available()
        print('Cuda is not available, switching to cpu running!')

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    ##################################################
    # data

    # read loom/anndata
    data_ext = os.path.splitext(data_file)[-1]
    if data_ext == '.loom':
        scvi_data = LoomDataset(data_file)
    elif data_ext == '.h5ad':
        adata = anndata.read(data_file)
        scvi_data = AnnDatasetFromAnnData(adata)
    else:
        msg = f'{data_ext} is not a recognized format.\n'
        msg += 'must be one of {h5ad, loom}'
        raise TypeError(msg)

    if issparse(scvi_data.X):
        scvi_data.X = scvi_data.X.todense()
    num_cells, num_genes = scvi_data.X.shape

    if args.known_doublets is not None:
        print('Removing known doublets for in silico doublet generation')
        print('Make sure known doublets are in the same order as your data')
        known_doublets = np.loadtxt(args.known_doublets, dtype=str) == 'True'

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

    # check for parameters
    if not os.path.exists(model_json_file):
        raise FileNotFoundError(f'{model_json_file} does not exist.')
    # read parameters
    with open(model_json_file, 'r') as model_json_open:
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

    if args.seed:
        if args.gpu:
            device = torch.device('cuda')
            vae.load_state_dict(torch.load(os.path.join(args.seed, 'vae.pt')))
            vae.to(device)
        else:
            map_loc = 'cpu'
            vae.load_state_dict(torch.load(os.path.join(args.seed, 'vae.pt'),
                                           map_location=map_loc))

        # copy latent representation
        latent_file = os.path.join(args.seed, 'latent.npy')
        if os.path.isfile(latent_file):
            shutil.copy(latent_file, os.path.join(args.out_dir, 'latent.npy'))

    else:
        stopping_params['early_stopping_metric'] = 'reconstruction_error'
        stopping_params['save_best_state_metric'] = 'reconstruction_error'

        # initialize unsupervised trainer
        utrainer = \
            UnsupervisedTrainer(vae, singlet_scvi_data,
                                train_size=(1. - valid_pct),
                                frequency=2,
                                metrics_to_monitor=['reconstruction_error'],
                                use_cuda=args.gpu,
                                early_stopping_kwargs=stopping_params)
        utrainer.history['reconstruction_error_test_set'].append(0)
        # initial epoch
        utrainer.train(n_epochs=2000, lr=learning_rate)

        # drop learning rate and continue
        utrainer.early_stopping.wait = 0
        utrainer.train(n_epochs=500, lr=0.5 * learning_rate)

        # save VAE
        torch.save(vae.state_dict(), os.path.join(args.out_dir, 'vae.pt'))

        # save latent representation
        full_posterior = utrainer.create_posterior(
            utrainer.model,
            singlet_scvi_data,
            indices=np.arange(len(singlet_scvi_data)))
        latent, _, _ = full_posterior.sequential().get_latent()
        np.save(os.path.join(args.out_dir, 'latent.npy'),
                latent.astype('float32'))

    ##################################################
    # simulate doublets

    non_zero_indexes = np.where(singlet_scvi_data.X > 0)
    cells = non_zero_indexes[0]
    genes = non_zero_indexes[1]
    cells_ids = defaultdict(list)
    for cell_id, gene in zip(cells, genes):
        cells_ids[cell_id].append(gene)

    # choose doublets function type
    if args.doublet_type == 'average':
        doublet_function = create_average_doublet
    elif args.doublet_type == 'sum':
        doublet_function = create_summed_doublet
    else:
        doublet_function = create_multinomial_doublet

    cell_depths = singlet_scvi_data.X.sum(axis=1)
    num_doublets = int(args.doublet_ratio * singlet_num_cells)
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
                             doublet_depth=args.doublet_depth,
                             cell_depths=cell_depths, cells_ids=cells_ids,
                             randomize_doublet_size=args.randomize_doublet_size)

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
    classifier_data.n_labels = 2
    cls_params = {'n_hidden': params['cl_hidden'],
                  'n_layers': params['cl_layers'],
                  'dropout_rate': params['dropout_rate']}
    scanvi = SCANVI(classifier_data.nb_genes, n_labels=classifier_data.n_labels, classifier_parameters=cls_params,  **vae_params)
    # for k, v in vae._modules.items():
    #     if scanvi._modules.get(k) is not None:
    #         scanvi._modules[k] = v
    scanvi.load_state_dict(vae.state_dict(), strict=False)

    trainer_scanvi = SemiSupervisedTrainer(scanvi, classifier_data,
                                           early_stopping_kwargs=stopping_params,
                                           metrics_to_monitor=['reconstruction_error', 'accuracy'])
    trainer_scanvi.labelled_set = trainer_scanvi.create_posterior(indices=(classifier_data.batch_indices == 0).ravel())
    trainer_scanvi.labelled_set.to_monitor = ['reconstruction_error', 'accuracy']
    trainer_scanvi.unlabelled_set = trainer_scanvi.create_posterior(indices=(classifier_data.batch_indices == 1).ravel())
    trainer_scanvi.unlabelled_set.to_monitor = ['reconstruction_error', 'accuracy']
    # initial
    trainer_scanvi.train(n_epochs=1000, lr=learning_rate)

    # drop learning rate and continue
    trainer_scanvi.early_stopping.wait = 0
    trainer_scanvi.train(n_epochs=300, lr=0.1 * learning_rate)
    strainer = trainer_scanvi.labelled_set

    # models evaluation mode
    vae.eval()
    scanvi.eval()

    # write predictions
    # softmax predictions
    order_y, order_score = strainer.compute_predictions(soft=True)
    _, order_pred = strainer.compute_predictions()
    doublet_score = order_score[:, 1]


    np.save(os.path.join(args.out_dir, 'softmax_scores.npy'), doublet_score[:num_cells])
    np.save(os.path.join(args.out_dir, 'softmax_scores_sim.npy'), doublet_score[num_cells:])

    if args.expected_number_of_doublets is not None:
        solo_scores = doublet_score[:num_cells]
        k = len(solo_scores) - args.expected_number_of_doublets
        if args.expected_number_of_doublets / len(solo_scores) > .5:
            print('''Make sure you actually expect more than half your cells
                   to be doublets. If not change your
                   -e parameter value''')
        assert k > 0
        idx = np.argpartition(solo_scores, k)
        threshold = np.max(solo_scores[idx[:k]])
        is_solo_doublet = doublet_score > threshold
    else:
        is_solo_doublet = order_pred[:num_cells]

    is_doublet = known_doublets
    new_doublets_idx = np.where(~(is_doublet) & is_solo_doublet[:num_cells])[0]
    is_doublet[new_doublets_idx] = True

    np.save(os.path.join(args.out_dir, 'is_doublet.npy'), is_doublet[:num_cells])
    np.save(os.path.join(args.out_dir, 'is_doublet_sim.npy'), is_doublet[num_cells:])

    np.save(os.path.join(args.out_dir, 'preds.npy'), order_pred[:num_cells])
    np.save(os.path.join(args.out_dir, 'preds_sim.npy'), order_pred[num_cells:])

    smoothed_preds = knn_smooth_pred_class(X=latent, pred_class=is_doublet[:num_cells])
    np.save(os.path.join(args.out_dir, 'smoothed_preds.npy'), smoothed_preds)

    if args.anndata_output and data_ext == '.h5ad':
        adata.obs['is_doublet'] = is_doublet[:num_cells]
        adata.obs['softmax_scores'] = doublet_score[:num_cells]
        adata.write(os.path.join(args.out_dir, "soloed.h5ad"))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        # plot ROC
        plt.figure()
        plt.plot(train_fpr, train_tpr, label='Train')
        plt.plot(test_fpr, test_tpr, label='Test')
        plt.gca().set_xlabel('False positive rate')
        plt.gca().set_ylabel('True positive rate')
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, 'roc.pdf'))
        plt.close()

        # plot accuracy
        plt.figure()
        plt.plot(train_t, train_acc, label='Train')
        plt.plot(test_t, test_acc, label='Test')
        plt.axvline(0.5, color='black', linestyle='--')
        plt.gca().set_xlabel('Threshold')
        plt.gca().set_ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, 'accuracy.pdf'))
        plt.close()

        # plot distributions
        plt.figure()
        sns.distplot(test_score[test_y], label='Simulated')
        sns.distplot(test_score[~test_y], label='Observed')
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, 'train_v_test_dist.pdf'))
        plt.close()

        plt.figure()
        sns.distplot(doublet_score[:num_cells], label='Simulated')
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, 'real_cells_dist.pdf'))
        plt.close()
###############################################################################
# __main__
###############################################################################


if __name__ == '__main__':
    main()
