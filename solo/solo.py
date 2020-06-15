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
    GeneExpressionDataset, Dataset10X
from scvi.models import Classifier, VAE
from scvi.inference import UnsupervisedTrainer, ClassifierTrainer
import torch
import umap

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
    parser.add_argument(dest='data_path',
                        help='path to h5ad, loom or 10x directory containing cell by genes counts')
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
    parser.add_argument('--random_size', dest='randomize_doublet_size',
                        default=False,
                        action='store_true',
                        help='Sample depth multipliers from Unif(1, \
                        DoubletDepth) \
                        to provide a diversity of possible doublet depths.'
                        )
    args = parser.parse_args()

    if not args.normal_logging:
        scvi._settings.set_verbosity(10)

    model_json_file = args.model_json_file
    data_path = args.data_path
    if args.gpu and not torch.cuda.is_available():
        args.gpu = torch.cuda.is_available()
        print('Cuda is not available, switching to cpu running!')

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    ##################################################
    # data

    # read loom/anndata
    data_ext = os.path.splitext(data_path)[-1]
    if data_ext == '.loom':
        scvi_data = LoomDataset(data_path)
    elif data_ext == '.h5ad':
        adata = anndata.read(data_path)
        if issparse(adata.X):
            adata.X = adata.X.todense()
        scvi_data = AnnDatasetFromAnnData(adata)
    elif os.path.isdir(data_path):
        scvi_data = Dataset10X(save_path=data_path,
                               measurement_names_column=1,
                               dense=True)
        cell_umi_depth = scvi_data.X.sum(axis=1)
        fifth, ninetyfifth = np.percentile(cell_umi_depth, [5, 95])
        min_cell_umi_depth = np.min(cell_umi_depth)
        max_cell_umi_depth = np.max(cell_umi_depth)
        if fifth * 10 < ninetyfifth:
            print("""WARNING YOUR DATA HAS A WIDE RANGE OF CELL DEPTHS.
            PLEASE MANUALLY REVIEW YOUR DATA""")
        print(f"Min cell depth: {min_cell_umi_depth}, Max cell depth: {max_cell_umi_depth}")
    else:
        msg = f'{data_path} is not a recognized format.\n'
        msg += 'must be one of {h5ad, loom, 10x directory}'
        raise TypeError(msg)

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
    batch_size = params.get('batch_size', 128)
    valid_pct = params.get('valid_pct', 0.1)
    learning_rate = params.get('learning_rate', 1e-3)
    stopping_params = {'patience': params.get('patience', 10), 'threshold': 0}

    # protect against single example batch
    while num_cells % batch_size == 1:
        batch_size = int(np.round(1.25*batch_size))
        print('Increasing batch_size to %d to avoid single example batch.' % batch_size)

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

        # save latent representation
        utrainer = \
            UnsupervisedTrainer(vae, singlet_scvi_data,
                                train_size=(1. - valid_pct),
                                frequency=2,
                                metrics_to_monitor=['reconstruction_error'],
                                use_cuda=args.gpu,
                                early_stopping_kwargs=stopping_params,
                                batch_size=batch_size)

        full_posterior = utrainer.create_posterior(
            utrainer.model,
            singlet_scvi_data,
            indices=np.arange(len(singlet_scvi_data)))
        latent, _, _ = full_posterior.sequential(batch_size).get_latent()
        np.save(os.path.join(args.out_dir, 'latent.npy'),
                latent.astype('float32'))

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
                                early_stopping_kwargs=stopping_params,
                                batch_size=batch_size)
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
        latent, _, _ = full_posterior.sequential(batch_size).get_latent()
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
                                 use_cuda=args.gpu,
                                 sampling_model=vae, sampling_zl=True,
                                 early_stopping_kwargs=stopping_params,
                                 batch_size=batch_size)

    # initial
    strainer.train(n_epochs=1000, lr=learning_rate)

    # drop learning rate and continue
    strainer.early_stopping.wait = 0
    strainer.train(n_epochs=300, lr=0.1 * learning_rate)
    torch.save(classifier.state_dict(), os.path.join(args.out_dir, 'classifier.pt'))


    ##################################################
    # post-processing
    # use logits for predictions for better results
    logits_classifier = Classifier(n_input=(vae.n_latent + 1),
                                   n_hidden=params['cl_hidden'],
                                   n_layers=params['cl_layers'], n_labels=2,
                                   dropout_rate=params['dropout_rate'],
                                   logits=True)
    logits_classifier.load_state_dict(classifier.state_dict())

    # using logits leads to better performance in for ranking
    logits_strainer = ClassifierTrainer(logits_classifier, classifier_data,
                                        train_size=(1. - valid_pct),
                                        frequency=2,
                                        metrics_to_monitor=['accuracy'],
                                        use_cuda=args.gpu,
                                        sampling_model=vae, sampling_zl=True,
                                        early_stopping_kwargs=stopping_params,
                                        batch_size=batch_size)

    # models evaluation mode
    vae.eval()
    classifier.eval()
    logits_classifier.eval()

    print('Train accuracy: %.4f' % strainer.train_set.accuracy())
    print('Test accuracy:  %.4f' % strainer.test_set.accuracy())

    # compute predictions manually
    # output logits
    train_y, train_score = strainer.train_set.compute_predictions(soft=True)
    test_y, test_score = strainer.test_set.compute_predictions(soft=True)
    # train_y == true label
    # train_score[:, 0] == singlet score; train_score[:, 1] == doublet score
    train_score = train_score[:, 1]
    train_y = train_y.astype('bool')
    test_score = test_score[:, 1]
    test_y = test_y.astype('bool')

    train_auroc = roc_auc_score(train_y, train_score)
    test_auroc = roc_auc_score(test_y, test_score)

    print('Train AUROC: %.4f' % train_auroc)
    print('Test AUROC:  %.4f' % test_auroc)

    train_fpr, train_tpr, train_t = roc_curve(train_y, train_score)
    test_fpr, test_tpr, test_t = roc_curve(test_y, test_score)
    train_t = np.minimum(train_t, 1 + 1e-9)
    test_t = np.minimum(test_t, 1 + 1e-9)

    train_acc = np.zeros(len(train_t))
    for i in range(len(train_t)):
        train_acc[i] = np.mean(train_y == (train_score > train_t[i]))
    test_acc = np.zeros(len(test_t))
    for i in range(len(test_t)):
        test_acc[i] = np.mean(test_y == (test_score > test_t[i]))

    # write predictions
    # softmax predictions
    order_y, order_score = strainer.compute_predictions(soft=True)
    _, order_pred = strainer.compute_predictions()
    doublet_score = order_score[:, 1]
    np.save(os.path.join(args.out_dir, 'no_updates_softmax_scores.npy'), doublet_score[:num_cells])
    np.save(os.path.join(args.out_dir, 'no_updates_softmax_scores_sim.npy'), doublet_score[num_cells:])

    # logit predictions
    logit_y, logit_score = logits_strainer.compute_predictions(soft=True)
    logit_doublet_score = logit_score[:, 1]
    np.save(os.path.join(args.out_dir, 'logit_scores.npy'), logit_doublet_score[:num_cells])
    np.save(os.path.join(args.out_dir, 'logit_scores_sim.npy'), logit_doublet_score[num_cells:])


    # update threshold as a function of Solo's estimate of the number of
    # doublets
    # essentially a log odds update
    # TODO put in a function
    diff = np.inf
    counter_update = 0
    solo_scores = doublet_score[:num_cells]
    logit_scores = logit_doublet_score[:num_cells]
    d_s = (args.doublet_ratio / (args.doublet_ratio + 1))
    while (diff > .01) | (counter_update < 5):

        # calculate log odss calibration for logits
        d_o = np.mean(solo_scores)
        c = np.log(d_o/(1-d_o)) - np.log(d_s/(1-d_s))

        # update soloe scores
        solo_scores = 1 / (1+np.exp(-(logit_scores + c)))

        # update while conditions
        diff = np.abs(d_o - np.mean(solo_scores))
        counter_update += 1

    np.save(os.path.join(args.out_dir, 'softmax_scores.npy'),
            solo_scores)

    if args.expected_number_of_doublets is not None:
        k = len(solo_scores) - args.expected_number_of_doublets
        if args.expected_number_of_doublets / len(solo_scores) > .5:
            print('''Make sure you actually expect more than half your cells
                   to be doublets. If not change your
                   -e parameter value''')
        assert k > 0
        idx = np.argpartition(solo_scores, k)
        threshold = np.max(solo_scores[idx[:k]])
        is_solo_doublet = solo_scores > threshold
    else:
        is_solo_doublet = solo_scores > .5

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
        adata.obs['logit_scores'] = logit_doublet_score[:num_cells]
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
        sns.distplot(doublet_score[:num_cells], label='Observed')
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, 'real_cells_dist.pdf'))
        plt.close()

        scvi_umap = umap.UMAP(n_neighbors=16).fit_transform(latent)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(scvi_umap[:, 0], scvi_umap[:, 1],
                   c=doublet_score[:num_cells], s=8, cmap="GnBu")

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        fig.savefig(os.path.join(args.out_dir, 'umap_solo_scores.pdf'))

###############################################################################
# __main__
###############################################################################


if __name__ == '__main__':
    main()
