#!/usr/bin/env python
from optparse import OptionParser
import json
import os
import shutil

import numpy as np
from scipy.stats import multinomial
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns

from scvi.dataset import AnnDataset, LoomDataset, GeneExpressionDataset
from scvi.models import Classifier, VAE
from scvi.inference import UnsupervisedTrainer, Trainer
from scvi.inference.annotation import AnnotationPosterior
import torch
from torch.nn import functional as F

'''
loner.py

Simulate doublets, train a VAE, and then a classifier on top.
'''

###############################################################################
# main
###############################################################################
def main():
    usage = 'usage: %prog [options] <model_json> <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='doublet_depth',
                      default=1., type='float',
                      help='Depth multiplier for a doublet relative to the \
                      average of its \
                      constituents[Default: % default]')
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
        scvi_data = AnnDataset(data_file, save_path='./')
    else:
        print('Unrecognized file format')

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

        # calico-specific
        latent_file = '%s/latent.npy' % os.path.split(options.seed)[0]
        if os.path.isfile(latent_file):
            shutil.copy(latent_file, '%s/latent.npy' % options.out_dir)
        else:
            # rest of world
            latent_file = '%s/latent.csv' % os.path.split(options.seed)[0]
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

        # save latent
        utrainer.use_cuda = False
        utrainer.model.cpu()
        utrainer.get_all_latent_and_imputed_values(save_latent=True,
                                                   filename_latent='%s/latent.csv' % options.out_dir)

        if options.gpu:
            utrainer.use_cuda = True
            utrainer.model.cuda()

    ##################################################
    # simulate doublets

    cell_depths = scvi_data.X.sum(axis=1)
    num_doublets = int(options.doublet_ratio * num_cells)
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
    doublet_means, doublet_var = GeneExpressionDataset.library_size(X_doublets)
    doublet_batch = np.zeros((num_doublets, 1), dtype='uint32')
    doublet_labels = np.ones((num_doublets, 1), dtype='uint32')
    doublet_data = GeneExpressionDataset(X_doublets, local_means=doublet_means,
                                         local_vars=doublet_var,
                                         batch_indices=doublet_batch,
                                         labels=doublet_labels,
                                         gene_names=scvi_data.gene_names)

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
                                 sampling_model=vae,
                                 early_stopping_kwargs=stopping_params)

    # initial
    strainer.train(n_epochs=1000, lr=learning_rate)

    # drop learning rate and continue
    strainer.early_stopping.wait = 0
    strainer.train(n_epochs=250, lr=0.1 * learning_rate)

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


class ClassifierTrainer(Trainer):
    r"""The ClassifierInference class for training a classifier either on the
        raw data or on top of the latent
        space of another model (VAE, VAEC, SCANVI).

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or and
                     integer for the number of training samples
            to use Default: ``0.8``.
        :\**kwargs: Other keywords arguments from the general Trainer class.


    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes,
            n_batch=gene_dataset.n_batches * False,
            ... n_labels=gene_dataset.n_labels)

        >>> classifier = Classifier(vae.n_latent,
                                    n_labels=cortex_dataset.n_labels)
        >>> trainer = ClassifierTrainer(classifier, gene_dataset,
                                        sampling_model=vae, train_size=0.5)
        >>> trainer.train(n_epochs=20, lr=1e-3)
        >>> trainer.test_set.accuracy()
    """

    def __init__(self, *args, train_size=0.8, sampling_model=None,
                 use_cuda=True, **kwargs):
        self.sampling_model = sampling_model
        super().__init__(*args, use_cuda=use_cuda, **kwargs)
        self.train_set, self.test_set = self.train_test(self.model,
                                                        self.gene_dataset,
                                                        train_size,
                                                        type_class=AnnotationPosterior)
        self.train_set.to_monitor = ['accuracy']
        self.test_set.to_monitor = ['accuracy']

    @property
    def posteriors_loop(self):
        return ['train_set']

    def __setattr__(self, key, value):
        if key in ["train_set", "test_set"]:
            value.sampling_model = self.sampling_model
        super().__setattr__(key, value)

    def loss(self, tensors_labelled):
        x, _, _, _, labels_train = tensors_labelled
        if self.sampling_model:
            if hasattr(self.sampling_model, 'classify'):
                return F.cross_entropy(self.sampling_model.classify(x), labels_train.view(-1))
            else:
                if self.sampling_model.log_variational:
                    x = torch.log(1 + x)
                # x = self.sampling_model.z_encoder(x)[0]
                z = self.sampling_model.z_encoder(x)[0]
                l = self.sampling_model.l_encoder(x)[0]
                x = torch.cat((z,l), dim=-1)
        return F.cross_entropy(self.model(x), labels_train.view(-1))

    @torch.no_grad()
    def compute_predictions(self, soft=False):
        '''
        :return: the true labels and the predicted labels
        :rtype: 2-tuple of :py:class:`numpy.int32`
        '''
        model, cls = (self.sampling_model, self.model) if hasattr(self, 'sampling_model') else (self.model, None)
        full_set = self.create_posterior(type_class=AnnotationPosterior)
        full_set.sampling_model = model
        return compute_predictions(model, full_set, classifier=cls, soft=soft)


@torch.no_grad()
def compute_predictions(model, data_loader, classifier=None, soft=False):
    all_y_pred = []
    all_y = []

    for i_batch, tensors in enumerate(data_loader):
        sample_batch, _, _, _, labels = tensors
        all_y += [labels.view(-1).cpu()]

        if hasattr(model, 'classify'):
            y_pred = model.classify(sample_batch)
        elif classifier is not None:
            # Then we use the specified classifier
            if model is not None:
                if model.log_variational:
                    sample_batch = torch.log(1 + sample_batch)
                # sample_batch, _, _ = model.z_encoder(sample_batch)
                z = model.z_encoder(sample_batch)[0]
                l = model.l_encoder(sample_batch)[0]
                sample_batch = torch.cat((z,l), dim=-1)
            y_pred = classifier(sample_batch)
        else:  # The model is the raw classifier
            y_pred = model(sample_batch)

        if not soft:
            y_pred = y_pred.argmax(dim=-1)

        all_y_pred += [y_pred.cpu()]

    all_y_pred = np.array(torch.cat(all_y_pred))
    all_y = np.array(torch.cat(all_y))
    return all_y, all_y_pred


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

