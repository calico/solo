import numpy as np

from scvi.dataset import GeneExpressionDataset
from scipy.stats import multinomial


def create_average_doublet(X, i, j, **kwargs):
    return (X[i, :] + X[j, :]).astype('float64') / 2


def create_summed_doublet(X, i, j, **kwargs):
    return (X[i, :] + X[j, :]).astype('float64')


def create_multinomial_doublet(X, i, j, **kwargs):

    doublet_depth = kwargs["doublet_depth"]
    cell_depths = kwargs["cell_depths"]

    # add their counts
    dp = (X[i, :]
          + X[j, :]).astype('float64')

    # normalize
    dp /= dp.sum()

    # choose depth
    dd = int(doublet_depth * (cell_depths[i] + cell_depths[j]) / 2)

    # sample counts from multinomial
    return multinomial.rvs(n=dd, p=dp)


def make_gene_expression_dataset(data, gene_names):
    means, var = GeneExpressionDataset.library_size(data)
    data_length = data.shape[0]
    batch = np.zeros((data_length, 1), dtype='uint32')
    labels = np.ones((data_length, 1), dtype='uint32')
    return GeneExpressionDataset(data, local_means=means, local_vars=var,
                                 batch_indices=batch, labels=labels,
                                 gene_names=gene_names)
