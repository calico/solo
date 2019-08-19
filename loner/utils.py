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

    # a huge hack caused by
    # https://github.com/numpy/numpy/issues/8317
    # fun fun fun https://stackoverflow.com/questions/23257587/how-can-i-avoid-value-errors-when-using-numpy-random-multinomial
    # okay with this hack because affects pro
    non_zero_indexes = np.where(dp > 0)[0]
    dp = dp[non_zero_indexes]
    # normalize
    dp /= dp.sum()

    # choose depth
    dd = int(doublet_depth * (cell_depths[i] + cell_depths[j]) / 2)

    # sample counts from multinomial
    non_zero_probs = multinomial.rvs(n=dd, p=dp)
    probs = np.zeros(X.shape[1])
    probs[non_zero_indexes] = non_zero_probs
    return probs


def make_gene_expression_dataset(data, gene_names):
    ge_data = GeneExpressionDataset()
    ge_data.populate_from_data(X=data, gene_names=gene_names)
    return ge_data
