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
          + X[j, :]).astype('float32')

    # normalize
    dp /= dp.sum()

    # a huge hack caused by
    # fun fun fun https://stackoverflow.com/questions/23257587/how-can-i-avoid-value-errors-when-using-numpy-random-multinomial
    # okay with this hack because affects pro
    off_zero = (dp.sum() - 1)
    if off_zero != 0:
        dp[np.where(dp > 0)[0][0]] = dp[dp > 0][0] - (off_zero)
    # choose depth
    dd = int(doublet_depth * (cell_depths[i] + cell_depths[j]) / 2)

    # sample counts from multinomial
    return multinomial.rvs(n=dd, p=dp)


def make_gene_expression_dataset(data, gene_names):
    ge_data = GeneExpressionDataset()
    ge_data.populate_from_data(X=data, gene_names=gene_names)
    return ge_data
