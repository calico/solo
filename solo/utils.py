import numpy as np

from scvi.dataset import GeneExpressionDataset
from scipy.stats import multinomial


def create_average_doublet(X: np.ndarray,
                           i: int,
                           j: int, **kwargs):
    '''make an average combination of 2 cells

    Parameters
    ----------
    X : np.array
      cell by genes matrix
    i : int,
      randomly chosen ith cell
    j : int,
      randomly chosen jth cell
    Returns
    -------
    float64
        average expression vector of two cells
    '''
    return (X[i, :] + X[j, :]).astype('float64') / 2


def create_summed_doublet(X: np.ndarray,
                          i: int,
                          j: int, **kwargs):
    '''make a sum combination of 2 cells

    Parameters
    ----------
    X : np.array
      cell by genes matrix
    i : int,
      randomly chosen ith cell
    j : int,
      randomly chosen jth cell
    Returns
    -------
    float64
        summed expression vector of two cells
    '''
    return (X[i, :] + X[j, :]).astype('float64')


def create_multinomial_doublet(X: np.ndarray,
                               i: int,
                               j: int, **kwargs):
    '''make a multinomial combination of 2 cells

    Parameters
    ----------
    X : np.array
        cell by genes matrix
    i : int,
        randomly chosen ith cell
    j : int,
        randomly chosen jth cell
    kwargs : dict,
        dict with doublet_depth, cell_depths and cells_ids as keys
        doublet_depth is an int
        cell_depths is an list of all cells total UMI counts as ints
        cell_ids list of lists with genes with counts for each cell
    Returns
    -------
    float64
        multinomial expression vector of two cells
    '''
    doublet_depth = kwargs["doublet_depth"]
    cell_depths = kwargs["cell_depths"]
    cells_ids = kwargs["cells_ids"]
    # add their counts
    dp = (X[i, :]
          + X[j, :]).astype('float64')
    dp = np.ravel(dp)
    non_zero_indexes = np.unique(cells_ids[i] + cells_ids[j])
    # a huge hack caused by
    # https://github.com/numpy/numpy/issues/8317
    # fun fun fun https://stackoverflow.com/questions/23257587/how-can-i-avoid-value-errors-when-using-numpy-random-multinomial
    # okay with this hack because affects pro
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


def make_gene_expression_dataset(data: np.ndarray, gene_names: np.ndarray):
    '''make an scVI GeneExpressionDataset

    Parameters
    ----------
    data : np.array
        cell by genes matrix
    gene_names : np.array,
        string array with gene names
    Returns
    -------
    ge_data : GeneExpressionDataset
        scVI GeneExpressionDataset for scVI processing
    '''
    ge_data = GeneExpressionDataset()
    ge_data.populate_from_data(X=data, gene_names=gene_names)
    return ge_data
