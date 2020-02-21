import numpy as np

from scvi.dataset import GeneExpressionDataset
from scipy.stats import multinomial
from sklearn.neighbors import NearestNeighbors


def knn_smooth_pred_class(X: np.ndarray,
                          pred_class: np.ndarray,
                          grouping: np.ndarray = None,
                          k: int = 15,) -> np.ndarray:
    '''
    Smooths class predictions by taking the modal class from each cell's
    nearest neighbors.
    Parameters
    ----------
    X : np.ndarray
        [N, Features] embedding space for calculation of nearest neighbors.
    pred_class : np.ndarray
        [N,] array of unique class labels.
    groupings : np.ndarray
        [N,] unique grouping labels for i.e. clusters.
        if provided, only considers nearest neighbors *within the cluster*.
    k : int
        number of nearest neighbors to use for smoothing.
    Returns
    -------
    smooth_pred_class : np.ndarray
        [N,] unique class labels, smoothed by kNN.
    Examples
    --------
    >>> smooth_pred_class = knn_smooth_pred_class(
    ...     X = X,
    ...     pred_class = raw_predicted_classes,
    ...     grouping = louvain_cluster_groups,
    ...     k = 15,)
    Notes
    -----
    scNym classifiers do not incorporate neighborhood information.
    By using a simple kNN smoothing heuristic, we can leverage neighborhood
    information to improve classification performance, smoothing out cells
    that have an outlier prediction relative to their local neighborhood.
    '''
    if grouping is None:
        # do not use a grouping to restrict local neighborhood
        # associations, create a universal pseudogroup `0`.
        grouping = np.zeros(X.shape[0])

    smooth_pred_class = np.zeros_like(pred_class)
    for group in np.unique(grouping):
        # identify only cells in the relevant group
        group_idx = np.where(grouping == group)[0].astype('int')
        X_group = X[grouping == group, :]
        # if there are < k cells in the group, change `k` to the
        # group size
        if X_group.shape[0] < k:
            k_use = X_group.shape[0]
        else:
            k_use = k
        # compute a nearest neighbor graph and identify kNN
        nns = NearestNeighbors(n_neighbors=k_use,).fit(X_group)
        dist, idx = nns.kneighbors(X_group)

        # for each cell in the group, assign a class as
        # the majority class of the kNN
        for i in range(X_group.shape[0]):
            classes = pred_class[group_idx[idx[i, :]]]
            uniq_classes, counts = np.unique(classes, return_counts=True)
            maj_class = uniq_classes[int(np.argmax(counts))]
            smooth_pred_class[group_idx[i]] = maj_class
    return smooth_pred_class


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
    randomize_doublet_size = kwargs["randomize_doublet_size"]

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
    if randomize_doublet_size:
        scale_factor = np.random.uniform(1., doublet_depth)
    else:
        scale_factor = doublet_depth
    # choose depth
    dd = int(scale_factor * (cell_depths[i] + cell_depths[j]) / 2)

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
