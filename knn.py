
def find_knearest_neighbor(dist_mat, k=1):
    """
    Args:
      - dist_mat (np.ndarray): distance matrix (shape = (n_query, n_label))
      - k: number of neighborhoods
    Return:
      - nn_labels (np.ndarray): label ids of each query's k-nearest neighbor (shape = (n_query, k))
    """
    return np.argsort(dist_mat, axis=1)[:, :k]


def cal_accuracy(nn_labels, gold_labels):
    """
    Args:
      - nn_labels (np.ndarray): label ids of each query's k-nearest neighbor (shape = (n_query, k))
      - gold_lables (list like object): gold nearest neighbor, paired ids [query, label] (shape=(num, 2))
    """
    n_dat = len(gold_labels)
    n_corr = sum(1 for q, l in gold_labels if l in nn_labels[q])
    return n_dat / n_corr


def cal_map(dist_mat, gold_labels):
    raise NotImplementedError


def simple_projection_decode(xs, ys, map_mat, metric="cos"):
    """
    Args:
      - xs: feature vectors in source space
      - ys: feature vectors in target space
      - map_mat: leant matrix for mapping from source to target
      - metric: similarity measure
    """
    proj_xs = xs.dot(map_mat)
    if metric == "cos":
        # normalize
        proj_xs /= np.broadcast_to(np.expand_dims(proj_xs, axis=1), proj_xs.shape)
        ys /= np.broadcast_to(np.expand_dims(ys, axis=1), proj_ys.shape)
        sim_mat = proj_xs.dot(ys.T)
        dist_mat = - sim_mat # as distance
    elif metric == "euclid":
        dist_mat = np.array([[np.sum(x-y)**2 for y in ys] for x in xs])
    else:
        raise NotImplementedError("Unavailable metric: {}".format(metric))
    return dist_mat
