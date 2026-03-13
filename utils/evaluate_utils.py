import math
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_acc(y_true, y_pred, topK=None, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)
        topK: (optional) number of top clusters to consider for accuracy calculation
        return_ind: whether to return the indices of the optimal assignment

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1  # number of clusters
    w = np.zeros((D, D), dtype=int)  # Confusion matrix
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Apply topK constraint to limit the number of clusters
    if topK:
        # Get cluster sizes and sort them in descending order
        cluster_sizes = np.sum(w, axis=1)
        topK_indices = np.argsort(cluster_sizes)[::-1][:topK]  # Select topK largest clusters
        w = w[topK_indices, :]
        w = w[:, topK_indices]  # Re-sort confusion matrix to only use the topK clusters

    # Perform Hungarian algorithm for optimal assignment
    ind = linear_assignment(w.max() - w)  # Maximize the matching score (subtracting from max)
    ind = np.vstack(ind).T  # Convert the assignment to the proper format

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def split_cluster_acc_v1(y_true, y_pred, mask, topK=None):
    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :param topK: (optional) number of top clusters to consider for accuracy calculation
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask], topK=topK)
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask], topK=topK)
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc


def split_cluster_acc_v2(y_true, y_pred, mask, topK=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)
        topK: (optional) number of top clusters to consider for accuracy calculation

    # Return
        accuracy, in [0,1]
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Apply topK constraint to limit the number of clusters
    if topK:
        # Get cluster sizes and sort them in descending order
        cluster_sizes = np.sum(w, axis=1)
        topK_indices = np.argsort(cluster_sizes)[::-1][:topK]  # Select topK largest clusters
        w = w[topK_indices, :]
        w = w[:, topK_indices]  # Re-sort confusion matrix to only use the topK clusters

    # Perform Hungarian algorithm for optimal assignment
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


def binomial_coefficient(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def get_dis_max(args):
    Y_U, L = args.unlabeled_nums, args.hash_code_length
    target = (2 ** L) / Y_U

    for d in range(2, L + 1):
        lower_sum = binomial_coefficient(L, d - 2)
        upper_sum = binomial_coefficient(L, d - 1)

        if lower_sum <= target <= upper_sum:
            print("d_max:", d)
            return d

    return 0
