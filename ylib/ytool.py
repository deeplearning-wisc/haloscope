import torch
import numpy as np


def cluster_acc(y_true, y_pred, print_ret=False):
    from scipy.optimize import linear_sum_assignment

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    acc = w[row_ind, col_ind].sum() / y_pred.size
    if print_ret:
        print("Fit acc: ", acc)
    return acc


class ArrayDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, features, labels=None) -> None:
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        if self.labels is None:
            return self.features[index]
        else:
            return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)