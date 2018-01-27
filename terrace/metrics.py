""""""

from itertools import chain

import torch
from torch.nn.functional import pad


def pad_with_zeros(x, y, axis=1, length=None):
    """Pad tensor x on provided axis to match tensor y or pad both to length."""
    if length is not None and (length < x.shape[axis] or length < y.shape[axis]):
        raise ValueError("Provided length is not compatible with these tensors.")
    if length is None:
        length = max(x.shape[axis], y.shape[axis])
    x_length_diff = length - x.shape[axis]
    y_length_diff = length - y.shape[axis]
    x_padding_spec = chain.from_iterable(
        [(0, x_length_diff) if i+1 == axis else (0, 0) for i in range(x.dim())])
    y_padding_spec = chain.from_iterable(
        [(0, y_length_diff) if i+1 == axis else (0, 0) for i in range(x.dim())])
    x_padded = pad(x, x_padding_spec, mode="constant", value=0).data
    y_padded = pad(y, y_padding_spec, mode="constant", value=0).data

    return x_padded, y_padded


def accuracy(predictions, labels, k, pad=True, weights_fn=torch.ones_like):
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    pass


def accuracy_topk(predictions, labels, pad=True, weights_fn=torch.ones_like):
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    pass


def sequence_accuracy(predictions, labels, pad=True, weights_fn=torch.ones_like):
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    pass


def mean_squared_error(predictions, labels, root=True, 
                       pad=True, weights_fn=torch.ones_like):
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    mse = torch.mean(torch.pow(predictions - labels, 2)*weights)
    if root:
        error = torch.sqrt(mse)
    else:
        error = mse

    return error


def neg_log_perplexity(predictions, labels, pad=True, weights_fn=torch.ones_like):
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    pass


def approximate_bleu(predictions, labels):
    raise NotImplementedError


def bleu(predictions, labels):
    raise NotImplementedError


def rouge_n_fscore(predictions, labels, n):
    raise NotImplementedError


def rouge_l_fscore(predictions, labels):
    raise NotImplementedError


