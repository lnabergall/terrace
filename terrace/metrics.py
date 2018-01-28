""""""

import functools
from itertools import chain

import torch
from torch.nn.functional import pad


BUILTIN_METRICS = {
    "accuracy": accuracy,
    "accuracy_topk": accuracy_topk,
    "accuracy_top5": functools.partial(accuracy_topk, k=5),
    "sequence_accuracy": sequence_accuracy,
    "mse": functools.partial(mean_squared_error, root=False),
    "rmse": mean_squared_error,
    "neg_log_perplexity": neg_log_perplexity,
}


def resolve_metrics(metrics):
    return [BUILTIN_METRICS[metric] if isinstance(metric, str) else metric 
            for metric in metrics]


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


def accuracy(predictions, labels, k, pad=True, 
             weights_fn=torch.ones_like, ignore=None):
    labels = labels.long()
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    if labels.dim() == predictions.dim():
        keepdim = True
    else:
        keepdim = False
    greedy_choices = predictions.max(-1, keepdim=keepdim)[1]
    return torch.mean(greedy_choices.eq(labels)*weights)


def accuracy_topk(predictions, labels, k=5, pad=True, 
                  weights_fn=torch.ones_like, ignore=None):
    labels = labels.long()
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    topk_greedy_choices = predictions.topk(k)[1]
    expanded_labels = labels.repeat(*[1 for i in range(labels.dim()-2)] + [k])
    return torch.mean(greedy_choices.eq(expanded_labels)*weights)


def sequence_accuracy(predictions, labels, pad=True, seq_axis=2, 
                      weights_fn=torch.ones_like, ignore=None):
    labels = labels.long()
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    if labels.dim() == predictions.dim():
        keepdim = True
    else:
        keepdim = False
    greedy_choices = predictions.max(-1, keepdim=keepdim)[1]
    not_correct = greedy_choices.ne(labels) * weights
    not_correct = not_correct.sum(time_axis)
    return 1 - (torch.nonzero(not_correct).size[0] / not_correct.nelement())


def mean_squared_error(predictions, labels, root=True, pad=True, 
                       weights_fn=torch.ones_like, ignore=None):
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    mse = torch.mean(torch.pow(predictions - labels, 2)*weights)
    if root:
        error = mse.sqrt()
    else:
        error = mse
    return error


def neg_log_perplexity(predictions, labels, base=2, pad=True, 
                       weights_fn=torch.ones_like, ignore=None):
    # Assumes predictions are probabilities, that is, lie between 0 and 1
    # Assumes labels is a torch.LongTensor
    labels = labels.long()
    if pad:
        predictions, labels = pad_with_zeros(predictions, labels)
    weights = weights_fn(labels)
    if labels.dim() < predictions.dim(): 
        # Assumes the last dim of predictions represents a distribution 
        # over classes and the last dim of labels is a class index. 
        label_probabilities = predictions.gather(-1, labels.unsqueeze(-1))
    else:
        # Assumes labels is an indicator Tensor of the same shape as predictions
        label_probabilities = predictions * labels
    return torch.mean(torch.log(label_probabilities)*weights)


def approximate_bleu(predictions, labels, ignore=None):
    raise NotImplementedError


def bleu(predictions, labels, ignore=None):
    raise NotImplementedError


def rouge_n_fscore(predictions, labels, n, ignore=None):
    raise NotImplementedError


def rouge_l_fscore(predictions, labels, ignore=None):
    raise NotImplementedError


