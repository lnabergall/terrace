""""""

import functools
from itertools import chain

import torch
from torch.nn.functional import pad

from . import tensor_utilities as utils


def accuracy(predictions, labels, pad=True, 
             weights_fn=torch.ones_like, ignore=None):
    labels = labels.long()
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    weights = weights_fn(labels).float()
    if labels.dim() == predictions.dim():
        keepdim = True
    else:
        keepdim = False
    greedy_choices = predictions.max(-1, keepdim=keepdim)[1]

    return torch.mean(greedy_choices.eq(labels).float()*weights)


def accuracy_topk(predictions, labels, k=5, pad=True, 
                  weights_fn=torch.ones_like, ignore=None):
    labels = labels.long()
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    weights = weights_fn(labels).float()
    topk_greedy_choices = predictions.topk(k)[1]
    expanded_labels = labels.repeat(*[1 for i in range(labels.dim()-1)] + [k])
    weights = weights.repeat(*[1 for i in range(labels.dim()-1)] + [k])

    return torch.mean(topk_greedy_choices.eq(expanded_labels).float()*weights)


def sequence_accuracy(predictions, labels, pad=True, seq_axis=2, 
                      weights_fn=torch.ones_like, ignore=None):
    labels = labels.long()
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    weights = weights_fn(labels).float()
    if labels.dim() == predictions.dim():
        keepdim = True
    else:
        keepdim = False
    greedy_choices = predictions.max(-1, keepdim=keepdim)[1]
    not_correct = greedy_choices.ne(labels).float() * weights
    not_correct = not_correct.sum(seq_axis-1)

    return 1 - (torch.nonzero(not_correct).shape[0] / not_correct.nelement())


def mean_squared_error(predictions, labels, root=True, pad=True, 
                       weights_fn=torch.ones_like, ignore=None):
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    weights = weights_fn(labels).float()
    mse = torch.mean(torch.pow(predictions - labels, 2).float()*weights)
    if root:
        error = mse.sqrt()
    else:
        error = mse
    return error


def neg_log_perplexity(predictions, labels, log_probs=True, base=2, pad=True, 
                       weights_fn=torch.ones_like, ignore=None):
    """
    If log_probs is False, assumes predictions are probabilities, that is, 
    lie between 0 and 1; otherwise, it assumes that they are log probabilities.
    Also, it assumes labels is a torch.LongTensor.
    """
    labels = labels.long()
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    weights = weights_fn(labels).float()
    if labels.dim() < predictions.dim(): 
        # Assumes the last dim of predictions represents a distribution 
        # over classes and the last dim of labels is a class index. 
        label_probabilities = predictions.gather(-1, labels.unsqueeze(-1))
    else:
        # Assumes labels is an indicator Tensor of the same shape as predictions
        label_probabilities = predictions * labels
    if log_probs:
        log_perp = torch.mean(label_probabilities.float()*weights)
    else:
        log_perp = torch.mean(
            (torch.log(label_probabilities.float())/log(base))*weights)

    return log_perp


def approximate_bleu(predictions, labels, ignore=None):
    raise NotImplementedError


def bleu(predictions, labels, ignore=None):
    raise NotImplementedError


def rouge_n_fscore(predictions, labels, n, ignore=None):
    raise NotImplementedError


def rouge_l_fscore(predictions, labels, ignore=None):
    raise NotImplementedError


BUILTIN_METRICS = {
    "accuracy": accuracy,
    "accuracy_topk": accuracy_topk,
    "accuracy_top5": functools.update_wrapper(
        functools.partial(accuracy_topk, k=5), accuracy_topk),
    "sequence_accuracy": sequence_accuracy,
    "mse": functools.update_wrapper(functools.partial(
        mean_squared_error, root=False), mean_squared_error),
    "rmse": mean_squared_error,
    "neg_log_perplexity": neg_log_perplexity,
}


def resolve_metrics(metrics):
    return [BUILTIN_METRICS[metric] if isinstance(metric, str) else metric 
            for metric in metrics]