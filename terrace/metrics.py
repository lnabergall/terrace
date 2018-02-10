""""""

import functools
from itertools import chain

import torch
from torch.nn.functional import pad

from . import tensor_utilities as utils


def accuracy(predictions, labels, pad=True, 
             weights_fn=torch.ones_like, ignore=0):
    labels = labels.long()
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    if ignore is not None:
        mask = (labels != ignore).float()
        mask_ratio = labels.nelement() / mask.nonzero().shape[0]
    else:
        mask = torch.ones_like(labels).float()
        mask_ratio = 1
    weights = weights_fn(labels).float()
    if labels.dim() == predictions.dim():
        keepdim = True
    else:
        keepdim = False
    greedy_choices = predictions.max(-1, keepdim=keepdim)[1]
    return mask_ratio * torch.mean(greedy_choices.eq(labels).float()*mask*weights)


def accuracy_topk(predictions, labels, k=5, pad=True, 
                  weights_fn=torch.ones_like, ignore=0):
    labels = labels.long()
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    if ignore is not None:
        mask = (labels != ignore).float()
        mask_ratio = labels.nelement() / mask.nonzero().shape[0]
    else:
        mask = torch.ones_like(labels).float()
        mask_ratio = 1
    weights = weights_fn(labels).float()
    topk_greedy_choices = predictions.topk(k)[1]
    expanded_labels = labels.unsqueeze(-1).expand(
        *[-1 for i in range(labels.dim())] + [k])
    weights = weights.unsqueeze(-1).expand(
        *[-1 for i in range(weights.dim())] + [k])
    mask = mask.unsqueeze(-1).expand(
        *[-1 for i in range(mask.dim())] + [k])
    return k * mask_ratio * torch.mean(
        topk_greedy_choices.eq(expanded_labels).float()*mask*weights)


def sequence_accuracy(predictions, labels, pad=True, seq_axis=2, 
                      weights_fn=torch.ones_like, ignore=0):
    labels = labels.long()
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    if ignore is not None:
        mask = (labels != ignore).long()
    else:
        mask = torch.ones_like(labels).long()
    weights = weights_fn(labels).float()
    if labels.dim() == predictions.dim():
        keepdim = True
    else:
        keepdim = False
    greedy_choices = predictions.max(-1, keepdim=keepdim)[1] * mask
    not_correct = greedy_choices.ne(labels).float() * weights
    not_correct = not_correct.sum(seq_axis-1)

    return 1 - (torch.nonzero(not_correct).shape[0] / not_correct.nelement())


def mean_squared_error(predictions, labels, root=True, pad=True, 
                       weights_fn=torch.ones_like, ignore=0):
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    if ignore is not None:
        mask = (labels != ignore).float()
        mask_ratio = labels.nelement() / mask.nonzero().shape[0]
    else:
        mask = torch.ones_like(labels).float()
        mask_ratio = 1
    weights = weights_fn(labels).float()
    mse = mask_ratio * torch.mean(
        torch.pow(predictions - labels, 2).float()*mask*weights)
    if root:
        error = mse.sqrt()
    else:
        error = mse
    return error


def neg_log_perplexity(predictions, labels, log_probs=True, base=2, pad=True, 
                       weights_fn=torch.ones_like, ignore=0):
    """
    If log_probs is False, assumes predictions are probabilities, that is, 
    lie between 0 and 1; otherwise, it assumes that they are log probabilities.
    Also, it assumes labels is a torch.LongTensor.
    """
    labels = labels.long()
    if pad:
        predictions, labels = utils.pad_with_zeros(predictions, labels)
    if ignore is not None:
        mask = (labels != ignore).float()
        mask_ratio = labels.nelement() / mask.nonzero().shape[0]
    else:
        mask = torch.ones_like(labels).float()
        mask_ratio = 1
    weights = weights_fn(labels).float()
    if labels.dim() < predictions.dim(): 
        # Assumes the last dim of predictions represents a distribution 
        # over classes and the last dim of labels is a class index. 
        label_probabilities = predictions.gather(
            -1, labels.unsqueeze(-1)).squeeze(-1)
    else:
        # Assumes labels is an indicator Tensor of the same shape as predictions
        label_probabilities = torch.max(predictions * labels, -1)[0]
    if log_probs:
        log_perp = mask_ratio * torch.mean(
            label_probabilities.float()*mask*weights)
    else:
        log_perp = mask_ratio * torch.mean(
            (torch.log(label_probabilities.float())/log(base))*mask*weights)

    return log_perp


def approximate_bleu(predictions, labels, ignore=0):
    raise NotImplementedError


def bleu(predictions, labels, ignore=0):
    raise NotImplementedError


def rouge_n_fscore(predictions, labels, n, ignore=0):
    raise NotImplementedError


def rouge_l_fscore(predictions, labels, ignore=0):
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

BUILTIN_METRICS["accuracy_top5"].__name__ = "accuracy_top5"


def resolve_metrics(metrics):
    return [BUILTIN_METRICS[metric] if isinstance(metric, str) else metric 
            for metric in metrics]