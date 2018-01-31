""""""

from itertools import chain

import torch
from torch.nn.functional import pad


def pad_with_zeros(x, y, axis=2, length=None):
    """Pad tensor x on provided axis to match tensor y or pad both to length."""
    if length is not None and (length < x.shape[axis] or length < y.shape[axis]):
        raise ValueError("Provided length is not compatible with these tensors.")
    if length is None:
        length = max(x.shape[axis], y.shape[axis])
    x_length_diff = length - x.shape[axis]
    y_length_diff = length - y.shape[axis]
    x_padding_spec = list(chain.from_iterable(
        [(0, x_length_diff) if x.dim()-i == axis else (0, 0) for i in range(x.dim())]))
    y_padding_spec = list(chain.from_iterable(
        [(0, y_length_diff) if x.dim()-i == axis else (0, 0) for i in range(x.dim())]))
    x_padded = pad(x, x_padding_spec, mode="constant", value=0).data
    y_padded = pad(y, y_padding_spec, mode="constant", value=0).data
    return x_padded, y_padded


def shift_right(x, pad_tensor=None, variable=False):
    """Shift the second axis of x right by one, removes the last 'column'."""
    if pad_tensor is None:
        x_shifting_spec = list(chain.from_iterable(
            [(1, 0) if x.dim()-i == 2 else (0, 0) for i in range(x.dim())]))
        x_shifted = pad(x, x_shifting_spec, mode="constant", value=0)[:, :-1, ...]
    else:
        x_shifted = torch.cat([pad_tensor, x], dim=1)[:, :-1, ...]
    if not variable:
        x_shifted = x_shifted.data
    return x_shifted
