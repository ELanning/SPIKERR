r"""
Module for encoding inputs into a format recognizable by SNNs.
"""
import torch as t
from typing import Optional, Tuple


Spikes = t.Tensor
Index = int


def poisson_encode(
    data: t.Tensor, spike_train_count: Optional[int]
) -> Tuple[Spikes, Index]:
    """
    Encodes the data into a tensor of Poisson spikes.
    Formally, it is a function f: R -> {0, 1}^n.
    A 4x4 image would be encoded into a 16x1 tensor: one spike per pixel.
    Can be made deterministic by invoking torch.manual_seed before calling.

    Reference: https://arxiv.org/ftp/arxiv/papers/1604/1604.06751.pdf

    @param data: The data to encode.
    @param spike_train_count: The number of times to iterate before terminating. Iterates forever if None.
    @return: A generator that provides a new spike tensor on each iteration, indicating if a data point spiked or not.
    @raise ValueError: spike_train_count is not None or a positive integer.
    """
    if spike_train_count is not None:
        if spike_train_count < 1:
            raise ValueError("spike_train_count must be a positive integer.")

    # Normalize the data, if necessary.
    if data.min() < 0:
        data += t.abs(data.min())
    if data.max() > 1.0:
        data /= data.max()

    # Typically a color pixel density, but could be anything.
    point_densities = data.flatten()
    point_densities_count = point_densities.numel()

    iterator = loop_forever()
    if spike_train_count is not None:
        iterator = range(spike_train_count)

    for index in iterator:
        uniform_tensor = t.empty(point_densities_count).uniform_(0, 1)
        # un-squeeze to format return as a 1xN spike train.
        spikes = point_densities.ge(uniform_tensor).float().unsqueeze(0)
        yield spikes, index


def loop_forever():
    i = -1
    while True:
        i += 1
        yield i
