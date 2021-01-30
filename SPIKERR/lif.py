r"""
Module for simulating the dynamics of a leaky integrate and fire spiking neural network.
"""
import torch as t
from typing import Union


def get_lif_output_current_(input_current_: t.Tensor, weights: t.Tensor) -> t.Tensor:
    # Edge case check: prevents inputs of 1x1 or 1 from being destroyed by squeeze.
    is_1x1_edge_case = input_current_.shape == (1, 1)
    is_1_dimension_edge_case = (
        len(input_current_.shape) == 1 and input_current_.shape[0] == 1
    )
    if not is_1x1_edge_case and not is_1_dimension_edge_case:
        input_current_.squeeze_()

    # Rescale the input result into a dimension that can be multiplied by the weights.
    # For example, a 24x24 image is rescaled and transformed to a 1x784 (24 * 24 = 784) spike train.
    # This 1x784 spike train is then multiplied by a 784x400 layer of neurons, which represents a fully connected
    # layer of 400 neurons receiving stimuli from 784 input neurons, to get the total voltage of each output neuron.
    input_dimension = input_current_.shape[0]
    output_dimension = weights.shape[1]
    rescaled_input_current = input_current_.repeat(output_dimension).view(
        (input_dimension, output_dimension)
    )
    output_current = t.sum(weights * rescaled_input_current, dim=0)
    return output_current


def get_lif_derivative(voltages: t.Tensor, input_current: t.Tensor) -> t.Tensor:
    return -1 * voltages + input_current


def get_spikes(voltages: t.Tensor, spike_threshold: Union[float, t.Tensor]) -> t.Tensor:
    ones = t.ones_like(voltages)
    zeros = t.zeros_like(voltages)
    spikes = zeros.where(voltages < spike_threshold, ones)
    return spikes


def reset_where_spiked_(
    voltages_: t.Tensor, spike_threshold: Union[float, t.Tensor]
) -> None:
    zeros = t.zeros_like(voltages_)
    after_spiked = voltages_.where(voltages_ < spike_threshold, zeros)
    voltages_.copy_(after_spiked)


def run_lif_step_(
    spike_threshold: float,
    excitatory_time_constant_ms: float,
    input_current_: t.Tensor,
    weights: t.Tensor,
    voltages_: t.Tensor,
) -> t.Tensor:
    current = get_lif_output_current_(input_current_, weights)
    derivative = get_lif_derivative(voltages_, current)
    voltages_.add_(derivative / excitatory_time_constant_ms)

    spikes = get_spikes(voltages_, spike_threshold)
    reset_where_spiked_(voltages_, spike_threshold)

    return spikes
