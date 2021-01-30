import torch as t
import torchvision as tv
from math import ceil, floor, sqrt
from torch.utils.tensorboard import SummaryWriter
from typing import List
from SPIKERR.spike_response_model import SpikeResponseModel
from SPIKERR.lif import run_lif_step_
from SPIKERR.poisson_encoder import PoissonEncoder
from SPIKERR.averages_decoder import AveragesDecoder
from SPIKERR.weights import xavier_normal

image_transform = tv.transforms.Compose([tv.transforms.ToTensor()])
train_data_set = tv.datasets.MNIST(
    "./", train=False, transform=image_transform, target_transform=None, download=True
)
tensorboard = SummaryWriter()
input_size = 28 * 28  # MNIST dimensions.


# Fast enough for the use case.
def factor(x: int):
    results = [1, x]
    limit = int(sqrt(x)) + 1
    for i in range(2, limit):
        if x % i == 0:
            results.append(int(i))
            results.append(x // i)
    return results


# SNN config.
spike_threshold = 0.1
spike_train_count = 10
excitatory_time_constant_ms = 100

hidden_layer_neuron_count = 80
hidden_layer_factors = factor(hidden_layer_neuron_count)
hidden_layer_print_width = hidden_layer_factors[-1]
hidden_layer_print_height = hidden_layer_factors[-2]

output_layer_neuron_count = 9
output_layer_factors = factor(output_layer_neuron_count)
output_layer_print_width = output_layer_factors[-1]
output_layer_print_height = output_layer_factors[-2]


def get_lif_hidden_spikes(
    input_spikes: t.Tensor,
    voltages: t.Tensor,
    weights: t.Tensor,
    spike_history: List[t.Tensor],
) -> t.Tensor:
    # Multiply by 10 to make the heatmap more visible and format as a rectangle.
    voltage_heatmap = 10 * voltages.reshape(
        1, hidden_layer_print_width, hidden_layer_print_height
    )
    tensorboard.add_image(
        "hidden_voltage_heatmap", voltage_heatmap, len(spike_history), dataformats="CHW"
    )

    spikes = run_lif_step_(
        spike_threshold, excitatory_time_constant_ms, input_spikes, weights, voltages
    )
    return spikes


def get_lif_output_spikes(
    input_spikes: t.Tensor,
    voltages: t.Tensor,
    weights: t.Tensor,
    spike_history: List[t.Tensor],
) -> t.Tensor:
    # Multiply by 10 to make the heatmap more visible and format as a rectangle.
    voltage_heatmap = 10 * voltages.reshape(
        1, output_layer_print_width, output_layer_print_height
    )
    tensorboard.add_image(
        "output_voltage_heatmap",
        voltage_heatmap,
        len(spike_history),
        dataformats="CHW",
    )

    spikes = run_lif_step_(
        spike_threshold, excitatory_time_constant_ms, input_spikes, weights, voltages
    )
    return spikes


model = t.nn.Sequential(
    PoissonEncoder(spike_train_count=10),
    SpikeResponseModel(
        xavier_normal(input_size, hidden_layer_neuron_count, positive_ratio=1.0),
        get_output_spikes_=get_lif_hidden_spikes,
    ),
    SpikeResponseModel(
        xavier_normal(
            hidden_layer_neuron_count, output_layer_neuron_count, positive_ratio=1.0
        ),
        get_output_spikes_=get_lif_output_spikes,
    ),
    AveragesDecoder(),
)


def train():
    i = 0
    for image_vector, label in train_data_set:
        i += 1
        if i == 10:
            break
        model.forward(image_vector)


train()
