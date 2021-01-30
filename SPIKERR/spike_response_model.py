import warnings
import torch as t
from torch.nn import Module
from typing import List, Callable, Generator, Tuple

InputSpikes = t.Tensor
Voltages = t.Tensor
Weights = t.Tensor
SpikeHistory = List[t.Tensor]
OutputSpikes = t.Tensor
Index = int


class SpikeResponseModel(Module):
    def __init__(
        self,
        weights: Weights,
        get_output_spikes_: Callable[
            [InputSpikes, Voltages, Weights, SpikeHistory], OutputSpikes
        ],
    ):
        super(SpikeResponseModel, self).__init__()

        if weights.dim() != 2:
            raise ValueError("weights must be a 2D tensor of input_size x output_size.")

        self.weights = weights
        self.get_output_spikes_ = get_output_spikes_

        self.spike_history: SpikeHistory = []
        output_size = weights.shape[1]
        self.voltages = t.zeros((1, output_size))

    # TODO: Handle batching properly.
    def forward(
        self, spike_train_generator: Generator[Tuple[InputSpikes, Index], None, None]
    ) -> Generator[Tuple[OutputSpikes, Index], None, None]:
        # Reset back to baseline.
        self.spike_history: SpikeHistory = []
        self.voltages.fill_(0)

        for spike_train, index in spike_train_generator:
            # Rough check to save compute time, but still catch most errors.
            passes_max_check = spike_train.max() == 0 or spike_train.max() == 1
            passes_min_check = spike_train.min() == 0 or spike_train.min() == 1
            is_likely_binary = passes_max_check and passes_min_check
            if not is_likely_binary:
                raise ValueError("Input tensor must contain only zeros and ones.")

            self.spike_history.append(spike_train)
            spikes = self.get_output_spikes_(
                spike_train, self.voltages, self.weights, self.spike_history
            )

            if self.voltages.shape != spikes.shape:
                warnings.warn(
                    f"voltage shape of {self.voltages.shape} differed from spike_train shape of {spike_train.shape}."
                )

            yield spikes, index
