import torch as t
from typing import Generator, Tuple

Index = int
InputSpikes = t.Tensor


class AveragesDecoder(t.nn.Module):
    def __init__(self):
        super(AveragesDecoder, self).__init__()

    def forward(
        self, spike_train_generator: Generator[Tuple[InputSpikes, Index], None, None]
    ) -> t.Tensor:
        result = None
        for spike_train, index in spike_train_generator:
            if result is None:
                result = spike_train
            else:
                result.add_(spike_train)

        if result is None:
            raise ValueError("spike_train_generator must yield at least one spike.")

        result.div_(index + 1)
        return result
