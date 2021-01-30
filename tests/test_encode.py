import unittest
import torch as t
from SPIKERR.encode import poisson_encode

arbitrary_seed = 333
t.manual_seed(arbitrary_seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False


class TestEncode(unittest.TestCase):
    def test_poisson_encode(self):
        example = t.empty(3, 3).uniform_(0, 1)
        # A reasonably high number was chosen so that the decoder can properly reconstruct the input.
        time_slice_count = 120
        spike_train_generator = poisson_encode(
            example, spike_train_count=time_slice_count
        )
        # Destructure as index is not needed.
        encoding, _ = next(spike_train_generator)
        for spike_train, _ in spike_train_generator:
            encoding = t.cat([encoding, spike_train], dim=0)

        # Validate output dimension.
        expected_dimension = (time_slice_count, example.numel())
        self.assertEqual(
            expected_dimension,
            encoding.shape,
            f"encoding dimension must be equal to {expected_dimension}.\n"
            f"Received encoding of shape {encoding.shape}.",
        )

        # Validate decoding back to the original example.
        decoding = encoding.mean(dim=0).view(example.shape)
        approximately_close = t.all(t.abs(example - decoding) < 0.3)
        self.assertTrue(
            approximately_close, "decoding must be approximately equal to the original."
        )

        # Check Fano Factor, which can be used to measure if a process is Poisson.
        histogram = t.histc(encoding.sum(dim=0))
        fano_factor = histogram.var() / histogram.mean()
        self.assertTrue(
            abs(fano_factor - 1) < 0.1,
            "Fano factor must be close to one.\n"
            f"Received a fano_factor of {fano_factor}.",
        )

    def test_poisson_encode_bad_inputs(self):
        negative_spike_train_count = -1
        empty_spike_train_count = 0

        self.assertRaises(
            ValueError,
            lambda: next(poisson_encode(t.empty(1), negative_spike_train_count)),
        )
        self.assertRaises(
            ValueError,
            lambda: next(poisson_encode(t.empty(1), empty_spike_train_count)),
        )