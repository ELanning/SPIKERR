import unittest
import torch as t
from SPIKERR.lif import run_lif_step_


class TestLif(unittest.TestCase):
    def test_lif_constant_current(self):
        # Validate Leaky Integrate and Fire dynamics on a single neuron,
        # using the Euler method with a constant input current.
        arbitrary_weight = 0.2125
        weights = t.tensor([[arbitrary_weight]])
        voltages = t.tensor([[0]]).float()

        for time_step in range(4):
            one_spike = t.tensor([1])
            run_lif_step_(
                spike_threshold=1000,  # Arbitrary high number such that the neuron never spikes.
                excitatory_time_constant_ms=1,
                input_current_=one_spike,
                weights=weights,
                voltages_=voltages,
            )
            neuron_voltage = voltages[0][0]
            expected_constant_voltage = 0.2125
            self.assertEqual(
                expected_constant_voltage,
                neuron_voltage,
                f"neuron_voltage must be equal to {expected_constant_voltage}.\n"
                f"Received a voltage of {neuron_voltage}",
            )
