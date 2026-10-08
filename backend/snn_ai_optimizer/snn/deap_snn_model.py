"""
Spiking Neural Network architecture for DEAP EEG Arousal classification.

Built using SpikingJelly activation-based multi-step Leaky Integrate-and-Fire (LIF)
neurons with surrogate-gradient backpropagation.

Architecture
------------
Input:  (T, B, 128) Poisson spike trains from 128 Welch PSD band-power features.
Layer 1: Linear(128, 256) -> LIFNode(surrogate=ATan())
Layer 2: Linear(256, 128) -> LIFNode(surrogate=ATan())
Layer 3: Linear(128, 2)   -> LIFNode(surrogate=ATan())
Output: (T, B, 2) binary spike trains.
Readout: Mean firing rate over simulation time-steps T: (B, 2) logits.

Research integrity
------------------
* Consumes real EEG band-power features extracted from DEAP recordings.
* Implements multi-step execution ('m' mode) for biological fidelity and efficiency.
* Firing rates represent continuous state evidence for classification.
"""

from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False


class FallbackLIF(nn.Module):
    """Fallback module in case SpikingJelly is not installed."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(x) > 0.5).float()


class DEAPArousalSNN(nn.Module):
    """
    3-layer Spiking Neural Network for binary Arousal classification.

    Parameters
    ----------
    input_dim : int, default=128
        Dimension of the input feature vector (32 channels * 4 frequency bands).
    hidden_dims : tuple of int, default=(256, 128)
        Hidden layer dimensions.
    num_classes : int, default=2
        Number of output classes (0: LOW_AROUSAL, 1: HIGH_AROUSAL).
    tau : float, default=2.0
        Membrane time constant for LIF neurons.
    v_threshold : float, default=1.0
        Threshold membrane potential for spike firing.
    v_reset : float, default=0.0
        Reset membrane potential after spike generation.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: Sequence[int] = (256, 128),
        num_classes: int = 2,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            if HAS_SPIKINGJELLY:
                layers.append(layer.Linear(prev_dim, h_dim))
                layers.append(
                    neuron.LIFNode(
                        tau=tau,
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        surrogate_function=surrogate.ATan(),
                    )
                )
            else:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(FallbackLIF())
            prev_dim = h_dim

        # Final readout classification layer
        if HAS_SPIKINGJELLY:
            layers.append(layer.Linear(prev_dim, num_classes))
            layers.append(
                neuron.LIFNode(
                    tau=tau,
                    v_threshold=v_threshold,
                    v_reset=v_reset,
                    surrogate_function=surrogate.ATan(),
                )
            )
        else:
            layers.append(nn.Linear(prev_dim, num_classes))
            layers.append(FallbackLIF())

        self.net = nn.Sequential(*layers)

        if HAS_SPIKINGJELLY:
            functional.set_step_mode(self.net, "m")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-step SNN.

        Parameters
        ----------
        x : torch.Tensor
            Spike tensor of shape (T, B, input_dim).

        Returns
        -------
        torch.Tensor
            Output spike trains of shape (T, B, num_classes).
        """
        return self.net(x)

    def firing_rate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean firing rate across the simulation time dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input spike tensor of shape (T, B, input_dim).

        Returns
        -------
        torch.Tensor
            Mean firing rate logits of shape (B, num_classes).
        """
        spikes = self.forward(x)
        return spikes.mean(dim=0)

    def reset(self) -> None:
        """Reset internal membrane potentials of all LIF neurons."""
        if HAS_SPIKINGJELLY:
            functional.reset_net(self.net)
