"""
Deep Neuromorphic Spiking Neuron Inspector.

Inspects learned synaptic weights, biases, cortical receptive fields,
and frequency band sensitivities for any individual neuron in the SNN.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch

from snn_ai_optimizer.features.eeg_features import BAND_ORDER, BANDS, N_FEATURES
from snn_ai_optimizer.snn.inference import CHANNEL_NAMES_10_20, get_snn_inference_engine

# Mapping 10-20 channels to anatomical cortical regions
CORTICAL_REGIONS = {
    "Frontal": ["Fp1", "FP2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC1", "FC2", "FC5", "FC6"],
    "Central": ["Cz", "C3", "C4"],
    "Temporal": ["T7", "T8"],
    "Parietal": ["Pz", "P3", "P4", "P7", "P8", "CP1", "CP2", "CP5", "CP6"],
    "Occipital": ["Oz", "O1", "O2", "PO3", "PO4"],
}


def _get_region_for_channel(channel: str) -> str:
    for region, chs in CORTICAL_REGIONS.items():
        if channel in chs:
            return region
    return "Central"


class SNNNeuronInspector:
    """Inspects internal SNN model parameters and translates them into neuro-computational insights."""

    def __init__(self) -> None:
        self.engine = get_snn_inference_engine()

    @property
    def model(self):
        return self.engine._model

    def explain_neuron(
        self,
        layer: str = "hidden1",
        neuron_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Explain the weights, receptive fields, and biological role of a specific neuron.

        Parameters
        ----------
        layer : str
            One of 'hidden1' (or 'lif1'), 'hidden2' (or 'lif2'), 'output'.
        neuron_index : int
            0-based index of the target neuron.

        Returns
        -------
        dict containing weights analysis, dominant band/region, and plain-language summary.
        """
        if self.model is None:
            return {
                "error": "Model checkpoint not loaded in memory.",
                "layer": layer,
                "neuron_index": neuron_index,
                "summary": "The SNN model checkpoint is currently unweighted or initializing.",
            }

        # Normalize layer key
        norm_layer = layer.lower().strip()
        if "1" in norm_layer or "hidden1" in norm_layer or "lif1" in norm_layer:
            layer_key = "hidden1"
            max_neurons = 256
        elif "2" in norm_layer or "hidden2" in norm_layer or "lif2" in norm_layer:
            layer_key = "hidden2"
            max_neurons = 128
        elif "out" in norm_layer:
            layer_key = "output"
            max_neurons = 2
        else:
            layer_key = "hidden1"
            max_neurons = 256

        # Zero-based clamp
        n_idx = max(0, min(neuron_index, max_neurons - 1))

        # Find linear weights corresponding to stages
        linear_layers = []
        for mod in self.model.net:
            if isinstance(mod, (torch.nn.Linear)):
                linear_layers.append(mod)

        if layer_key == "hidden1":
            return self._explain_hidden1_neuron(linear_layers[0], n_idx)
        elif layer_key == "hidden2":
            return self._explain_hidden2_neuron(linear_layers[1], n_idx)
        else:
            return self._explain_output_neuron(linear_layers[2], n_idx)

    def _explain_hidden1_neuron(self, linear_layer: torch.nn.Linear, n_idx: int) -> Dict[str, Any]:
        weight = linear_layer.weight[n_idx].detach().cpu().numpy()  # shape (128,)
        bias = float(linear_layer.bias[n_idx].detach().cpu().item())

        # Weight breakdown by 32 channels x 4 bands
        # index = ch * 4 + band_idx
        feat_data = []
        band_totals = {b: 0.0 for b in BAND_ORDER}
        region_totals = {r: 0.0 for r in CORTICAL_REGIONS}

        for ch_idx, ch_name in enumerate(CHANNEL_NAMES_10_20):
            region = _get_region_for_channel(ch_name)
            for b_idx, b_name in enumerate(BAND_ORDER):
                idx = ch_idx * len(BAND_ORDER) + b_idx
                w_val = float(weight[idx])
                feat_data.append({
                    "channel": ch_name,
                    "band": b_name,
                    "region": region,
                    "weight": round(w_val, 4),
                    "feature_name": f"{ch_name} {b_name.capitalize()}",
                })
                band_totals[b_name] += abs(w_val)
                region_totals[region] += abs(w_val)

        # Sort features by absolute weight
        sorted_feats = sorted(feat_data, key=lambda x: abs(x["weight"]), reverse=True)
        top_excitatory = [f for f in sorted_feats if f["weight"] > 0][:5]
        top_inhibitory = [f for f in sorted_feats if f["weight"] < 0][:5]

        dominant_band = max(band_totals.items(), key=lambda x: x[1])[0]
        dominant_region = max(region_totals.items(), key=lambda x: x[1])[0]

        # Interpret neuro-computational role
        band_meanings = {
            "beta": "active analytical cognition, problem-solving, and elevated arousal",
            "alpha": "calm alertness, steady engagement, and sensory suppression",
            "theta": "working memory consolidation, creativity, or cognitive fatigue",
            "delta": "deep parasympathetic recovery or resting baseline",
        }
        role_desc = band_meanings.get(dominant_band, "neural oscillatory synchronization")

        formatted = (
            f"### Neuron {n_idx + 1} (Hidden LIF Layer 1)\n"
            f"* **Architecture:** Leaky Integrate-and-Fire unit with $\\tau = 2.0$, threshold $V_{{\\text{{th}}}} = 1.0\\text{{ V}}$, bias $b = {bias:.4f}$.\n"
            f"* **Dominant Frequency Band:** **{dominant_band.capitalize()} band** ({BANDS[dominant_band][0]}–{BANDS[dominant_band][1]} Hz), sensitive to {role_desc}.\n"
            f"* **Primary Cortical Region:** **{dominant_region} lobe**, driven heavily by electrode sites like "
            f"{', '.join([f['channel'] for f in top_excitatory[:3]]) or 'diffuse sensors'}.\n"
            f"* **Top Excitatory Inputs:** "
            + ", ".join([f"`{f['feature_name']}` (+{f['weight']:.3f})" for f in top_excitatory[:3]])
            + ".\n"
            f"* **Top Inhibitory Inputs:** "
            + (", ".join([f"`{f['feature_name']}` ({f['weight']:.3f})" for f in top_inhibitory[:3]]) if top_inhibitory else "None (predominantly excitatory).")
            + "\n"
            f"* **Functional Role:** When {dominant_band} oscillations surge in the {dominant_region.lower()} cortex, "
            f"Neuron {n_idx + 1} charges rapidly toward its $1.0\\text{{ V}}$ threshold, emitting temporal action potentials "
            f"that signal elevated cognitive activation to Hidden Layer 2."
        )

        return {
            "layer": "Hidden LIF Layer 1",
            "neuron_number": n_idx + 1,
            "neuron_index": n_idx,
            "bias": bias,
            "dominant_band": dominant_band,
            "dominant_region": dominant_region,
            "top_excitatory": top_excitatory,
            "top_inhibitory": top_inhibitory,
            "summary": formatted,
        }

    def _explain_hidden2_neuron(self, linear_layer: torch.nn.Linear, n_idx: int) -> Dict[str, Any]:
        weight = linear_layer.weight[n_idx].detach().cpu().numpy()  # shape (256,)
        bias = float(linear_layer.bias[n_idx].detach().cpu().item())

        top_excitatory_idx = np.argsort(weight)[-4:][::-1]
        top_inhibitory_idx = np.argsort(weight)[:4]

        top_pos = [{"neuron": int(i + 1), "weight": round(float(weight[i]), 4)} for i in top_excitatory_idx if weight[i] > 0]
        top_neg = [{"neuron": int(i + 1), "weight": round(float(weight[i]), 4)} for i in top_inhibitory_idx if weight[i] < 0]

        formatted = (
            f"### Neuron {n_idx + 1} (Hidden LIF Layer 2)\n"
            f"* **Architecture:** Second-tier integrator with 256 upstream LIF synapses, bias $b = {bias:.4f}$, $\\tau = 2.0$.\n"
            f"* **Primary Upstream Drivers (Layer 1):** "
            + ", ".join([f"Neuron {p['neuron']} (+{p['weight']:.3f})" for p in top_pos[:3]])
            + ".\n"
            f"* **Functional Role:** Acts as an intermediate cortical pattern synthesizer, integrating raw temporal "
            f"spikes from Hidden Layer 1 into higher-level representations for final Arousal readout."
        )

        return {
            "layer": "Hidden LIF Layer 2",
            "neuron_number": n_idx + 1,
            "neuron_index": n_idx,
            "bias": bias,
            "top_excitatory_upstream": top_pos,
            "top_inhibitory_upstream": top_neg,
            "summary": formatted,
        }

    def _explain_output_neuron(self, linear_layer: torch.nn.Linear, n_idx: int) -> Dict[str, Any]:
        weight = linear_layer.weight[n_idx].detach().cpu().numpy()  # shape (128,)
        bias = float(linear_layer.bias[n_idx].detach().cpu().item())
        class_name = "Low Arousal (<= 5.0)" if n_idx == 0 else "High Arousal (> 5.0)"

        formatted = (
            f"### Readout Unit {n_idx} — {class_name}\n"
            f"* **Architecture:** Output classification node with bias $b = {bias:.4f}$.\n"
            f"* **Readout Mechanism:** Accumulates spike trains from Hidden Layer 2 over $T = 50$ time steps. "
            f"Its mean temporal firing rate $\\bar{{S}}_{{{n_idx}}}$ produces the softmax logit for classifying "
            f"the user's affective state as {class_name}."
        )

        return {
            "layer": "Output Readout Layer",
            "class_name": class_name,
            "neuron_index": n_idx,
            "bias": bias,
            "summary": formatted,
        }
