"""
Data and Knowledge Access Layer for SNN Chatbot.

Provides verified, real-time access to:
1. Live Neuromorphic Traces: continuous V(t) trajectories, spike trains S(t),
   thresholds, reset potentials, and layer averages.
2. Graph Legends & Series: mappings of plotted lines, colors (#059669 emerald,
   #dc2626 red threshold, #64748b reset, #94a3b8 mean), axes, and units.
3. Learned Model Parameters & Synaptic Weights: via SNNNeuronInspector.
4. Empirical Benchmark Provenance: results/snn/experiment.json and metrics.json
   (real AUROC 0.5532 vs SVM 0.4834, DEAP subject splits).
5. Closed-loop Q-Learning & Clinical Appointments Storage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from snn_ai_optimizer.chatbot.appointment_service import AppointmentService
from snn_ai_optimizer.chatbot.neuron_inspector import SNNNeuronInspector
from snn_ai_optimizer.snn.api import project_scalars_to_128_features
from snn_ai_optimizer.snn.inference import get_snn_inference_engine

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EXPERIMENT_FILE = _REPO_ROOT / "results" / "snn" / "experiment.json"
_METRICS_FILE = _REPO_ROOT / "results" / "snn" / "metrics.json"
_BENCHMARK_REPORT_FILE = _REPO_ROOT / "results" / "benchmark_report.json"
_Q_TABLE_FILE = _REPO_ROOT / "results" / "q_table.json"

# Visualizer plotting metadata matching SNNSpikeRasterVisualizer.jsx
GRAPH_METADATA = {
    "membrane_potential_trajectory": {
        "title": "Membrane Potential Trajectory V(t)",
        "component": "SNNSpikeRasterVisualizer.jsx",
        "description": "Visualizes continuous leaky charge accumulation, threshold firing events (1.0 V), and instant refractory reset (0.0 V).",
        "x_axis": {
            "name": "Simulation Step (t)",
            "type": "discrete integer steps",
            "range": [0, 49],
            "total_steps": 50,
            "unit": "simulation step (t)",
        },
        "y_axis": {
            "name": "Voltage (V)",
            "type": "continuous membrane potential",
            "range": [-0.1, 1.2],
            "ticks": [0.0, 0.25, 0.5, 0.75, 1.0],
            "unit": "volts (V)",
        },
        "lines": {
            "green_line": {
                "color": "#059669",
                "color_name": "emerald green",
                "stroke_width": 2.4,
                "data_key": "v_selected",
                "name_template": "{Neuron Label} V(t)",
                "description": "The continuous membrane potential trajectory V(t) of the currently selected neuron across the 50 simulation steps.",
            },
            "red_dashed_line": {
                "color": "#dc2626",
                "color_name": "crimson red",
                "style": "dashed (strokeDasharray 4 4)",
                "stroke_width": 1.5,
                "y_value": 1.0,
                "name": "Firing Threshold (V_th)",
                "description": "Biological threshold potential (1.0 V). When V(t) reaches or exceeds this boundary, the neuron emits a spike and resets.",
            },
            "grey_dashed_line_reset": {
                "color": "#64748b",
                "color_name": "slate grey",
                "style": "dashed (strokeDasharray 3 3)",
                "y_value": 0.0,
                "name": "Rest / Reset Potential (V_reset)",
                "description": "Baseline refractory potential (0.0 V). Immediately following an action potential, the membrane potential resets to this level.",
            },
            "grey_dashed_line_mean": {
                "color": "#94a3b8",
                "color_name": "light slate grey",
                "style": "dashed (strokeDasharray 3 3)",
                "stroke_width": 1.2,
                "data_key": "v_mean",
                "name": "Layer Mean V(t)",
                "description": "Average membrane potential across all sampled neurons in the currently viewed layer at each time step.",
            },
        },
    }
}


class DataAccessLayer:
    """Provides ground-truth data retrieval for the SNN conversational assistant."""

    def __init__(self) -> None:
        self.engine = get_snn_inference_engine()
        self.inspector = SNNNeuronInspector()
        self.appointments = AppointmentService()
        self._cached_traces: Optional[Dict[str, Any]] = None
        self._traces_params: Tuple[float, float, float] = (-1.0, -1.0, -1.0)

    def get_simulation_traces(
        self,
        alpha: float = 0.65,
        beta: float = 0.72,
        lf_hf: float = 1.15,
        time_steps: int = 50,
        n_neurons: int = 16,
    ) -> Dict[str, Any]:
        """
        Retrieve live neuromorphic traces (V(t) trajectories and S(t) spike trains).
        Caches traces when inputs have not changed to ensure consistency with the frontend.
        """
        cache_key = (round(alpha, 3), round(beta, 3), round(lf_hf, 3))
        if self._cached_traces is not None and self._traces_params == cache_key:
            return self._cached_traces

        feats = project_scalars_to_128_features(alpha=alpha, beta=beta, lf_hf=lf_hf)
        traces = self.engine.get_neuromorphic_traces(
            features=feats,
            time_steps=time_steps,
            n_sampled_neurons=n_neurons,
        )
        self._cached_traces = traces
        self._traces_params = cache_key
        return traces

    def get_neuron_trajectory_point(
        self,
        layer: str,
        neuron_index: int,
        time_step: int,
        alpha: float = 0.65,
        beta: float = 0.72,
    ) -> Dict[str, Any]:
        """
        Inspect exact membrane potential and spike status for a neuron at step t.
        """
        norm_layer = "hidden1"
        if "2" in layer:
            norm_layer = "hidden2"
        elif "out" in layer:
            norm_layer = "output"

        traces = self.get_simulation_traces(alpha=alpha, beta=beta)
        layers_data = traces.get("layers", {})
        layer_dict = layers_data.get(norm_layer, {})

        v_matrix = layer_dict.get("v", [])
        s_matrix = layer_dict.get("spikes", [])

        total_steps = len(v_matrix)
        if total_steps == 0:
            return {
                "error": "No simulation traces available.",
                "layer": norm_layer,
                "neuron_index": neuron_index,
                "time_step": time_step,
            }

        # Check step range
        if time_step < 0 or time_step >= total_steps:
            return {
                "error": f"Time step t = {time_step} is out of bounds (valid range is 0 to {total_steps - 1}).",
                "valid_range": [0, total_steps - 1],
                "layer": norm_layer,
                "neuron_index": neuron_index,
                "time_step": time_step,
            }

        # Clamp neuron index safely to sampled range
        n_sampled = len(v_matrix[0]) if len(v_matrix) > 0 else 1
        safe_n_idx = max(0, min(neuron_index, n_sampled - 1))

        voltage = float(v_matrix[time_step][safe_n_idx])
        spiked = bool(s_matrix[time_step][safe_n_idx]) if s_matrix else False
        v_th = float(traces.get("v_threshold", 1.0))
        v_reset = float(traces.get("v_reset", 0.0))

        # Check all firing steps across T for this neuron
        firing_steps = [
            t for t in range(total_steps)
            if s_matrix and len(s_matrix[t]) > safe_n_idx and s_matrix[t][safe_n_idx] == 1
        ]

        # Nearby trajectory context
        start_t = max(0, time_step - 2)
        end_t = min(total_steps, time_step + 3)
        trajectory_slice = [
            {"step": t, "voltage": round(float(v_matrix[t][safe_n_idx]), 4), "spiked": bool(s_matrix[t][safe_n_idx])}
            for t in range(start_t, end_t)
        ]

        return {
            "layer": norm_layer,
            "neuron_index": safe_n_idx,
            "neuron_number": safe_n_idx + 1,
            "time_step": time_step,
            "total_steps": total_steps,
            "voltage": round(voltage, 4),
            "spiked": spiked,
            "v_threshold": v_th,
            "v_reset": v_reset,
            "distance_to_threshold": round(v_th - voltage, 4),
            "is_subthreshold": voltage < v_th,
            "firing_steps": firing_steps,
            "trajectory_slice": trajectory_slice,
        }

    def get_graph_metadata(self, graph_key: str = "membrane_potential_trajectory") -> Dict[str, Any]:
        """Returns plotting metadata for visualizer graphs."""
        return GRAPH_METADATA.get(graph_key, GRAPH_METADATA["membrane_potential_trajectory"])

    def get_experiment_provenance(self) -> Dict[str, Any]:
        """Reads ground-truth empirical experiment configuration and evaluation metrics."""
        data: Dict[str, Any] = {}
        if _EXPERIMENT_FILE.exists():
            try:
                data = json.loads(_EXPERIMENT_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

        metrics_data: Dict[str, Any] = {}
        if _METRICS_FILE.exists():
            try:
                metrics_data = json.loads(_METRICS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

        benchmark_report: Dict[str, Any] = {}
        if _BENCHMARK_REPORT_FILE.exists():
            try:
                benchmark_report = json.loads(_BENCHMARK_REPORT_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

        return {
            "experiment": data,
            "metrics": metrics_data,
            "benchmarks": benchmark_report.get("benchmarks", {}),
        }

    def get_q_learning_status(self) -> Dict[str, Any]:
        """Loads active Q-table parameters and state action values."""
        if _Q_TABLE_FILE.exists():
            try:
                return json.loads(_Q_TABLE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "states": ["Stressed", "Focused", "Fatigued", "Neutral"],
            "actions": [0, 1, 2, 3, 4],
            "learning_rate": 0.1,
            "discount_factor": 0.9,
            "epsilon": 0.05,
        }
