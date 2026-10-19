import numpy as np
import os
import json

class SNNRecommender:
    """
    Pure-Numpy Leaky Integrate-and-Fire (LIF) Spiking Neural Network for Task Recommendation.
    Takes cognitive state as input and outputs a difficulty band (1-5).
    """
    def __init__(self, input_size=3, hidden_size=16, output_size=5, time_steps=20):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.time_steps = time_steps
        
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        
        # LIF parameters
        self.tau = 0.8
        self.v_thresh = 1.0
        self.v_reset = 0.0

    def forward(self, x):
        """
        x: input spike train of shape [time_steps, input_size]
        Returns output spikes of shape [time_steps, output_size]
        """
        # Initialize membrane potentials
        v1 = np.zeros(self.hidden_size)
        v2 = np.zeros(self.output_size)
        
        output_spikes = np.zeros((self.time_steps, self.output_size))
        
        for t in range(self.time_steps):
            # Layer 1
            i1 = np.dot(x[t], self.W1)
            v1 = self.tau * v1 + i1
            s1 = (v1 >= self.v_thresh).astype(float)
            v1[v1 >= self.v_thresh] = self.v_reset
            
            # Layer 2
            i2 = np.dot(s1, self.W2)
            v2 = self.tau * v2 + i2
            s2 = (v2 >= self.v_thresh).astype(float)
            v2[v2 >= self.v_thresh] = self.v_reset
            
            output_spikes[t] = s2
            
        return output_spikes

    def predict(self, state: str) -> int:
        """
        Predict a difficulty band (1–5) from cognitive state via spike-count readout.

        The network maps one-hot state input (Focused / Neutral / Stressed) to an
        output neuron through two layers of LIF dynamics. The neuron with the highest
        cumulative spike count wins. No hardcoded bias is injected — the output is
        determined entirely by weight-induced dynamics.

        Parameters
        ----------
        state : str — "Focused", "Neutral", or "Stressed".

        Returns
        -------
        int : Difficulty level 1–5.
        """
        state_map = {"Focused": 0, "Neutral": 1, "Stressed": 2}
        idx = state_map.get(state, 1)

        # One-hot constant-rate spike input
        x = np.zeros((self.time_steps, self.input_size))
        x[:, idx] = 1.0

        out_spikes = self.forward(x)
        spike_counts = out_spikes.sum(axis=0)

        # No additive bias — network weights determine outcome
        best_idx = int(np.argmax(spike_counts))
        return best_idx + 1  # Difficulty 1–5

    def save(self, path="results/snn/recommender_model.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, W1=self.W1, W2=self.W2)

    def load(self, path="results/snn/recommender_model.npz"):
        if os.path.exists(path):
            data = np.load(path)
            self.W1 = data['W1']
            self.W2 = data['W2']
            return True
        return False
