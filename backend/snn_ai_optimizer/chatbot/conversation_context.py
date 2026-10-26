"""
Multi-Turn Conversation Context Manager for SNN Assistant.

Maintains and resolves:
- Active neuron index and user-facing neuron number.
- Active network layer (hidden1, hidden2, output).
- Active simulation time step (t = 23, etc.).
- Active graph and visualization target.
- Pronoun and anaphora resolution ("the green line", "it", "at this time", "why did it drop").
- Context isolation preventing historical targets from overriding explicit new questions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class DialogueContext:
    """Encapsulates the resolved conversational context for the current turn."""

    def __init__(
        self,
        active_layer: str = "hidden1",
        active_neuron_index: int = 3,  # Defaults to index 3 (Neuron 4) as initial focal unit
        active_neuron_number: int = 4,
        active_time_step: Optional[int] = 23,
        active_graph: str = "membrane_potential_trajectory",
        referenced_pronouns: bool = False,
        recent_topics: Optional[List[str]] = None,
    ) -> None:
        self.active_layer = active_layer
        self.active_neuron_index = active_neuron_index
        self.active_neuron_number = active_neuron_number
        self.active_time_step = active_time_step
        self.active_graph = active_graph
        self.referenced_pronouns = referenced_pronouns
        self.recent_topics = recent_topics or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_layer": self.active_layer,
            "active_neuron_index": self.active_neuron_index,
            "active_neuron_number": self.active_neuron_number,
            "active_time_step": self.active_time_step,
            "active_graph": self.active_graph,
            "referenced_pronouns": self.referenced_pronouns,
            "recent_topics": self.recent_topics,
        }


class ConversationContextManager:
    """Analyzes message history and client state to extract active entities."""

    @staticmethod
    def resolve_context(
        current_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        client_state: Optional[Dict[str, Any]] = None,
    ) -> DialogueContext:
        """
        Resolves the active neuron, layer, and time step from current query and past turns.
        """
        msg_lower = current_message.lower()
        hist = history or []
        c_state = client_state or {}

        # Default fallbacks from client state if present
        def_layer = c_state.get("activeLayer", "hidden1")
        def_neuron_idx = int(c_state.get("selectedNeuronIndex", 3))  # index 3 = Neuron 4
        def_neuron_num = def_neuron_idx + 1

        resolved_layer = def_layer
        resolved_neuron_idx = def_neuron_idx
        resolved_neuron_num = def_neuron_num
        resolved_time_step: Optional[int] = None
        has_pronoun = False
        recent_topics: List[str] = []

        # 1. Check if the current message explicitly mentions a neuron
        neuron_pattern = re.search(r"(?:neuron|unit|node)\s*#?\s*(\d+)", msg_lower)
        if neuron_pattern:
            num = int(neuron_pattern.group(1))
            resolved_neuron_num = num
            resolved_neuron_idx = max(0, num - 1)
        else:
            # Check history (most recent to oldest)
            for turn in reversed(hist):
                turn_text = str(turn.get("text", "") or turn.get("content", "")).lower()
                hist_match = re.search(r"(?:neuron|unit|node)\s*#?\s*(\d+)", turn_text)
                if hist_match:
                    num = int(hist_match.group(1))
                    resolved_neuron_num = num
                    resolved_neuron_idx = max(0, num - 1)
                    break

        # 2. Check if current message explicitly mentions a layer
        if any(k in msg_lower for k in ("hidden 2", "hidden lif 2", "lif 2", "layer 2", "lif2")):
            resolved_layer = "hidden2"
        elif any(k in msg_lower for k in ("output", "readout", "arousal layer", "layer 3", "lif 3")):
            resolved_layer = "output"
        elif any(k in msg_lower for k in ("hidden 1", "hidden lif 1", "lif 1", "layer 1", "lif1")):
            resolved_layer = "hidden1"
        else:
            # Check history
            for turn in reversed(hist):
                turn_text = str(turn.get("text", "") or turn.get("content", "")).lower()
                if any(k in turn_text for k in ("hidden 2", "hidden lif 2", "lif 2", "layer 2")):
                    resolved_layer = "hidden2"
                    break
                elif any(k in turn_text for k in ("output", "readout", "arousal layer")):
                    resolved_layer = "output"
                    break
                elif any(k in turn_text for k in ("hidden 1", "hidden lif 1", "lif 1")):
                    resolved_layer = "hidden1"
                    break

        # 3. Check if current message specifies time step t
        time_pattern = re.search(r"\bt\s*=\s*(\d+)\b|\bstep\s*#?\s*(\d+)\b|\btime\s*step\s*(\d+)\b|\bat\s+(\d+)\s*(?:ms|steps?)\b", msg_lower)
        if time_pattern:
            for g in time_pattern.groups():
                if g is not None:
                    resolved_time_step = int(g)
                    break
        else:
            # Check history if question refers to "this time", "that point", etc.
            if any(k in msg_lower for k in ("this time", "that time", "this point", "that step", "at this moment")):
                for turn in reversed(hist):
                    turn_text = str(turn.get("text", "") or turn.get("content", "")).lower()
                    h_time = re.search(r"\bt\s*=\s*(\d+)\b|\bstep\s*#?\s*(\d+)\b", turn_text)
                    if h_time:
                        for g in h_time.groups():
                            if g is not None:
                                resolved_time_step = int(g)
                                break
                        if resolved_time_step is not None:
                            break

        # 4. Check for pronoun / elliptical references
        if any(w in msg_lower.split() for w in ("it", "its", "this", "that", "the")) or "green line" in msg_lower:
            has_pronoun = True

        return DialogueContext(
            active_layer=resolved_layer,
            active_neuron_index=resolved_neuron_idx,
            active_neuron_number=resolved_neuron_num,
            active_time_step=resolved_time_step,
            active_graph="membrane_potential_trajectory",
            referenced_pronouns=has_pronoun,
            recent_topics=recent_topics,
        )
