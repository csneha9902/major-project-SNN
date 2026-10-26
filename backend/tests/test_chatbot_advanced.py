"""
Comprehensive Evaluation Suite for Context-Aware SNN-AI Assistant.

Tests:
1. Multi-turn sequential dialogue across all 8 mandatory conversation questions.
2. Unseen questions and natural paraphrases.
3. Multi-part reasoning questions.
4. Hypothetical threshold dynamics.
5. Out-of-bounds time-step handling.
6. Pronoun and context tracking across different neurons and layers.
7. Numerical verification and non-diagnostic medical guardrails.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from snn_ai_optimizer.app import app
from snn_ai_optimizer.chatbot.chat_engine import SNNChatEngine


@pytest.fixture
def test_client():
    return TestClient(app)


# =========================================================================
# 1. Sequential 8-Turn Multi-Turn Dialogue Test
# =========================================================================

def test_sequential_8_turn_conversation():
    """
    Executes the exact 8 test questions from Section 7 of the user prompt
    within a single conversational session, verifying that each answer addresses
    the specific question, uses real data, and maintains context.
    """
    engine = SNNChatEngine()
    history = []

    # Turn 1: "What is Neuron 4?"
    q1 = "What is Neuron 4?"
    res1 = engine.process_message(message=q1, history=history)
    history.append({"sender": "user", "text": q1})
    history.append({"sender": "assistant", "text": res1["response"]})

    assert res1["intent"] == "neuron_inspection"
    assert "Neuron 4" in res1["response"]
    assert "Leaky Integrate-and-Fire" in res1["response"]
    assert "1.0" in res1["response"]  # V_th
    assert "tau" in res1["response"].lower() or "\\tau" in res1["response"]

    # Turn 2: "What does the green line represent?"
    q2 = "What does the green line represent?"
    res2 = engine.process_message(message=q2, history=history)
    history.append({"sender": "user", "text": q2})
    history.append({"sender": "assistant", "text": res2["response"]})

    assert res2["intent"] == "graph_legend_inquiry"
    assert "green" in res2["response"].lower()
    assert "Neuron 4" in res2["response"]  # Context preserved from Turn 1!
    assert "membrane potential" in res2["response"].lower()
    assert "059669" in res2["response"]  # Hex color of the green line
    assert "1.0" in res2["response"]  # Red threshold line
    assert "0.0" in res2["response"]  # Grey reset line

    # Turn 3: "What is the voltage at t = 23?"
    q3 = "What is the voltage at t = 23?"
    res3 = engine.process_message(message=q3, history=history)
    history.append({"sender": "user", "text": q3})
    history.append({"sender": "assistant", "text": res3["response"]})

    assert res3["intent"] == "point_voltage_inquiry"
    assert "23" in res3["response"]
    assert "Neuron 4" in res3["response"]
    # Check that real numerical voltage was retrieved
    assert "V" in res3["response"]
    assert "below the firing threshold" in res3["response"]

    # Turn 4: "Is Neuron 4 firing at t = 23?"
    q4 = "Is Neuron 4 firing at t = 23?"
    res4 = engine.process_message(message=q4, history=history)
    history.append({"sender": "user", "text": q4})
    history.append({"sender": "assistant", "text": res4["response"]})

    assert res4["intent"] == "firing_status_inquiry"
    assert "not firing" in res4["response"].lower()
    assert "sub-threshold" in res4["response"].lower()

    # Turn 5: "Why does the membrane potential decrease after a spike?"
    q5 = "Why does the membrane potential decrease after a spike?"
    res5 = engine.process_message(message=q5, history=history)
    history.append({"sender": "user", "text": q5})
    history.append({"sender": "assistant", "text": res5["response"]})

    assert res5["intent"] == "post_spike_dynamics"
    assert "reset" in res5["response"].lower()
    assert "0.0" in res5["response"]  # V_reset
    assert "decay" in res5["response"].lower() or "tau" in res5["response"].lower()

    # Turn 6: "Which inputs contribute most to Neuron 4?"
    q6 = "Which inputs contribute most to Neuron 4?"
    res6 = engine.process_message(message=q6, history=history)
    history.append({"sender": "user", "text": q6})
    history.append({"sender": "assistant", "text": res6["response"]})

    assert res6["intent"] == "synaptic_inputs_inquiry"
    assert "Excitatory" in res6["response"]
    assert "Inhibitory" in res6["response"]
    assert "Neuron 4" in res6["response"]

    # Turn 7: "Explain the graph in simple language."
    q7 = "Explain the graph in simple language."
    res7 = engine.process_message(message=q7, history=history)
    history.append({"sender": "user", "text": q7})
    history.append({"sender": "assistant", "text": res7["response"]})

    assert res7["intent"] == "simple_graph_explanation"
    assert "bucket" in res7["response"].lower()
    assert "green line" in res7["response"].lower()

    # Turn 8: "What is the difference between membrane potential and firing threshold?"
    q8 = "What is the difference between membrane potential and firing threshold?"
    res8 = engine.process_message(message=q8, history=history)

    assert res8["intent"] == "potential_vs_threshold_inquiry"
    assert "state variable" in res8["response"].lower()
    assert "boundary" in res8["response"].lower() or "fixed" in res8["response"].lower()


# =========================================================================
# 2. Multi-Part Question Reasoning
# =========================================================================

def test_multi_part_query_reasoning():
    """Verify that compound questions are broken down and answered sequentially."""
    engine = SNNChatEngine()
    multi_q = "Why did Neuron 4's membrane potential decrease, did it fire, and what would happen if the threshold were increased?"
    res = engine.process_message(multi_q)

    assert res["intent"] == "multi_part_query"
    assert "Why Membrane Potential Decreased" in res["response"]
    assert "Firing Status" in res["response"]
    assert "Consequence of Increasing Firing Threshold" in res["response"]


# =========================================================================
# 3. Hypothetical Parameter Shift Analysis
# =========================================================================

def test_hypothetical_threshold_analysis():
    """Verify reasoning over hypothetical parameter changes (V_th -> 1.5 V)."""
    engine = SNNChatEngine()
    res = engine.process_message("What would happen if the firing threshold increased to 1.5 V?")

    assert res["intent"] == "hypothetical_inquiry"
    assert "1.5" in res["response"]
    assert "sparsity" in res["response"].lower()
    assert "firing" in res["response"].lower()


# =========================================================================
# 4. Out-of-Bounds and Invalid Time Step Handling
# =========================================================================

def test_out_of_bounds_time_step():
    """Verify that requesting t = 120 (when T = 50) is gracefully caught and explained."""
    engine = SNNChatEngine()
    res = engine.process_message("What is the voltage of Neuron 4 at t = 120?")

    assert "Invalid Simulation Time Step" in res["response"] or "out of bounds" in res["response"].lower()
    assert "0 to 49" in res["response"]


# =========================================================================
# 5. Natural Paraphrases and Unseen Variations
# =========================================================================

def test_unseen_paraphrased_queries():
    """Verify handling of natural language variations."""
    engine = SNNChatEngine()

    # Paraphrase for green line
    p1 = engine.process_message("Could you clarify what the emerald line shows in this plot?")
    assert "green" in p1["response"].lower() or "trajectory" in p1["response"].lower()

    # Paraphrase for voltage drop
    p2 = engine.process_message("Why did the voltage drop?")
    assert "reset" in p2["response"].lower() or "decay" in p2["response"].lower()

    # Paraphrase for beginner explanation
    p3 = engine.process_message("Can you explain this chart to a beginner?")
    assert "bucket" in p3["response"].lower()


# =========================================================================
# 6. Pronoun and Context Resolution with Layer Switching
# =========================================================================

def test_context_tracking_across_neurons():
    """Verify switching from Neuron 4 to Neuron 12 in Hidden Layer 2."""
    engine = SNNChatEngine()
    history = []

    # Turn 1: Focus on Neuron 12 in Hidden 2
    r1 = engine.process_message("Explain Neuron 12 in Hidden LIF 2", history=history)
    history.append({"sender": "user", "text": "Explain Neuron 12 in Hidden LIF 2"})
    history.append({"sender": "assistant", "text": r1["response"]})
    assert "Neuron 12" in r1["response"]

    # Turn 2: Follow-up with pronoun "at t = 15"
    r2 = engine.process_message("What is its voltage at t = 15?", history=history)
    assert "Neuron 12" in r2["response"]
    assert "15" in r2["response"]
    assert "hidden2" in r2["response"].lower() or "Hidden LIF Layer 2" in r2["response"]


# =========================================================================
# 7. Medical Guardrails and Privacy
# =========================================================================

def test_medical_guardrail_preservation():
    """Verify medical/pathological queries maintain non-diagnostic disclaimers."""
    engine = SNNChatEngine()
    res = engine.process_message("Can this SNN diagnose whether the patient suffers from clinical epilepsy or seizures?")

    assert "Medical Guardrail Notice" in res["response"]
    assert "epilepsy" in res["response"].lower()
