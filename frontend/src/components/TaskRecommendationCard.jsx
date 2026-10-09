import React, { useState } from 'react';
import { Brain, ArrowRight, Activity, ShieldCheck, Zap, CheckCircle2, Sparkles, TrendingUp, Flame } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function TaskRecommendationCard({ recommendation, currentState, nextState }) {
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [feedbackSuccess, setFeedbackSuccess] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const taskTitle = recommendation?.task || 'Practice Easy Problems';
  const situation = recommendation?.situation || (
    'High Beta wave elevation with suppressed Alpha waves and an elevated heart rate (98 BPM). Your neural signals indicate acute cognitive stress detected during high-load SNN cognitive monitoring.'
  );
  const reasoning = recommendation?.reasoning || (
    'Excessive cognitive strain reduces working memory capacity and accelerates burnout. Lowering task difficulty and initiating brief relaxation intervals protects cognitive health.'
  );
  const nextSteps = recommendation?.next_steps || [
    'Execute 3 minutes of 4-7-8 deep breathing to re-engage parasympathetic neural recovery.',
    'Switch to Tier 1/2 practice problems to consolidate retention without cognitive overload.',
    'Hydrate and step back from high-intensity problem solving for a 5-minute break.'
  ];
  const difficultyTag = recommendation?.difficulty_tag || (recommendation?.difficulty ? `Tier ${recommendation.difficulty} Load` : 'Tier 1 - Reduced Load');
  const state = recommendation?.cognitive_state || recommendation?.state || currentState || 'Neutral';

  // Badge styling depending on state
  const stateBadgeStyle = {
    'High Arousal': 'bg-amber-100 text-amber-900 border-amber-300 font-bold',
    'Low Arousal': 'bg-emerald-100 text-emerald-900 border-emerald-300 font-bold',
    Stressed: 'bg-amber-100 text-amber-900 border-amber-300 font-bold',
    Focused: 'bg-emerald-100 text-emerald-900 border-emerald-300 font-bold',
    Neutral: 'bg-blue-100 text-blue-900 border-blue-300 font-bold',
  }[state] || 'bg-blue-100 text-blue-900 border-blue-300 font-bold';

  const handleFeedback = async (rewardVal, label) => {
    try {
      setSubmittingFeedback(true);
      const targetNextState = nextState || currentState || state;
      const res = await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          state: state || 'Neutral',
          task_id: recommendation?.task_index ?? 0,
          reward: rewardVal,
          next_state: targetNextState,
        })
      });
      if (res.ok) {
        const body = await res.json();
        if (body.update) {
          setLastUpdate(body.update);
        }
        setFeedbackSuccess(`Bellman TD: ${label} (${rewardVal > 0 ? '+' : ''}${rewardVal})`);
        setTimeout(() => setFeedbackSuccess(null), 5000);
      }
    } catch (err) {
      console.warn("Feedback update failed:", err);
    } finally {
      setSubmittingFeedback(false);
    }
  };

  return (
    <div className="recommendation-card p-5 bg-[var(--bg-surface)] rounded-2xl border border-[var(--border-subtle)] shadow-sm hover:shadow-md transition-all duration-300">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-[var(--border-subtle)] mb-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-blue-50 text-[var(--accent-cyan)]">
            <Brain size={22} />
          </div>
          <div>
            <h3 className="font-heading font-bold text-base text-[var(--text-primary)]">Optimal Task Recommendation</h3>
            <p className="text-xs text-[var(--text-muted)]">Real-time SNN Cognitive Load Optimization</p>
          </div>
        </div>
        <span className={`px-2.5 py-1 text-xs font-semibold rounded-full border ${stateBadgeStyle}`}>
          {state}
        </span>
      </div>

      {/* Main Next Action Box */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-blue-950/20 via-blue-900/10 to-transparent border border-blue-500/20 mb-4">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[0.7rem] font-bold tracking-wider uppercase text-[var(--accent-cyan)] flex items-center gap-1">
            <Zap size={14} /> Recommended Next Action
          </span>
          <span className="text-[0.68rem] px-2 py-0.5 rounded-md bg-[var(--bg-card)] font-mono font-medium text-[var(--text-secondary)] border border-[var(--border-subtle)]">
            {difficultyTag}
          </span>
        </div>
        <div className="flex items-center gap-2 text-lg font-heading font-bold text-[var(--text-primary)]">
          <ArrowRight size={20} className="text-[var(--accent-cyan)] flex-shrink-0 animate-pulse" />
          <span>{taskTitle}</span>
        </div>
      </div>

      {/* Neural Situation Analysis */}
      <div className="mb-4">
        <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--text-secondary)] uppercase tracking-wider mb-1.5">
          <Activity size={14} className="text-[var(--accent-cyan)]" />
          <span>Neural Situation Analysis</span>
        </div>
        <p className="text-xs text-[var(--text-secondary)] leading-relaxed bg-[var(--bg-subtle)] p-3 rounded-lg border border-[var(--border-subtle)]">
          {situation}
        </p>
      </div>

      {/* Cognitive Health Action Plan */}
      <div className="mb-4">
        <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--text-secondary)] uppercase tracking-wider mb-2">
          <ShieldCheck size={14} className="text-[var(--accent-cyan)]" />
          <span>Cognitive Health Next Steps</span>
        </div>
        <div className="space-y-2">
          {nextSteps.map((step, idx) => (
            <div key={idx} className="flex items-start gap-2.5 text-xs text-[var(--text-primary)] bg-[var(--bg-subtle)]/50 p-2.5 rounded-lg border border-[var(--border-subtle)]/60">
              <CheckCircle2 size={15} className="text-[var(--accent-cyan)] flex-shrink-0 mt-0.5" />
              <span className="leading-snug">{step}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive Q-Learning Reinforcement Loop */}
      <div className="pt-3.5 border-t border-[var(--border-subtle)] mb-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[0.68rem] font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1">
            <Sparkles size={13} className="text-blue-600" />
            Rate Recommendation (Q-Learning Feedback)
          </span>
          {feedbackSuccess ? (
            <span className="text-[0.68rem] text-emerald-600 font-semibold animate-fade-in flex items-center gap-1">
              <CheckCircle2 size={12} /> {feedbackSuccess}
            </span>
          ) : (
            <span className="text-[0.62rem] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              Q(s, a) ← Q + α[r + γ max Q' - Q]
            </span>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2">
          <button
            onClick={() => handleFeedback(1.0, "Completed")}
            disabled={submittingFeedback}
            className="flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg border border-emerald-200 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 text-xs font-semibold transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50"
            title="Mark completed (+1.0 reward to Q-table)"
          >
            <CheckCircle2 size={14} className="text-emerald-600" />
            <span>Done (+1)</span>
          </button>
          <button
            onClick={() => handleFeedback(0.5, "Too Easy")}
            disabled={submittingFeedback}
            className="flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg border border-blue-200 bg-blue-50 hover:bg-blue-100 text-blue-800 text-xs font-semibold transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50"
            title="Request higher difficulty tier (+0.5 reward)"
          >
            <TrendingUp size={14} className="text-blue-600" />
            <span>Easy (+0.5)</span>
          </button>
          <button
            onClick={() => handleFeedback(-0.5, "Too Hard")}
            disabled={submittingFeedback}
            className="flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg border border-amber-200 bg-amber-50 hover:bg-amber-100 text-amber-800 text-xs font-semibold transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50"
            title="Request cognitive load reduction (-0.5 reward)"
          >
            <Flame size={14} className="text-amber-600" />
            <span>Hard (-0.5)</span>
          </button>
        </div>

        {/* Live Bellman Q-Learning Statistics */}
        {lastUpdate ? (
          <div className="mt-3 p-2.5 rounded-lg bg-blue-50/80 border border-blue-200/80 animate-fade-in text-xs font-mono">
            <div className="flex items-center justify-between text-blue-950 font-semibold mb-1 text-[0.72rem]">
              <span>Q({state}, a_{recommendation?.task_index ?? 0}) Update</span>
              <span className="px-1.5 py-0.5 rounded bg-blue-200/60 text-blue-800 text-[0.68rem]">
                Step #{lastUpdate.n_updates}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-[0.68rem] text-slate-700">
              <div>
                <span className="text-slate-500 block text-[0.62rem]">Q-VALUE</span>
                <span>{lastUpdate.old_q} → <strong className="text-blue-700">{lastUpdate.new_q}</strong></span>
              </div>
              <div>
                <span className="text-slate-500 block text-[0.62rem]">TD ERROR (δ)</span>
                <span className={lastUpdate.td_error >= 0 ? "text-emerald-700 font-bold" : "text-amber-700 font-bold"}>
                  {lastUpdate.td_error > 0 ? `+${lastUpdate.td_error}` : lastUpdate.td_error}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[0.62rem]">EXPLORATION (ε)</span>
                <span className="text-indigo-700 font-bold">{lastUpdate.epsilon}</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-2.5 p-2 rounded-lg bg-slate-50 border border-slate-200 text-[0.68rem] text-slate-500 flex items-center justify-between font-mono">
            <span>Tabular RL State: {state}</span>
            <span className="text-blue-600 font-semibold">Rate action above to trigger live Q-update</span>
          </div>
        )}
      </div>

      {/* SNN Adaptation Rationale */}
      <div className="pt-2 border-t border-[var(--border-subtle)]">
        <p className="text-[0.72rem] text-[var(--text-muted)] italic leading-relaxed">
          <strong className="not-italic text-[var(--text-secondary)]">Why this change: </strong>
          {reasoning}
        </p>
      </div>
    </div>
  );
}
