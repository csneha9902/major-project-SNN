import { Activity, Brain, Zap } from 'lucide-react';

const stateConfig = {
  'High Arousal': {
    label: 'High Arousal (SNN)',
    description: 'Elevated Cognitive Activation (Arousal > 5.0)',
    icon: Zap,
    gradient: 'from-amber-600/15 to-red-500/10',
    textColor: 'text-amber-900',
    glowColor: 'rgba(217, 119, 6, 0.2)',
    iconGlow: 'drop-shadow(0 0 8px rgba(217,119,6,0.6))',
    dotColor: '#D97706',
  },
  'Low Arousal': {
    label: 'Low Arousal (SNN)',
    description: 'Baseline / Relaxed State (Arousal <= 5.0)',
    icon: Brain,
    gradient: 'from-emerald-600/20 to-blue-600/15',
    textColor: 'text-emerald-900',
    glowColor: 'rgba(16, 185, 129, 0.25)',
    iconGlow: 'drop-shadow(0 0 8px rgba(16,185,129,0.6))',
    dotColor: '#10B981',
  },
  Stressed: {
    label: 'High Arousal (High Cognitive Load)',
    description: 'Elevated Beta Synchrony (Arousal > 5.0)',
    icon: Zap,
    gradient: 'from-amber-600/15 to-red-500/10',
    textColor: 'text-amber-900',
    glowColor: 'rgba(217, 119, 6, 0.2)',
    iconGlow: 'drop-shadow(0 0 8px rgba(217,119,6,0.6))',
    dotColor: '#D97706',
  },
  Focused: {
    label: 'Low Arousal (Steady Engagement)',
    description: 'Alpha-Synchronized Focus (Arousal <= 5.0)',
    icon: Brain,
    gradient: 'from-emerald-600/20 to-blue-600/15',
    textColor: 'text-emerald-900',
    glowColor: 'rgba(16, 185, 129, 0.25)',
    iconGlow: 'drop-shadow(0 0 8px rgba(16,185,129,0.6))',
    dotColor: '#10B981',
  },
  Neutral: {
    label: 'Calibrated Baseline (SNN)',
    description: 'Resting EEG State (Arousal ~ 5.0)',
    icon: Activity,
    gradient: 'from-blue-600/15 to-emerald-500/10',
    textColor: 'text-blue-900',
    glowColor: 'rgba(59, 130, 246, 0.15)',
    iconGlow: 'drop-shadow(0 0 6px rgba(59,130,246,0.5))',
    dotColor: '#3B82F6',
  },
};

export default function CurrentStateCard({ state }) {
  const config = stateConfig[state] || stateConfig.Neutral;
  const Icon = config.icon;

  return (
    <div
      className={`state-card bg-gradient-to-r ${config.gradient} ${config.textColor}`}
      style={{ boxShadow: `var(--shadow-card), 0 0 30px ${config.glowColor}` }}
    >
      <Icon className="state-icon" style={{ filter: config.iconGlow }} />
      <div className="flex flex-col">
        <span className="font-heading font-bold text-sm">{config.label}</span>
        <span className="text-[0.68rem] text-slate-600 font-mono">{config.description}</span>
      </div>
      <span className="live-dot ml-auto" style={{
        background: config.dotColor,
      }} />
    </div>
  );
}
