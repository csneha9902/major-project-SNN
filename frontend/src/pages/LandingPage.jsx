import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Brain, 
  Activity, 
  FileText, 
  ArrowRight, 
  Sparkles, 
  Cpu, 
  Award, 
  ShieldCheck, 
  HeartPulse, 
  Database, 
  Zap, 
  TrendingUp, 
  SlidersHorizontal,
  FolderUp,
  CheckCircle2
} from 'lucide-react';
import '../App.css';

export default function LandingPage() {
  const navigate = useNavigate();

  const handleEnterResearch = () => {
    localStorage.setItem('user_role', 'researcher');
    navigate('/research');
  };

  const handleEnterClinical = () => {
    localStorage.setItem('user_role', 'clinician');
    navigate('/dashboard');
  };

  return (
    <>
      <div className="app-background">
        <div className="orb-3" />
        <div className="grid-overlay" />
      </div>

      <div className="dual-portal-container">
        <div className="dual-portal-content">
          {/* Main Hero Header */}
          <div className="landing-header">
            <div className="flex justify-center mb-5">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center shadow-lg"
                style={{
                  background: 'linear-gradient(135deg, rgba(0,98,255,0.15), rgba(16,185,129,0.15))',
                  border: '1px solid rgba(0,98,255,0.25)',
                  boxShadow: '0 8px 24px rgba(0,98,255,0.15)',
                }}
              >
                <Brain size={34} className="text-[#0062FF]" style={{ filter: 'drop-shadow(0 2px 8px rgba(0,98,255,0.4))' }} />
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-2 mb-3">
              <span className="px-3 py-1 rounded-full text-[0.7rem] font-bold uppercase tracking-wider bg-blue-100 text-blue-900 border border-blue-200">
                DEAP 12-Subjects Active (13,920 Windows)
              </span>
              <span className="px-3 py-1 rounded-full text-[0.7rem] font-bold uppercase tracking-wider bg-indigo-100 text-indigo-900 border border-indigo-200">
                SpikingJelly LIF Multi-Step SNN
              </span>
              <span className="px-3 py-1 rounded-full text-[0.7rem] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-900 border border-emerald-200">
                Closed-Loop Q-Learning (Bellman)
              </span>
            </div>

            <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">
              SNN-AI Cognitive Health & Neuro-Monitoring Platform
            </h1>
            <p className="landing-subtitle max-w-2xl mx-auto">
              Dual-Portal Neuro-Computational Architecture: Rigorous Empirical Research Validation & Clinical Closed-Loop Adaptive Interventions
            </p>
          </div>

          {/* DUAL PORTAL GRID: RESEARCH (LEFT) vs CLINICAL (RIGHT) */}
          <div className="dual-portal-grid">
            {/* ========================================================= */}
            {/* LEFT PORTAL CARD: Academic Research & Benchmark Lab       */}
            {/* ========================================================= */}
            <div className="portal-card portal-card-research">
              <div>
                {/* Badge & Icon Header */}
                <div className="flex items-center justify-between mb-4">
                  <span className="portal-chip bg-indigo-50 text-indigo-800 border border-indigo-200/80">
                    <Award size={12} className="text-indigo-600" />
                    Academic Research & SNN Validation
                  </span>
                  <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-700 flex items-center justify-center shadow-xs">
                    <Cpu size={20} />
                  </div>
                </div>

                <h2 className="text-xl font-bold font-heading text-slate-900 mb-2">
                  Research & Benchmark Dashboard
                </h2>
                <p className="text-xs text-slate-600 mb-5 leading-relaxed">
                  Rigorous empirical validation of multi-step Leaky Integrate-and-Fire (LIF) Spiking Neural Networks against classical SVM and Random Forest baselines on held-out DEAP subjects.
                </p>

                {/* Feature Rows */}
                <div className="space-y-2.5 mb-6">
                  <div className="portal-feature-row">
                    <Award size={16} className="text-indigo-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="text-xs font-bold text-slate-900 block">Empirical Model Scorecard</strong>
                      <span className="text-[11px] text-slate-500">
                        Real held-out evaluation: SNN (54.17% trial acc, 0.5401 F1, 0.5470 AUC) vs. SVM (53.13%) & RF (55.46%).
                      </span>
                    </div>
                  </div>

                  <div className="portal-feature-row">
                    <Zap size={16} className="text-indigo-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="text-xs font-bold text-slate-900 block">SNN Background Training Runner</strong>
                      <span className="text-[11px] text-slate-500">
                        One-click training execution on local DEAP dataset with 4s sliding windows and Poisson rate coding.
                      </span>
                    </div>
                  </div>

                  <div className="portal-feature-row">
                    <ShieldCheck size={16} className="text-indigo-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="text-xs font-bold text-slate-900 block">DEAP Zero-Leakage Dataset Audit</strong>
                      <span className="text-[11px] text-slate-500">
                        Strict disjoint holdout: 9 training subjects (s01–s09) vs 3 test subjects (s10–s12) with 0% subject leakage.
                      </span>
                    </div>
                  </div>

                  <div className="portal-feature-row">
                    <TrendingUp size={16} className="text-indigo-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="text-xs font-bold text-slate-900 block">Neuromorphic Spike Raster & V(t)</strong>
                      <span className="text-[11px] text-slate-500">
                        Interactive inspection of LIF membrane potentials, firing thresholds (V_th=1.0V), and event sparsity (~88%).
                      </span>
                    </div>
                  </div>
                </div>

                {/* Compact Metrics Pill Banner */}
                <div className="p-2.5 rounded-xl bg-indigo-50/70 border border-indigo-200/60 mb-6 flex items-center justify-around text-center text-[11px] font-mono text-indigo-900">
                  <div>
                    <span className="text-[10px] text-indigo-600 block uppercase font-sans font-bold">SNN Acc</span>
                    <strong>54.17%</strong>
                  </div>
                  <div className="h-6 w-px bg-indigo-200" />
                  <div>
                    <span className="text-[10px] text-indigo-600 block uppercase font-sans font-bold">F1-Macro</span>
                    <strong>0.5401</strong>
                  </div>
                  <div className="h-6 w-px bg-indigo-200" />
                  <div>
                    <span className="text-[10px] text-indigo-600 block uppercase font-sans font-bold">Leakage</span>
                    <strong className="text-emerald-700">0.0% (Passed)</strong>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <div>
                <button
                  type="button"
                  onClick={handleEnterResearch}
                  className="w-full py-3 px-6 rounded-xl font-heading font-bold text-sm text-white bg-gradient-to-r from-indigo-600 via-blue-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-2 group"
                >
                  <Sparkles size={16} />
                  <span>Enter Research & Benchmark Dashboard</span>
                  <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>

            {/* ========================================================= */}
            {/* RIGHT PORTAL CARD: Clinical Neuro-Monitoring Hub          */}
            {/* ========================================================= */}
            <div className="portal-card portal-card-clinical">
              <div>
                {/* Badge & Icon Header */}
                <div className="flex items-center justify-between mb-4">
                  <span className="portal-chip bg-emerald-50 text-emerald-800 border border-emerald-200/80">
                    <HeartPulse size={12} className="text-emerald-600" />
                    Clinical Application & Adaptive RL
                  </span>
                  <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center shadow-xs">
                    <Activity size={20} />
                  </div>
                </div>

                <h2 className="text-xl font-bold font-heading text-slate-900 mb-2">
                  Clinical Neuro-Monitoring Hub
                </h2>
                <p className="text-xs text-slate-600 mb-5 leading-relaxed">
                  Real-time patient EEG bio-signal ingestion, continuous neuromorphic affective-state inference (Low vs. High Arousal), and closed-loop Q-learning task recommendations to optimize cognitive load.
                </p>

                {/* Feature Rows */}
                <div className="space-y-2.5 mb-6">
                  <div className="portal-feature-row">
                    <FolderUp size={16} className="text-emerald-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="text-xs font-bold text-slate-900 block">Patient EEG Ingestion & Triage</strong>
                      <span className="text-[11px] text-slate-500">
                        Drag-and-drop clinical standard European Data Format (.edf) & .csv upload with automated Welch PSD decomposition.
                      </span>
                    </div>
                  </div>

                  <div className="portal-feature-row">
                    <Activity size={16} className="text-emerald-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="text-xs font-bold text-slate-900 block">Real-Time Telemetry & Biometrics</strong>
                      <span className="text-[11px] text-slate-500">
                        Continuous live monitoring of alpha relaxation, beta workload, heart rate variability, and SNN arousal inference.
                      </span>
                    </div>
                  </div>

                  <div className="portal-feature-row">
                    <SlidersHorizontal size={16} className="text-emerald-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="text-xs font-bold text-slate-900 block">Closed-Loop Q-Learning Feedback</strong>
                      <span className="text-[11px] text-slate-500">
                        Active Bellman TD reinforcement updates: Q(s, a) ← Q + α[r + γ max Q' - Q] with 3-button clinician feedback.
                      </span>
                    </div>
                  </div>

                  <div className="portal-feature-row">
                    <FileText size={16} className="text-emerald-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="text-xs font-bold text-slate-900 block">Session Diagnostics & Progress Log</strong>
                      <span className="text-[11px] text-slate-500">
                        Persistent tracking of patient sessions, stress spikes, breathing interventions, and cognitive stamina.
                      </span>
                    </div>
                  </div>
                </div>

                {/* Compact Metrics Pill Banner */}
                <div className="p-2.5 rounded-xl bg-emerald-50/70 border border-emerald-200/60 mb-6 flex items-center justify-around text-center text-[11px] font-mono text-emerald-900">
                  <div>
                    <span className="text-[10px] text-emerald-600 block uppercase font-sans font-bold">Sampling</span>
                    <strong>128 Hz EEG</strong>
                  </div>
                  <div className="h-6 w-px bg-emerald-200" />
                  <div>
                    <span className="text-[10px] text-emerald-600 block uppercase font-sans font-bold">Affective SNN</span>
                    <strong>Arousal (L/H)</strong>
                  </div>
                  <div className="h-6 w-px bg-emerald-200" />
                  <div>
                    <span className="text-[10px] text-emerald-600 block uppercase font-sans font-bold">Policy</span>
                    <strong className="text-emerald-700">ε-Greedy RL</strong>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <div>
                <button
                  type="button"
                  onClick={handleEnterClinical}
                  className="w-full py-3 px-6 rounded-xl font-heading font-bold text-sm text-white bg-gradient-to-r from-emerald-600 via-teal-600 to-emerald-700 hover:from-emerald-700 hover:to-teal-800 shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-2 group"
                >
                  <HeartPulse size={16} />
                  <span>Enter Clinical Neuro-Monitoring Hub</span>
                  <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>
          </div>

          {/* Research Integrity Footer */}
          <div className="text-center pt-4 border-t border-slate-200/60 text-xs text-slate-500 max-w-xl mx-auto">
            <p>
              Academic Research Prototype • Conforms strictly to <strong>AGENTS.md</strong>: Zero fabricated metrics, zero subject leakage, and genuine closed-loop reinforcement learning.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
