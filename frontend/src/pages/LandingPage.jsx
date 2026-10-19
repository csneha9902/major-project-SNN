import { useNavigate } from 'react-router-dom';
import { Brain, Activity, FileText, ArrowRight, Sparkles } from 'lucide-react';

export default function LandingPage() {
  const navigate = useNavigate();

  const handleEnterWorkspace = () => {
    localStorage.setItem('user_role', 'researcher');
    navigate('/dashboard');
  };

  return (
    <>
      <div className="app-background">
        <div className="orb-3" />
        <div className="grid-overlay" />
      </div>

      <div className="landing-container">
        <div className="landing-content">
          <div className="landing-header">
            <div className="flex justify-center mb-6">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, rgba(0,98,255,0.12), rgba(0,180,216,0.12))',
                  border: '1px solid rgba(0,98,255,0.25)',
                  boxShadow: '0 8px 24px rgba(0,98,255,0.15)',
                }}
              >
                <Brain size={34} className="text-[#0062FF]" style={{ filter: 'drop-shadow(0 2px 8px rgba(0,98,255,0.4))' }} />
              </div>
            </div>
            <h1>SNN-AI Cognitive Health & Learning Optimizer</h1>
            <p className="landing-subtitle">
              EEG-Based Affective State Estimation & Reinforcement Learning Task Recommendation
            </p>
          </div>

          <div className="landing-card text-center">
            <div className="mb-6">
              <h2 className="text-xl font-bold font-heading text-slate-900 mb-2">Research & Optimization Portal</h2>
              <p className="landing-description">
                Access the neuro-computational workspace to ingest EEG bio-signals, monitor Spiking Neural Network (LIF) affective states, and receive adaptive Q-learning task recommendations.
              </p>
            </div>

            <div className="my-8 flex justify-center">
              <button
                className="btn-credentials-submit flex items-center justify-center gap-2 text-base py-3 px-8 shadow-lg hover:shadow-xl transition-all"
                onClick={handleEnterWorkspace}
              >
                <Sparkles size={18} />
                <span>Enter Research Dashboard</span>
                <ArrowRight size={18} />
              </button>
            </div>

            <p className="landing-note">
              Research prototype mode active • Subject-wise held-out evaluation & canonical 128 band-power features.
            </p>
          </div>

          <div className="landing-features stagger-children">
            <div className="feature-item animate-slide-up">
              <div className="flex justify-center mb-3">
                <Activity size={28} className="text-[#0062FF]" style={{ filter: 'drop-shadow(0 2px 6px rgba(0,98,255,0.3))' }} />
              </div>
              <h3>Real-Time Telemetry</h3>
              <p>Continuous EEG bio-signal ingestion with live LIF spiking neural inference</p>
            </div>
            <div className="feature-item animate-slide-up">
              <div className="flex justify-center mb-3">
                <Brain size={28} className="text-[#00B4D8]" style={{ filter: 'drop-shadow(0 2px 6px rgba(0,180,216,0.3))' }} />
              </div>
              <h3>128-ch Feature Pipeline</h3>
              <p>Welch PSD frequency band decomposition matching DEAP research training</p>
            </div>
            <div className="feature-item animate-slide-up">
              <div className="flex justify-center mb-3">
                <FileText size={28} className="text-[#4F46E5]" style={{ filter: 'drop-shadow(0 2px 6px rgba(79,70,229,0.3))' }} />
              </div>
              <h3>Adaptive Q-Learning</h3>
              <p>Closed-loop Bellman reinforcement optimization for personalized cognitive tasks</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
