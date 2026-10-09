import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, Activity, FileText, UserCheck, Mail, Lock, LogIn } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function LandingPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      navigate('/dashboard');
    }
  }, [navigate]);

  const handleGoogleLogin = async () => {
    try {
      localStorage.setItem('user_role', 'researcher');
      window.location.href = `/auth/login`;
    } catch (error) {
      console.error('Login error:', error);
      alert('Login failed. Please try again.');
    }
  };

  const handleDemoLogin = async () => {
    setError('');
    setLoading(true);
    try {
      const demoUser = 'dr.smith';
      const demoPass = 'doctor123';
      const res = await fetch('/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: demoUser, password: demoPass }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || 'Demo authentication failed');
        return;
      }

      const data = await res.json();
      localStorage.setItem('auth_token', data.access_token);
      const backendRole = data.user?.role || 'Researcher';
      localStorage.setItem('user_role', 'researcher');
      localStorage.setItem('user_backend_role', backendRole);
      navigate('/dashboard');
    } catch (err) {
      setError('Network error — make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleCredentialsSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await fetch('/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: email, password }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || 'Invalid username or password');
        return;
      }

      const data = await res.json();
      localStorage.setItem('auth_token', data.access_token);
      const backendRole = data.user?.role || 'researcher';
      localStorage.setItem('user_role', 'researcher');
      localStorage.setItem('user_backend_role', backendRole);
      navigate('/dashboard');
    } catch (err) {
      setError('Network error — make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Animated background */}
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

          <div className="landing-card">
            {/* Header */}
            <div className="mb-6">
              <h2 className="text-xl font-bold font-heading text-slate-900 mb-2">Research & Optimization Portal</h2>
              <p className="landing-description">
                Access the neuro-computational workspace to ingest EEG bio-signals, monitor Spiking Neural Network (LIF) affective states, and receive adaptive Q-learning task recommendations.
              </p>
            </div>

            {/* Fillable Credentials Form */}
            <form className="credentials-form" onSubmit={handleCredentialsSubmit}>
              <div className="form-group">
                <label className="form-label">
                  <Mail size={15} />
                  <span>Username or Email</span>
                </label>
                <div className="input-wrapper">
                  <input
                    type="text"
                    className="credentials-input"
                    placeholder="researcher@domain.edu or admin"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">
                  <Lock size={15} />
                  <span>Password</span>
                </label>
                <div className="input-wrapper">
                  <input
                    type="password"
                    className="credentials-input"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
              </div>

              <button type="submit" className="btn-credentials-submit" disabled={loading}>
                <LogIn size={18} />
                <span>{loading ? 'Authenticating...' : 'Sign In to Workspace'}</span>
              </button>

              {error && (
                <div style={{
                  marginTop: '0.75rem',
                  padding: '0.6rem 1rem',
                  borderRadius: '8px',
                  background: 'rgba(239,68,68,0.1)',
                  border: '1px solid rgba(239,68,68,0.3)',
                  color: '#dc2626',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  textAlign: 'center',
                }}>
                  {error}
                </div>
              )}

              <div style={{
                marginTop: '0.75rem',
                padding: '0.75rem 1rem',
                borderRadius: '8px',
                background: '#F0F7FF',
                border: '1px solid #BAE6FD',
                fontSize: '0.78rem',
                color: '#0369A1',
                lineHeight: '1.6',
              }}>
                <strong style={{color:'#0284C7', fontWeight: 700}}>Quick Access Credentials:</strong>
                <br />
                Researcher: <code>dr.smith</code> / <code>doctor123</code>
                <br />
                Admin: <code>admin</code> / <code>admin123</code>
              </div>
            </form>

            <div className="divider">
              <span>OR CONTINUE WITH</span>
            </div>

            {/* OAuth and Demo Login Options */}
            <button className="btn-google-login" onClick={handleGoogleLogin}>
              <svg className="google-icon" viewBox="0 0 24 24" width="20" height="20">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
              </svg>
              <span>Continue with Google</span>
            </button>

            <button className="btn-demo-login flex items-center justify-center gap-2" onClick={handleDemoLogin} disabled={loading}>
              <UserCheck size={18} />
              <span>Launch Research Workspace (One-Click Demo Access)</span>
            </button>

            <p className="landing-note">
              Secure authentication. Your research data is protected.
              <br />
              <small className="opacity-75">Demo access available for instant evaluation without OAuth setup.</small>
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
