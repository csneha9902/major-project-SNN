import { useState, useEffect, useMemo } from 'react';
import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ComposedChart, LineChart, Line, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend, CartesianGrid,
} from 'recharts';
import FileUpload from '../components/FileUpload';
import ChartTooltip from '../components/ui/ChartTooltip';
import GlowButton from '../components/ui/GlowButton';
import { ArrowLeft, RotateCcw } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const COLORS = ['#16A34A', '#059669', '#15803D', '#22C55E', '#14532D'];

const formatTimestamp = (ts) => {
  if (!ts) return '00:00';
  if (ts > 1000000000) {
    return new Date(ts * 1000).toLocaleTimeString();
  }
  const minutes = Math.floor(ts / 60);
  const seconds = Math.floor(ts % 60);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

export default function AnalysisView() {
  const { uploadId } = useParams();
  const navigate = useNavigate();
  const getAuthHeaders = () => ({});
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [chartKey, setChartKey] = useState(0);

  useEffect(() => {
    if (uploadId) {
      loadAnalysis(uploadId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadId]);

  useEffect(() => {
    if (analysisData && analysisData.time_series && analysisData.time_series.length > 0) {
      setChartKey(prev => prev + 1);
    }
  }, [analysisData]);

  const loadAnalysis = async (id) => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_BASE}/api/analysis/${id}`, {
        headers: getAuthHeaders(),
      });
      if (!res.ok) throw new Error('Failed to load analysis');
      const data = await res.json();
      setAnalysisData(data);
    } catch (err) {
      setError(err.message || 'Failed to load analysis');
    } finally {
      setLoading(false);
    }
  };


  const timeSeries = analysisData?.time_series || [];
  const extended = analysisData?.extended_analysis || {};
  const patterns = extended.patterns || {};
  const freqDomain = extended.frequency_domain || {};
  const stats = extended.statistics || {};

  const displayData = useMemo(() => {
    if (!analysisData || !timeSeries.length) {
      return [];
    }
    return timeSeries.map(t => ({
      timestamp: formatTimestamp(t.timestamp),
      alpha: t.alpha,
      beta: t.beta,
      heartRate: t.heart_rate || 0,
      state: t.cognitive_state,
    }));
  }, [timeSeries, analysisData]);

  if (loading) {
    return (
      <>
        <div className="app-background"><div className="orb-3" /><div className="grid-overlay" /></div>
        <div className="loading-container">
          <div className="loading-spinner flex items-center gap-3">
            <span className="live-dot" />
            Loading analysis...
          </div>
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <div className="app-background"><div className="orb-3" /><div className="grid-overlay" /></div>
        <div className="app-container">
          <div className="error-message">{error}</div>
          <FileUpload onUploadSuccess={(id) => navigate(`/analysis/${id}`)} />
        </div>
      </>
    );
  }

  if (!analysisData) {
    return (
      <>
        <div className="app-background"><div className="orb-3" /><div className="grid-overlay" /></div>
        <div className="app-container">
          <div className="analysis-header">
            <GlowButton variant="ghost" onClick={() => navigate('/dashboard')}>
              <ArrowLeft size={18} />
              Back to Dashboard
            </GlowButton>
            <h1 className="font-heading font-bold">File Analysis</h1>
            <div />
          </div>
          <FileUpload onUploadSuccess={(id) => navigate(`/analysis/${id}`)} />
        </div>
      </>
    );
  }

  const handleReplayAnimation = () => {
    setChartKey(prev => prev + 1);
  };

  const stateDistribution = Object.entries(patterns.state_distribution || {}).map(([name, value]) => ({
    name,
    value,
  }));

  const freqData = (freqDomain.frequencies || []).slice(0, 50).map((freq, i) => ({
    frequency: freq,
    alphaPower: (freqDomain.alpha_power || [])[i] || 0,
    betaPower: (freqDomain.beta_power || [])[i] || 0,
  }));

  // Custom label for pie chart
  const renderPieLabel = ({ name, percent, cx, cy, midAngle, outerRadius }) => {
    const RADIAN = Math.PI / 180;
    const radius = outerRadius + 20;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    return (
      <text
        x={x}
        y={y}
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
        fill="var(--text-secondary)"
        fontSize={11}
        fontFamily="'Exo 2', sans-serif"
      >
        {`${name}: ${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <>
      <div className="app-background"><div className="orb-3" /><div className="grid-overlay" /></div>
      <div className="app-container">
        <div className="analysis-header">
          <GlowButton variant="ghost" onClick={() => navigate('/dashboard')}>
            <ArrowLeft size={18} />
            Back to Dashboard
          </GlowButton>
          <h1 className="font-heading font-bold">
            File Analysis: {analysisData.filename || 'EDF File'}
          </h1>
        </div>

        <div className="analysis-content stagger-children">
          <div className="analysis-left">
            {/* Time Series */}
            <div className="analysis-card animate-slide-up">
              <div className="chart-header-controls">
                <h3>Time Series Analysis</h3>
                {timeSeries.length > 0 && (
                  <div className="animation-controls">
                    <button className="btn-animation" onClick={handleReplayAnimation} title="Replay Animation">
                      <RotateCcw size={14} />
                      Replay
                    </button>
                  </div>
                )}
              </div>
              <ResponsiveContainer width="100%" height={320}>
                <ComposedChart
                  key={`chart-${chartKey}`}
                  data={displayData}
                  margin={{ top: 10, right: 30, left: 0, bottom: 5 }}
                >
                  <defs>
                    <linearGradient id="anaAlpha" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#059669" stopOpacity={0.25} />
                      <stop offset="100%" stopColor="#059669" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="anaBeta" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#16A34A" stopOpacity={0.25} />
                      <stop offset="100%" stopColor="#16A34A" stopOpacity={0} />
                    </linearGradient>
                    <filter id="anaGlow">
                      <feGaussianBlur stdDeviation="3" result="blur" />
                      <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                  </defs>
                  <CartesianGrid strokeDasharray="3 6" stroke="rgba(34,197,94,0.12)" vertical={false} />
                  <XAxis
                    dataKey="timestamp"
                    stroke="transparent"
                    tick={{ fill: '#3F6212', fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    yAxisId="left"
                    stroke="transparent"
                    tick={{ fill: '#3F6212', fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
                    domain={[0, 2]}
                    axisLine={false}
                    tickLine={false}
                    label={{ value: 'α/β', angle: -90, position: 'insideLeft', fill: '#166534', fontSize: 11 }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    stroke="transparent"
                    tick={{ fill: '#D97706', fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
                    domain={['dataMin', 'dataMax']}
                    axisLine={false}
                    tickLine={false}
                    label={{ value: 'BPM', angle: 90, position: 'insideRight', fill: '#D97706', fontSize: 11 }}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend
                    wrapperStyle={{ paddingTop: '8px', fontSize: '0.75rem', fontFamily: "'Exo 2', sans-serif" }}
                    iconType="circle"
                    iconSize={8}
                  />
                  <Area yAxisId="left" type="monotone" dataKey="alpha" fill="url(#anaAlpha)" stroke="none" />
                  <Area yAxisId="left" type="monotone" dataKey="beta" fill="url(#anaBeta)" stroke="none" />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="alpha"
                    stroke="#059669"
                    name="Relaxation (Alpha)"
                    dot={false}
                    isAnimationActive={true}
                    animationDuration={1500}
                    animationEasing="ease-in-out"
                    strokeWidth={2.5}
                    filter="url(#anaGlow)"
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="beta"
                    stroke="#16A34A"
                    name="Focus (Beta)"
                    dot={false}
                    isAnimationActive={true}
                    animationDuration={1500}
                    animationEasing="ease-in-out"
                    strokeWidth={2.5}
                    filter="url(#anaGlow)"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="heartRate"
                    stroke="#D97706"
                    name="Heart Rate (BPM)"
                    dot={false}
                    isAnimationActive={true}
                    animationDuration={1500}
                    animationEasing="ease-in-out"
                    strokeWidth={2}
                    strokeDasharray="6 3"
                    filter="url(#anaGlow)"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Frequency Domain */}
            {freqData.length > 0 && (
              <div className="analysis-card animate-slide-up">
                <h3>Frequency Domain (FFT)</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={freqData}>
                    <defs>
                      <linearGradient id="fftAlpha" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#059669" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#059669" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="fftBeta" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#16A34A" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#16A34A" stopOpacity={0} />
                      </linearGradient>
                      <filter id="fftGlow">
                        <feGaussianBlur stdDeviation="2.5" result="blur" />
                        <feMerge>
                          <feMergeNode in="blur" />
                          <feMergeNode in="SourceGraphic" />
                        </feMerge>
                      </filter>
                    </defs>
                    <CartesianGrid strokeDasharray="3 6" stroke="rgba(34,197,94,0.12)" vertical={false} />
                    <XAxis
                      dataKey="frequency"
                      stroke="transparent"
                      tick={{ fill: '#3F6212', fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
                      axisLine={false}
                      tickLine={false}
                      label={{ value: 'Frequency (Hz)', position: 'insideBottom', fill: '#166534', fontSize: 11, offset: -5 }}
                    />
                    <YAxis
                      stroke="transparent"
                      tick={{ fill: '#3F6212', fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
                      axisLine={false}
                      tickLine={false}
                      label={{ value: 'Power', angle: -90, fill: '#166534', fontSize: 11 }}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend
                      wrapperStyle={{ paddingTop: '8px', fontSize: '0.75rem', fontFamily: "'Exo 2', sans-serif" }}
                      iconType="circle"
                      iconSize={8}
                    />
                    <Area type="monotone" dataKey="alphaPower" fill="url(#fftAlpha)" stroke="none" />
                    <Area type="monotone" dataKey="betaPower" fill="url(#fftBeta)" stroke="none" />
                    <Line
                      type="monotone"
                      dataKey="alphaPower"
                      stroke="#059669"
                      name="Alpha Power"
                      dot={false}
                      isAnimationActive={true}
                      animationDuration={1500}
                      animationEasing="ease-in-out"
                      strokeWidth={2.5}
                      filter="url(#fftGlow)"
                    />
                    <Line
                      type="monotone"
                      dataKey="betaPower"
                      stroke="#16A34A"
                      name="Beta Power"
                      dot={false}
                      isAnimationActive={true}
                      animationDuration={1500}
                      animationEasing="ease-in-out"
                      strokeWidth={2.5}
                      filter="url(#fftGlow)"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <div className="analysis-right">
            {/* Pie Chart - State Distribution */}
            {stateDistribution.length > 0 && (
              <div className="analysis-card animate-slide-up">
                <h3>State Distribution</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <defs>
                      {COLORS.map((color, i) => (
                        <filter key={i} id={`pieGlow${i}`}>
                          <feGaussianBlur stdDeviation="2" result="blur" />
                          <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                          </feMerge>
                        </filter>
                      ))}
                    </defs>
                    <Pie
                      data={stateDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={renderPieLabel}
                      outerRadius={80}
                      innerRadius={35}
                      fill="#16A34A"
                      dataKey="value"
                      stroke="rgba(248,252,246,0.95)"
                      strokeWidth={2}
                      animationDuration={1500}
                      animationEasing="ease-out"
                    >
                      {stateDistribution.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={COLORS[index % COLORS.length]}
                          style={{ filter: `drop-shadow(0 0 6px ${COLORS[index % COLORS.length]}66)` }}
                        />
                      ))}
                    </Pie>
                    <Tooltip content={<ChartTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Extended Insights */}
            <div className="analysis-card animate-slide-up">
              <h3>Extended Insights</h3>
              <div className="insights-content">
                {extended.insights_text && extended.insights_text.map((insight, i) => (
                  <p key={i} className="insight-item">{insight}</p>
                ))}
              </div>
            </div>

            {/* Pattern Detection */}
            <div className="analysis-card animate-slide-up">
              <h3>Pattern Detection</h3>
              <div className="pattern-stats">
                <div className="stat-item">
                  <span className="stat-label">Stress Events</span>
                  <span className="stat-value">{patterns.stress_event_count || 0}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Focus Periods</span>
                  <span className="stat-value">{patterns.focus_period_count || 0}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">State Transitions</span>
                  <span className="stat-value">{patterns.transition_count || 0}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Dominant State</span>
                  <span className="stat-value">{patterns.dominant_state || 'N/A'}</span>
                </div>
              </div>
            </div>

            {/* Statistical Summary */}
            {stats.alpha && (
              <div className="analysis-card animate-slide-up">
                <h3>Statistical Summary</h3>
                <div className="stats-grid">
                  <div className="stat-box">
                    <div className="stat-title" style={{ color: '#059669' }}>Alpha</div>
                    <div className="stat-detail">Mean: {stats.alpha.mean?.toFixed(3)}</div>
                    <div className="stat-detail">Std: {stats.alpha.std?.toFixed(3)}</div>
                    <div className="stat-detail">Range: {stats.alpha.min?.toFixed(3)} – {stats.alpha.max?.toFixed(3)}</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-title" style={{ color: '#16A34A' }}>Beta</div>
                    <div className="stat-detail">Mean: {stats.beta.mean?.toFixed(3)}</div>
                    <div className="stat-detail">Std: {stats.beta.std?.toFixed(3)}</div>
                    <div className="stat-detail">Range: {stats.beta.min?.toFixed(3)} – {stats.beta.max?.toFixed(3)}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
