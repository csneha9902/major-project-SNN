import React, { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import CurrentStateCard from "../components/CurrentStateCard";
import BiometricTrendsChart from "../components/BiometricTrendsChart";
import TaskRecommendationCard from "../components/TaskRecommendationCard";
import WellnessTipsPanel from "../components/WellnessTipsPanel";
import SessionSummaryPanel from "../components/SessionSummaryPanel";
import ProgressCalendar from "../components/ProgressCalendar";
import UserOnboardingHub from "../components/UserOnboardingHub";
import { useDataStream } from "../hooks/useDataStream";
import { useAuth } from "../context/AuthContext";
import GlowButton from "../components/ui/GlowButton";
import EmailReportModal from "../components/EmailReportModal";
import { Download, Sparkles, Activity, FileText, CheckCircle2, RotateCcw, Mail } from "lucide-react";
import "../App.css";

const API_BASE = import.meta.env.VITE_API_URL || "";

const formatTimestamp = (ts) => {
  if (!ts) return "00:00";
  if (ts > 1000000000) {
    return new Date(ts * 1000).toLocaleTimeString();
  }
  const minutes = Math.floor(ts / 60);
  const seconds = Math.floor(ts % 60);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

export default function Dashboard() {
  const { frame, running, startSimulation, stopSimulation } = useDataStream();
  const { user, logout, getAuthHeaders } = useAuth();
  const navigate = useNavigate();

  const userRole = localStorage.getItem('user_role') || 'student';

  // Data Source mode: 'none' | 'file' | 'stream'
  const [dataSource, setDataSource] = useState('none');
  const [backendRec, setBackendRec] = useState(null);
  const [uploadId, setUploadId] = useState(null);
  const [sourceMetadata, setSourceMetadata] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [emailModalOpen, setEmailModalOpen] = useState(false);

  // Active chart & state data
  const [chartData, setChartData] = useState([]);
  const [hoveredState, setHoveredState] = useState(null);
  const [hoveredRecommendation, setHoveredRecommendation] = useState(null);

  // Load uploaded file analysis
  const loadAnalysis = async (id) => {
    setLoadingAnalysis(true);
    try {
      const res = await fetch(`${API_BASE}/api/analysis/${id}`, {
        headers: getAuthHeaders(),
      });
      if (!res.ok) throw new Error('Failed to load file analysis');
      const data = await res.json();
      setAnalysisData(data);
      setUploadId(id);
      setSourceMetadata({
        filename: data.filename || 'EEG_Recording.edf',
        duration: data.metadata?.duration || (data.time_series ? data.time_series.length : 45),
        uploadId: id
      });

      // Format time series into chart data
      if (data.time_series && data.time_series.length > 0) {
        const points = data.time_series.map((t) => ({
          timestamp: formatTimestamp(t.timestamp),
          alpha: Number(t.alpha.toFixed(2)),
          beta: Number(t.beta.toFixed(2)),
          heartRate: Math.round(t.heart_rate || 75),
          cognitive_state: t.cognitive_state || 'Neutral',
          recommendation: t.recommendation || null,
        }));
        setChartData(points);
      }
      setDataSource('file');
    } catch (err) {
      console.error("Error loading analysis:", err);
      alert("Could not load analysis: " + err.message);
    } finally {
      setLoadingAnalysis(false);
    }
  };

  // Handle upload success from onboarding hub
  const handleUploadSuccess = (newUploadId) => {
    loadAnalysis(newUploadId);
  };

  // Handle start live stream from onboarding hub
  const handleStartLiveStream = () => {
    setDataSource('stream');
    setChartData([]);
    setSourceMetadata({ filename: 'Live Wireless Telemetry Feed', duration: 'Real-Time' });
    startSimulation();
  };

  // Switch back to onboarding hub
  const handleSwitchSource = () => {
    if (running) {
      stopSimulation();
    }
    setDataSource('none');
    setUploadId(null);
    setSourceMetadata(null);
    setAnalysisData(null);
    setChartData([]);
  };

  // Live Stream frame updates
  useEffect(() => {
    if (dataSource === 'stream' && frame && running) {
      setChartData(prev => {
        const newRow = {
          timestamp: new Date(frame.timestamp * 1000).toLocaleTimeString(),
          alpha: frame?.eeg?.alpha ?? 0.5,
          beta: frame?.eeg?.beta ?? 0.5,
          heartRate: frame?.hrv?.heart_rate_bpm ?? 75,
          cognitive_state: frame?.cognitive_state || "Neutral",
          recommendation: frame?.recommendation || null,
        };
        const next = [...prev, newRow];
        return next.slice(-60);
      });
    }
  }, [frame, running, dataSource]);

  // Export PDF handler
  const handleExportPDF = async () => {
    if (!uploadId) return;
    try {
      const res = await fetch(`${API_BASE}/api/analysis/${uploadId}/export-pdf`, {
        headers: getAuthHeaders(),
      });
      if (!res.ok) throw new Error('Failed to generate PDF');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `cognitive_analysis_${uploadId}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      alert('Failed to export PDF: ' + err.message);
    }
  };

  // Calculate current live/file metrics
  const lastPoint = chartData[chartData.length - 1];
  const currentMetrics = dataSource === 'stream' && frame ? {
    alpha: frame?.eeg?.alpha ?? 0.5,
    beta: frame?.eeg?.beta ?? 0.5,
    heartRate: frame?.hrv?.heart_rate_bpm ?? 75,
  } : lastPoint ? {
    alpha: lastPoint.alpha,
    beta: lastPoint.beta,
    heartRate: lastPoint.heartRate,
  } : {
    alpha: 0.55,
    beta: 0.50,
    heartRate: 72,
  };

  const dominantState = analysisData?.extended_analysis?.patterns?.dominant_state;
  const displayState = hoveredState || (dataSource === 'stream' ? frame?.cognitive_state : dominantState) || lastPoint?.cognitive_state || 'Neutral';

  // Dynamic fetch of Q-table recommendation if not in frame
  useEffect(() => {
    let active = true;
    const fetchRec = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/recommend?state=${encodeURIComponent(displayState)}`);
        if (res.ok && active) {
          const rec = await res.json();
          setBackendRec(rec);
        }
      } catch (e) {
        // Graceful silent fallback
      }
    };
    if (!frame?.recommendation && !lastPoint?.recommendation) {
      fetchRec();
    }
    return () => { active = false; };
  }, [displayState, frame?.recommendation, lastPoint?.recommendation]);

  const displayRecommendation = hoveredRecommendation || 
    (dataSource === 'stream' ? frame?.recommendation : null) || 
    lastPoint?.recommendation || 
    backendRec;

  return (
    <div className="app-container">
      {/* Dynamic Header with Provenance Badge & Switcher */}
      <Header 
        isRunning={running} 
        onStart={startSimulation} 
        onStop={stopSimulation} 
        onLogout={logout} 
        dataSource={dataSource}
        sourceMetadata={sourceMetadata}
        onSwitchSource={handleSwitchSource}
      />

      {/* STATE 1: No Data Source Selected -> Show Onboarding Hub */}
      {dataSource === 'none' && (
        <UserOnboardingHub 
          onUploadSuccess={handleUploadSuccess}
          onStartLiveStream={handleStartLiveStream}
          loading={loadingAnalysis}
        />
      )}

      {/* STATE 2 & 3: File or Live Stream Data Active -> Show Full Dashboard */}
      {dataSource !== 'none' && (
        <div className="animate-fade-in">
          {/* Explicit DEMO MODE Warning Banner per AGENTS.md Rule 4 */}
          {dataSource === 'stream' && (
            <div className="demo-mode-banner p-3.5 rounded-xl mb-5 bg-amber-500/10 border border-amber-500/30 flex flex-wrap items-center justify-between gap-3 text-xs text-amber-950">
              <div className="flex items-center gap-2.5">
                <span className="px-2.5 py-1 rounded-full font-bold bg-amber-600 text-white text-[0.68rem] tracking-wider uppercase shadow-xs">
                  DEMO MODE
                </span>
                <span className="font-medium text-slate-700">
                  Live stream uses synthetic EEG telemetry for interactive demonstration. Not evaluated on held-out research subjects.
                </span>
              </div>
              {frame?.snn_inference && (
                <div className="flex items-center gap-2 font-mono text-[0.7rem] bg-white/90 px-2.5 py-1 rounded-lg border border-amber-300/80 shadow-xs">
                  <span className="text-slate-600">SNN Arousal:</span>
                  <strong className="text-amber-800">{frame.snn_inference.arousal_label}</strong>
                  <span className="text-slate-500">({Math.round((frame.snn_inference.confidence || 0) * 100)}% conf)</span>
                </div>
              )}
            </div>
          )}

          {/* File Analysis Action Banner if file is active */}
          {dataSource === 'file' && (
            <div className="file-active-banner p-4 rounded-xl mb-6 bg-blue-50/80 border border-blue-200 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-600 text-white flex items-center justify-center font-bold">
                  <FileText size={20} />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-0.5">
                    <h4 className="font-heading font-bold text-sm text-[var(--text-primary)]">
                      Analyzing Recording: <span className="font-mono text-blue-700">{sourceMetadata?.filename}</span>
                    </h4>
                    <span className="px-2 py-0.5 rounded text-[0.65rem] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-800 border border-emerald-300">
                      RESEARCH MODE — 128-CH SNN
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-secondary)]">
                    Full SNN cognitive decomposition complete • {chartData.length} samples processed
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <GlowButton variant="cyan" onClick={() => setEmailModalOpen(true)} className="text-xs py-2 px-3">
                  <Mail size={14} />
                  <span>Email Report</span>
                </GlowButton>
                <GlowButton variant="success" onClick={handleExportPDF} className="text-xs py-2 px-3">
                  <Download size={14} />
                  <span>Download PDF</span>
                </GlowButton>
                <GlowButton variant="ghost" onClick={handleSwitchSource} className="text-xs py-2 px-3">
                  <RotateCcw size={14} />
                  <span>Upload Different File</span>
                </GlowButton>
              </div>
            </div>
          )}

          {/* Email Report Modal */}
          <EmailReportModal
            isOpen={emailModalOpen}
            onClose={() => setEmailModalOpen(false)}
            uploadId={uploadId}
            filename={sourceMetadata?.filename}
            defaultRecipientName={user?.name || "Patient"}
          />

          <div className="main-content">
            <div className="left-section">
              <CurrentStateCard state={displayState} />
              <BiometricTrendsChart 
                data={chartData} 
                currentMetrics={currentMetrics}
                onHover={(state, recommendation) => {
                  setHoveredState(state);
                  setHoveredRecommendation(recommendation);
                }}
                onHoverEnd={() => {
                  setHoveredState(null);
                  setHoveredRecommendation(null);
                }}
              />
              <WellnessTipsPanel currentState={displayState} />
            </div>

            <div className="right-section">
              <TaskRecommendationCard 
                recommendation={displayRecommendation} 
                currentState={displayState}
                nextState={displayState}
              />
              <SessionSummaryPanel isRunning={running} />
            </div>
          </div>

          {/* Dynamic Progress & Health Calendar Keeper */}
          <ProgressCalendar />
        </div>
      )}
    </div>
  );
}
