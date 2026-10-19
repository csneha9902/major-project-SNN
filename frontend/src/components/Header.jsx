import React from 'react';
import GlowButton from './ui/GlowButton';
import { Play, Square, LogOut, RefreshCw, FileText, Activity, Sparkles } from 'lucide-react';

export default function Header({ 
  isRunning, 
  onStart, 
  onStop, 
  onLogout, 
  dataSource = 'none', 
  sourceMetadata = null, 
  onSwitchSource 
}) {
  return (
    <header className="header-container">
      <div className="title-section">
        <div className="flex flex-wrap items-center gap-3">
          <h1>SNN-AI Cognitive Health & Learning Optimizer</h1>
          {dataSource === 'file' && (
            <div className="flex items-center gap-2">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-800 text-xs font-mono font-medium shadow-xs">
                <FileText size={13} className="text-blue-600" />
                <span>Source: {sourceMetadata?.filename || 'Uploaded Recording'}</span>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-[0.68rem] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-800 border border-emerald-300">
                RESEARCH MODE (DEAP SNN)
              </span>
            </div>
          )}
          {dataSource === 'stream' && (
            <div className="flex items-center gap-2">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-mono font-medium shadow-xs">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span>Live Telemetry Active</span>
              </div>
              <span className="px-2 py-0.5 rounded-full text-[0.68rem] font-bold uppercase tracking-wider bg-amber-100 text-amber-900 border border-amber-300">
                DEMO MODE (SYNTHETIC STREAM)
              </span>
            </div>
          )}
          {dataSource === 'none' && (
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-slate-100 border border-slate-200 text-slate-600 text-xs font-medium">
              <span>Ready for Ingestion</span>
            </div>
          )}
        </div>
        <p className="tagline">
          {isRunning && <span className="live-dot mr-2" />}
          {dataSource === 'file' 
            ? `Analyzing parsed signal features (${sourceMetadata?.duration || '45'}s duration) with Event-Driven SNN.`
            : dataSource === 'stream' 
            ? 'Continuous bio-sensor ingestion stream.'
            : 'Personalized neural assessment and adaptive learning optimizer.'}
        </p>
      </div>

      <div className="header-controls flex items-center gap-2">
        {dataSource !== 'none' && onSwitchSource && (
          <GlowButton variant="secondary" onClick={onSwitchSource} className="text-xs">
            <RefreshCw size={14} />
            <span>Switch Data Source</span>
          </GlowButton>
        )}

        {dataSource === 'stream' && (
          <>
            <GlowButton variant="cyan" onClick={onStart} disabled={isRunning}>
              <Play size={16} />
              Start Stream
            </GlowButton>
            <GlowButton variant="danger" onClick={onStop} disabled={!isRunning}>
              <Square size={14} />
              Stop
            </GlowButton>
          </>
        )}

        {onLogout && (
          <GlowButton variant="ghost" onClick={onLogout}>
            <LogOut size={16} />
            Logout
          </GlowButton>
        )}
      </div>
    </header>
  );
}
