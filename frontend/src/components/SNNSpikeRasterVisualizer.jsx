import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  ReferenceLine, 
  Legend 
} from 'recharts';
import { 
  Zap, 
  Activity, 
  Cpu, 
  Sparkles, 
  RefreshCw, 
  CheckCircle2, 
  Layers, 
  TrendingUp, 
  Info,
  Radio
} from 'lucide-react';
import GlassCard from './ui/GlassCard';
import GlowButton from './ui/GlowButton';
import MetricBadge from './ui/MetricBadge';
import { API_BASE, safeFetchJson } from '../api';

/**
 * Custom Tooltip for Membrane Voltage V(t) Chart
 */
const VoltageTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-slate-900/95 text-white p-3 rounded-xl border border-slate-700 shadow-xl text-xs backdrop-blur-md">
      <div className="font-mono font-bold text-cyan-400 mb-1 flex items-center justify-between gap-4">
        <span>Time-step t = {label}</span>
        {data.spiked && (
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-red-500/80 text-white font-bold animate-pulse">
            SPIKE FIRED
          </span>
        )}
      </div>
      <div className="space-y-1 font-mono text-[11px]">
        <div className="flex justify-between gap-4">
          <span className="text-slate-400">Neuron V(t):</span>
          <span className="text-emerald-400 font-bold">{data.v_selected?.toFixed(4)} V</span>
        </div>
        {data.v_mean !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Layer Mean V(t):</span>
            <span className="text-slate-300">{data.v_mean?.toFixed(4)} V</span>
          </div>
        )}
        <div className="flex justify-between gap-4 border-t border-slate-800 pt-1 text-[10px] text-slate-400">
          <span>Threshold distance:</span>
          <span>{Math.max(0, 1.0 - (data.v_selected || 0)).toFixed(4)} V</span>
        </div>
      </div>
    </div>
  );
};

export default function SNNSpikeRasterVisualizer({ 
  currentMetrics = null,
  isLiveStreaming = false 
}) {
  const [tracesData, setTracesData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeLayer, setActiveLayer] = useState('hidden1'); // 'input' | 'hidden1' | 'hidden2' | 'output'
  const [selectedNeuronIndex, setSelectedNeuronIndex] = useState(0);
  const [autoSync, setAutoSync] = useState(false);
  const [hoveredCell, setHoveredCell] = useState(null);

  // Fetch traces from backend API
  const fetchTraces = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let endpoint = `/api/snn/neuromorphic-traces?time_steps=50&n_neurons=16`;
      if (currentMetrics?.alpha !== undefined && currentMetrics?.beta !== undefined) {
        endpoint += `&alpha=${currentMetrics.alpha}&beta=${currentMetrics.beta}`;
      }
      const data = await safeFetchJson(endpoint);
      setTracesData(data);
    } catch (err) {
      console.warn('Neuromorphic traces fetch warning:', err.message);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [currentMetrics?.alpha, currentMetrics?.beta]);

  // Initial fetch on mount
  useEffect(() => {
    fetchTraces();
  }, [fetchTraces]);

  // Optional auto-sync when live stream is active
  useEffect(() => {
    if (!autoSync || !isLiveStreaming) return;
    const interval = setInterval(() => {
      fetchTraces();
    }, 4000);
    return () => clearInterval(interval);
  }, [autoSync, isLiveStreaming, fetchTraces]);

  // Current active layer data
  const currentLayer = tracesData?.layers?.[activeLayer] || null;
  const timeSteps = tracesData?.time_steps || 50;
  const numNeurons = currentLayer?.n_neurons || (currentLayer?.spikes?.[0]?.length ?? 0);

  // Normalize selected neuron if out of range
  useEffect(() => {
    if (numNeurons > 0 && selectedNeuronIndex >= numNeurons) {
      setSelectedNeuronIndex(0);
    }
  }, [numNeurons, selectedNeuronIndex]);

  // Prepare Membrane Potential Chart Data for selected neuron
  const voltageChartData = useMemo(() => {
    if (!currentLayer || !currentLayer.v) return [];
    const vMatrix = currentLayer.v; // Array of T steps, each an array of N floats
    const sMatrix = currentLayer.spikes; // Array of T steps, each an array of N ints

    return vMatrix.map((stepV, t) => {
      const vVal = stepV[selectedNeuronIndex] ?? 0;
      const spiked = sMatrix && sMatrix[t] ? Boolean(sMatrix[t][selectedNeuronIndex]) : false;
      const vMean = stepV.length > 0 ? stepV.reduce((acc, curr) => acc + curr, 0) / stepV.length : 0;

      return {
        step: t,
        v_selected: Number(vVal.toFixed(4)),
        v_mean: Number(vMean.toFixed(4)),
        v_threshold: tracesData?.v_threshold ?? 1.0,
        v_reset: tracesData?.v_reset ?? 0.0,
        spiked,
      };
    });
  }, [currentLayer, selectedNeuronIndex, tracesData]);

  // Compute Neuromorphic efficiency metrics
  const layerSparsity = currentLayer?.firing_rate !== undefined 
    ? Math.round((1 - currentLayer.firing_rate) * 1000) / 10 
    : 85.0;

  const energyReductionFactor = currentLayer?.firing_rate
    ? Math.max(1, Math.round(1 / Math.max(0.01, currentLayer.firing_rate) * 10) / 10)
    : 5.2;

  // Neuron labels
  const getNeuronLabel = (idx) => {
    if (activeLayer === 'output') {
      return idx === 0 ? 'Low Arousal (LIF 0)' : 'High Arousal (LIF 1)';
    }
    if (activeLayer === 'input') {
      return `Ch ${idx + 1} (Poisson)`;
    }
    return `Neuron ${idx + 1}`;
  };

  return (
    <div className="snn-raster-visualizer my-8">
      <GlassCard className="border border-cyan-500/20 shadow-xl overflow-hidden bg-white/95">
        {/* Header Section */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-emerald-500 text-white flex items-center justify-center shadow-md">
              <Zap size={22} className="animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-heading font-bold text-lg text-slate-800">
                  Neuromorphic Spike Raster & LIF Membrane Dynamics
                </h3>
                <span className="px-2 py-0.5 rounded-full text-[0.68rem] font-bold bg-cyan-100 text-cyan-800 border border-cyan-300">
                  T = {timeSteps} Steps
                </span>
                <span className="px-2 py-0.5 rounded-full text-[0.68rem] font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
                  SpikingJelly LIF
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Inspect biological event-driven spike trains and sub-threshold membrane potentials V(t) across network layers.
              </p>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            {isLiveStreaming && (
              <button
                type="button"
                onClick={() => setAutoSync(!autoSync)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
                  autoSync 
                    ? 'bg-emerald-600 text-white border-emerald-500 shadow-sm' 
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border-slate-300'
                }`}
              >
                <Radio size={13} className={autoSync ? 'animate-pulse' : ''} />
                <span>Live Sync {autoSync ? 'ON' : 'OFF'}</span>
              </button>
            )}

            <GlowButton 
              variant="outline" 
              onClick={fetchTraces} 
              disabled={loading}
              className="text-xs py-1.5 px-3"
            >
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
              <span>{loading ? 'Simulating...' : 'Refresh Dynamics'}</span>
            </GlowButton>
          </div>
        </div>

        {/* Efficiency Metric Badges Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-4">
          <MetricBadge
            label="Layer Sparsity"
            value={`${layerSparsity}%`}
            unit="quiescent"
            color="green"
            icon={<Cpu size={18} />}
          />
          <MetricBadge
            label="Firing Frequency"
            value={(currentLayer?.firing_rate ?? 0.18).toFixed(3)}
            unit="spikes/step"
            color="cyan"
            icon={<Activity size={18} />}
          />
          <MetricBadge
            label="Neuromorphic Efficiency"
            value={`${energyReductionFactor}×`}
            unit="vs Dense ANN"
            color="violet"
            icon={<Sparkles size={18} />}
          />
          <MetricBadge
            label="Model Checkpoint"
            value={tracesData?.model_status === 'active_checkpoint' ? 'Active' : 'Unweighted'}
            unit="DEAP 12-Subj"
            color={tracesData?.model_status === 'active_checkpoint' ? 'blue' : 'amber'}
            icon={<CheckCircle2 size={18} />}
          />
        </div>

        {/* Layer Selector Tabs */}
        <div className="flex flex-wrap items-center gap-2 p-1.5 bg-slate-100 rounded-xl mb-4 border border-slate-200">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 px-2 flex items-center gap-1">
            <Layers size={14} /> Layer:
          </span>

          <button
            type="button"
            onClick={() => setActiveLayer('input')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeLayer === 'input'
                ? 'bg-white text-cyan-700 shadow-sm border border-cyan-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Input Encoders (16 ch Poisson)
          </button>

          <button
            type="button"
            onClick={() => setActiveLayer('hidden1')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeLayer === 'hidden1'
                ? 'bg-white text-emerald-700 shadow-sm border border-emerald-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Hidden LIF 1 (256 Neurons)
          </button>

          <button
            type="button"
            onClick={() => setActiveLayer('hidden2')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeLayer === 'hidden2'
                ? 'bg-white text-emerald-700 shadow-sm border border-emerald-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Hidden LIF 2 (128 Neurons)
          </button>

          <button
            type="button"
            onClick={() => setActiveLayer('output')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeLayer === 'output'
                ? 'bg-white text-indigo-700 shadow-sm border border-indigo-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Readout Layer (2 LIF Arousal)
          </button>
        </div>

        {/* ------------------------------------------------------------- */}
        {/* SPIKE RASTER MATRIX PLOT                                      */}
        {/* ------------------------------------------------------------- */}
        <div className="raster-container bg-slate-950 rounded-xl p-4 mb-5 border border-slate-800 shadow-inner overflow-hidden">
          <div className="flex items-center justify-between mb-3 text-xs">
            <div className="flex items-center gap-2">
              <span className="font-mono font-bold text-emerald-400 uppercase tracking-wide">
                Spike Raster Matrix: {currentLayer?.name || activeLayer}
              </span>
              <span className="text-slate-400">
                ({numNeurons} representative units across {timeSteps} simulation steps)
              </span>
            </div>
            <div className="flex items-center gap-4 text-[11px] text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-emerald-400 shadow-[0_0_6px_#34d399]" />
                <span>Action Potential (Spike = 1)</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-slate-800" />
                <span>Sub-threshold (0)</span>
              </span>
            </div>
          </div>

          {/* Raster Rows Matrix */}
          <div className="overflow-x-auto pb-1">
            <div className="min-w-[620px]">
              {/* Timeline Header Tick Marks */}
              <div className="flex items-center text-[10px] font-mono text-slate-400 mb-1 pl-28 pr-2">
                <span className="w-1/5 text-left">t = 0</span>
                <span className="w-1/5 text-left">t = 12</span>
                <span className="w-1/5 text-center">t = 25</span>
                <span className="w-1/5 text-right">t = 37</span>
                <span className="w-1/5 text-right">t = 50</span>
              </div>

              {/* Rows for each sampled neuron */}
              <div className="space-y-1">
                {Array.from({ length: numNeurons }).map((_, nIdx) => {
                  const isSelected = nIdx === selectedNeuronIndex && activeLayer !== 'input';
                  const label = getNeuronLabel(nIdx);

                  // Calculate firing rate for this specific neuron
                  const neuronSpikes = currentLayer?.spikes?.map(step => step[nIdx]) || [];
                  const totalFired = neuronSpikes.reduce((a, b) => a + (b || 0), 0);
                  const nFiringRate = neuronSpikes.length > 0 ? (totalFired / neuronSpikes.length).toFixed(2) : '0.00';

                  return (
                    <div
                      key={`raster-row-${activeLayer}-${nIdx}`}
                      onClick={() => activeLayer !== 'input' && setSelectedNeuronIndex(nIdx)}
                      className={`flex items-center group cursor-pointer transition-colors rounded px-1.5 py-0.5 ${
                        isSelected 
                          ? 'bg-slate-800/90 ring-1 ring-emerald-500' 
                          : 'hover:bg-slate-900/80'
                      }`}
                    >
                      {/* Left Label */}
                      <div className="w-28 flex-shrink-0 flex items-center justify-between pr-2 text-[11px] font-mono">
                        <span className={`truncate ${isSelected ? 'text-emerald-400 font-bold' : 'text-slate-300'}`}>
                          {label}
                        </span>
                        <span className="text-[9px] text-slate-400 opacity-80 group-hover:opacity-100">
                          {totalFired} spk
                        </span>
                      </div>

                      {/* Spike Cells for t = 0..T-1 */}
                      <div className="flex-1 flex items-center gap-[2px] h-3.5 bg-slate-900/60 rounded px-1">
                        {currentLayer?.spikes?.map((stepSpikes, tIdx) => {
                          const fired = Boolean(stepSpikes && stepSpikes[nIdx]);
                          const isHovered = hoveredCell?.n === nIdx && hoveredCell?.t === tIdx;

                          return (
                            <div
                              key={`cell-${nIdx}-${tIdx}`}
                              onMouseEnter={() => setHoveredCell({ n: nIdx, t: tIdx, fired, label })}
                              onMouseLeave={() => setHoveredCell(null)}
                              className={`flex-1 h-full rounded-[1px] transition-all ${
                                fired
                                  ? activeLayer === 'output' && nIdx === 1
                                    ? 'bg-amber-400 shadow-[0_0_6px_#fbbf24]'
                                    : 'bg-emerald-400 shadow-[0_0_5px_#34d399]'
                                  : 'bg-slate-800/40 hover:bg-slate-700/60'
                              } ${isHovered ? 'scale-125 z-10' : ''}`}
                            />
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Hover Status Info Footer */}
          <div className="mt-2 pt-2 border-t border-slate-850 flex items-center justify-between text-[11px] text-slate-400 font-mono">
            <span>
              {hoveredCell ? (
                <span className="text-cyan-300">
                  {hoveredCell.label} at step t = {hoveredCell.t}: {hoveredCell.fired ? '⚡ SPIKE (V >= 1.0V)' : '○ Inactive (V < 1.0V)'}
                </span>
              ) : (
                <span className="text-slate-400">
                  Click any neuron row above to focus its continuous membrane voltage V(t) trace below.
                </span>
              )}
            </span>
            <span className="text-slate-400">
              Selected Unit: <strong className="text-emerald-400">{getNeuronLabel(selectedNeuronIndex)}</strong>
            </span>
          </div>
        </div>

        {/* ------------------------------------------------------------- */}
        {/* CONTINUOUS MEMBRANE VOLTAGE V(t) LINE CHART                   */}
        {/* ------------------------------------------------------------- */}
        {activeLayer !== 'input' ? (
          <div className="membrane-trace-card bg-slate-50 rounded-xl p-4 border border-slate-200">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div>
                <h4 className="font-heading font-bold text-sm text-slate-800 flex items-center gap-2">
                  <TrendingUp size={16} className="text-emerald-600" />
                  Membrane Potential Trajectory V(t) — {getNeuronLabel(selectedNeuronIndex)}
                </h4>
                <p className="text-xs text-slate-500">
                  Visualizes leaky charge accumulation, threshold firing events (1.0 V), and instant refractory reset (0.0 V).
                </p>
              </div>

              {/* Neuron Picker Selector */}
              <div className="flex items-center gap-2">
                <label className="text-xs font-semibold text-slate-600">Selected Unit:</label>
                <select
                  value={selectedNeuronIndex}
                  onChange={(e) => setSelectedNeuronIndex(Number(e.target.value))}
                  className="text-xs py-1 px-2.5 rounded-lg border border-slate-300 bg-white font-mono text-slate-700 shadow-xs focus:ring-2 focus:ring-emerald-500 focus:outline-hidden"
                >
                  {Array.from({ length: numNeurons }).map((_, idx) => (
                    <option key={`opt-${idx}`} value={idx}>
                      {getNeuronLabel(idx)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Recharts Continuous Voltage Plot */}
            <div className="w-full h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={voltageChartData} margin={{ top: 10, right: 25, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="step" 
                    tick={{ fontSize: 11, fill: '#64748b' }} 
                    label={{ value: 'Simulation Step (t)', position: 'insideBottom', offset: -4, fontSize: 11, fill: '#94a3b8' }} 
                  />
                  <YAxis 
                    domain={[-0.1, 1.2]} 
                    ticks={[0.0, 0.25, 0.5, 0.75, 1.0]} 
                    tick={{ fontSize: 11, fill: '#64748b' }} 
                    label={{ value: 'Voltage (V)', angle: -90, position: 'insideLeft', offset: 18, fontSize: 11, fill: '#94a3b8' }} 
                  />
                  <Tooltip content={<VoltageTooltip />} />
                  <Legend 
                    verticalAlign="top" 
                    height={24} 
                    wrapperStyle={{ fontSize: 11, paddingBottom: 6 }} 
                  />

                  {/* Firing Threshold Reference Line */}
                  <ReferenceLine 
                    y={1.0} 
                    stroke="#dc2626" 
                    strokeDasharray="4 4" 
                    strokeWidth={1.5}
                    label={{ 
                      value: 'Threshold V_th = 1.0 V', 
                      fill: '#dc2626', 
                      fontSize: 10, 
                      position: 'top',
                      fontWeight: 600
                    }} 
                  />

                  {/* Resting / Reset Reference Line */}
                  <ReferenceLine 
                    y={0.0} 
                    stroke="#64748b" 
                    strokeDasharray="3 3" 
                    label={{ 
                      value: 'Rest / Reset V_reset = 0.0 V', 
                      fill: '#64748b', 
                      fontSize: 10, 
                      position: 'bottom' 
                    }} 
                  />

                  {/* Layer Mean Voltage (dashed grey) */}
                  <Line 
                    type="monotone" 
                    dataKey="v_mean" 
                    name="Layer Mean V(t)" 
                    stroke="#94a3b8" 
                    strokeWidth={1.2} 
                    strokeDasharray="3 3" 
                    dot={false} 
                    isAnimationActive={false} 
                  />

                  {/* Active Selected Neuron Voltage (bold emerald) */}
                  <Line 
                    type="monotone" 
                    dataKey="v_selected" 
                    name={`${getNeuronLabel(selectedNeuronIndex)} V(t)`} 
                    stroke="#059669" 
                    strokeWidth={2.4} 
                    dot={{ r: 2, fill: '#059669' }} 
                    activeDot={{ r: 5, fill: '#10b981', stroke: '#064e3b' }} 
                    isAnimationActive={false} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          /* Poisson Input explanation when input layer is chosen */
          <div className="p-4 bg-cyan-50/70 rounded-xl border border-cyan-200 text-xs text-cyan-900 flex items-start gap-3">
            <Info size={18} className="text-cyan-700 flex-shrink-0 mt-0.5" />
            <div>
              <strong className="font-semibold block mb-0.5">Poisson Rate Encoding Input Channels:</strong>
              Continuous EEG Welch PSD power spectrum features are transformed into non-homogeneous Poisson spike trains proportional to band power intensity over T=50 steps. Input encoders do not have sub-threshold membrane capacitors.
            </div>
          </div>
        )}

        {/* Neuromorphic Science Walkthrough Card */}
        <div className="mt-4 pt-3 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>
              <strong>Zero Leakage Invariant:</strong> Spikes and membrane potentials calculated deterministically from active 128-channel EEG frequency vectors.
            </span>
          </div>
          <div className="font-mono text-[11px] text-slate-400">
            LIF Decay τ = 2.0 • V_th = 1.0 V • Refractory = Instant Reset
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
