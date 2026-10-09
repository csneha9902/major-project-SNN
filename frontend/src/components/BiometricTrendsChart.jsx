import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid, Area, AreaChart, ComposedChart } from 'recharts';
import { useCallback } from 'react';
import { BarChart3 } from 'lucide-react';
import ChartTooltip from './ui/ChartTooltip';

const BiometricTooltip = ({ active, payload, label, onHover, onHoverEnd }) => {
  if (!active || !payload || !payload[0]?.payload) {
    if (onHoverEnd) onHoverEnd();
    return null;
  }

  const dataPoint = payload[0].payload;
  if (onHover && dataPoint.cognitive_state) {
    onHover(dataPoint.cognitive_state, dataPoint.recommendation);
  }

  const items = [
    { name: 'Beta Power (13-30 Hz)', value: payload[0]?.value, color: '#16A34A' },
    { name: 'Alpha Power (8-12 Hz)', value: payload[1]?.value, color: '#059669' },
    { name: 'Heart Rate', value: payload[2]?.value, color: '#D97706' },
  ];

  return (
    <ChartTooltip
      active={active}
      payload={payload}
      label={label}
      items={items}
      cognitiveState={dataPoint.cognitive_state}
      recommendation={dataPoint.recommendation?.task}
    />
  );
};

export default function BiometricTrendsChart({ data, currentMetrics, onHover, onHoverEnd }) {
  const handleMouseMove = useCallback((state) => {
    if (state && state.activePayload && state.activePayload.length > 0) {
      const point = state.activePayload[0].payload;
      if (onHover && point.cognitive_state) {
        onHover(point.cognitive_state, point.recommendation);
      }
    }
  }, [onHover]);

  const handleMouseLeave = useCallback(() => {
    if (onHoverEnd) onHoverEnd();
  }, [onHoverEnd]);

  return (
    <div className="chart-card">
      <h3 className="flex items-center gap-2">
        <BarChart3 size={20} className="text-[var(--accent-cyan)] flex-shrink-0" />
        Biometric Trends
        {data.length > 0 && <span className="live-dot ml-2" />}
      </h3>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart
          data={data}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          margin={{ top: 10, right: 30, left: 0, bottom: 5 }}
        >
          <defs>
            <linearGradient id="gradBeta" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#16A34A" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#16A34A" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gradAlpha" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#059669" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#059669" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gradHR" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#D97706" stopOpacity={0.2} />
              <stop offset="100%" stopColor="#D97706" stopOpacity={0} />
            </linearGradient>
            <filter id="glowGreen">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glowMint">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glowAmber">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <CartesianGrid
            strokeDasharray="3 6"
            stroke="rgba(34,197,94,0.12)"
            vertical={false}
          />
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
            domain={[50, 120]}
            axisLine={false}
            tickLine={false}
            label={{ value: 'BPM', angle: 90, position: 'insideRight', fill: '#D97706', fontSize: 11 }}
          />
          <Tooltip
            content={<BiometricTooltip onHover={onHover} onHoverEnd={onHoverEnd} />}
          />
          <Legend
            wrapperStyle={{ paddingTop: '12px', fontSize: '0.75rem', fontFamily: "'Exo 2', sans-serif" }}
            iconType="circle"
            iconSize={8}
          />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="beta"
            fill="url(#gradBeta)"
            stroke="none"
            legendType="none"
          />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="alpha"
            fill="url(#gradAlpha)"
            stroke="none"
            legendType="none"
          />
          <Area
            yAxisId="right"
            type="monotone"
            dataKey="heartRate"
            fill="url(#gradHR)"
            stroke="none"
            legendType="none"
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="beta"
            name="Beta Power (13-30 Hz)"
            stroke="#16A34A"
            strokeWidth={2.5}
            dot={false}
            filter="url(#glowGreen)"
            animationDuration={300}
            isAnimationActive={true}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="alpha"
            name="Alpha Power (8-12 Hz)"
            stroke="#059669"
            strokeWidth={2.5}
            dot={false}
            filter="url(#glowMint)"
            animationDuration={300}
            isAnimationActive={true}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="heartRate"
            name="Heart Rate (BPM)"
            stroke="#D97706"
            strokeWidth={2}
            dot={false}
            filter="url(#glowAmber)"
            strokeDasharray="6 3"
            animationDuration={300}
            isAnimationActive={true}
          />
        </ComposedChart>
      </ResponsiveContainer>
      {currentMetrics && (
        <div className="current-values">
          <span>
            <span className="text-[var(--text-muted)] text-[0.7rem] uppercase tracking-wider block mb-0.5">Beta Power (β)</span>
            <span className="text-[#16A34A] font-mono font-bold">{currentMetrics.beta?.toFixed?.(2)}</span>
          </span>
          <span>
            <span className="text-[var(--text-muted)] text-[0.7rem] uppercase tracking-wider block mb-0.5">Alpha Power (α)</span>
            <span className="text-[#059669] font-mono font-bold">{currentMetrics.alpha?.toFixed?.(2)}</span>
          </span>
          <span>
            <span className="text-[var(--text-muted)] text-[0.7rem] uppercase tracking-wider block mb-0.5">Heart Rate</span>
            <span className="text-[#D97706] font-mono font-bold">{currentMetrics.heartRate?.toFixed?.(0)} <span className="text-[0.65rem] opacity-70">BPM</span></span>
          </span>
        </div>
      )}
    </div>
  );
}
