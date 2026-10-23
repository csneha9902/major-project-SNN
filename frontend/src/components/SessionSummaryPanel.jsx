import { Brain, BarChart3 } from 'lucide-react';
import { useState } from 'react';
import GlowButton from './ui/GlowButton';
import { safeFetchJson } from '../api';

export default function SessionSummaryPanel({ isRunning }) {
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [error, setError] = useState('');

  const fetchSummary = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await safeFetchJson('/api/session-summary');
      if (data?.summary) {
        setSummary(data.summary);
        setExpanded(true);
      } else {
        setError('No summary data available. Start the simulation and let it run for a few seconds.');
      }
    } catch (err) {
      console.error('Failed to fetch summary:', err.message);
      setError(err.message || 'Failed to fetch session summary.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="summary-panel">
      <div className="panel-header">
        <Brain className="icon-brain" />
        <h3>Cognitive Session Summary</h3>
      </div>
      {expanded && summary ? (
        <div className="summary-content">
          <p>{summary}</p>
          <GlowButton 
            variant="cyan" 
            onClick={() => { setExpanded(false); setSummary(''); fetchSummary(); }}
            disabled={loading}
            className="mt-3"
          >
            Refresh Summary
          </GlowButton>
        </div>
      ) : (
        <div className="summary-actions">
          {error && (
            <div className="summary-error">
              <p>{error}</p>
            </div>
          )}
          {!isRunning && !error && (
            <p className="summary-hint">Start the simulation to generate a session summary.</p>
          )}
          <GlowButton 
            variant="violet" 
            onClick={fetchSummary} 
            disabled={loading || !isRunning}
          >
            <BarChart3 size={16} />
            {loading ? 'Loading...' : 'Get Session Summary'}
          </GlowButton>
        </div>
      )}
    </div>
  );
}
