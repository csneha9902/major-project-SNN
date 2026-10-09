import React, { useState, useEffect } from 'react';
import { 
  Calendar as CalendarIcon, 
  ChevronLeft, 
  ChevronRight, 
  Lock,
  Radio,
  CheckCircle2, 
  Clock,
  Sparkles, 
  FileText,
  HeartPulse,
  RefreshCw
} from 'lucide-react';

// Helper to format date strings YYYY-MM-DD
const formatDateKey = (d) => {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

// Check if a date is strictly in the future (after today)
const isFutureDate = (d, todayRef = new Date()) => {
  const todayMidnight = new Date(todayRef);
  todayMidnight.setHours(0, 0, 0, 0);
  const targetMidnight = new Date(d);
  targetMidnight.setHours(0, 0, 0, 0);
  return targetMidnight > todayMidnight;
};

// Load persistent progress history from localStorage, defaulting to today's active session
const getInitialProgressLogs = (todayRef = new Date()) => {
  try {
    const saved = localStorage.getItem('snn_learning_sessions');
    if (saved) {
      const parsed = JSON.parse(saved);
      if (typeof parsed === 'object' && parsed !== null) {
        return parsed;
      }
    }
  } catch (e) {
    console.warn('Failed to parse saved calendar logs:', e);
  }

  const key = formatDateKey(todayRef);
  return {
    [key]: {
      state: 'Active Session',
      label: 'Active EEG Monitoring Session',
      avgBeta: 0.65,
      avgAlpha: 0.58,
      avgHeartRate: 72,
      sessionHours: 'Active',
      stressSpikes: 0,
      breathingBreaks: 0,
      snnScore: '--',
      notes: 'Active EEG monitoring session. Real SNN arousal classification and band powers recorded live during stream or file analysis.'
    }
  };
};

export default function ProgressCalendar() {
  const [todayState, setTodayState] = useState(() => new Date());
  const [currentMonthDate, setCurrentMonthDate] = useState(() => new Date(todayState.getFullYear(), todayState.getMonth(), 1));
  const [selectedDate, setSelectedDate] = useState(() => todayState);
  const [progressLogs, setProgressLogs] = useState(() => getInitialProgressLogs(todayState));
  const [customNote, setCustomNote] = useState('');

  // Framework to automatically trigger date rollover after 12:00 AM (midnight)
  useEffect(() => {
    let timerId;

    const checkAndRollover = () => {
      const now = new Date();
      if (
        now.getDate() !== todayState.getDate() ||
        now.getMonth() !== todayState.getMonth() ||
        now.getFullYear() !== todayState.getFullYear()
      ) {
        const newToday = new Date();
        setTodayState(newToday);

        // If selected date was tracking previous today, automatically advance to new today
        setSelectedDate(prevSel => {
          if (formatDateKey(prevSel) === formatDateKey(todayState)) {
            return newToday;
          }
          return prevSel;
        });

        // Keep current month view updated if month changed at midnight
        setCurrentMonthDate(prevMonth => {
          if (prevMonth.getMonth() !== newToday.getMonth() || prevMonth.getFullYear() !== newToday.getFullYear()) {
            return new Date(newToday.getFullYear(), newToday.getMonth(), 1);
          }
          return prevMonth;
        });
      }
    };

    const scheduleMidnightCheck = () => {
      const now = new Date();
      const nextMidnight = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1, 0, 0, 1);
      const msUntilMidnight = Math.max(1000, nextMidnight.getTime() - now.getTime());

      timerId = setTimeout(() => {
        checkAndRollover();
        scheduleMidnightCheck();
      }, msUntilMidnight);
    };

    scheduleMidnightCheck();

    // Also check when tab wakes up or regains focus after midnight
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        checkAndRollover();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearTimeout(timerId);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [todayState]);

  const year = currentMonthDate.getFullYear();
  const month = currentMonthDate.getMonth();

  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  // Calendar matrix computation
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const firstDayOfWeek = new Date(year, month, 1).getDay(); // 0 = Sun, 1 = Mon...

  const handlePrevMonth = () => {
    setCurrentMonthDate(new Date(year, month - 1, 1));
  };

  const handleNextMonth = () => {
    setCurrentMonthDate(new Date(year, month + 1, 1));
  };

  const handleSelectQuickDate = (type) => {
    const d = new Date(todayState);
    if (type === 'yesterday') d.setDate(todayState.getDate() - 1);
    if (type === 'tomorrow') d.setDate(todayState.getDate() + 1);
    setSelectedDate(d);
    setCurrentMonthDate(new Date(d.getFullYear(), d.getMonth(), 1));
  };

  const selectedKey = formatDateKey(selectedDate);
  const selectedIsFuture = isFutureDate(selectedDate, todayState);

  const selectedLog = progressLogs[selectedKey] || (selectedIsFuture ? {
    state: 'Upcoming',
    label: 'Upcoming Date',
    avgBeta: 0,
    avgAlpha: 0,
    avgHeartRate: 0,
    sessionHours: '--',
    stressSpikes: 0,
    breathingBreaks: 0,
    snnScore: '--',
    notes: 'Upcoming date — cognitive session logging will activate when this date arrives.'
  } : {
    state: 'No Session',
    label: 'No Session Recorded',
    avgBeta: 0,
    avgAlpha: 0,
    avgHeartRate: 0,
    sessionHours: '--',
    stressSpikes: 0,
    breathingBreaks: 0,
    snnScore: '--',
    notes: 'No active EEG session recorded for this date.'
  });

  const handleUpdateState = (newState) => {
    if (selectedIsFuture) return; // Block editing future dates
    setProgressLogs(prev => {
      const updated = {
        ...prev,
        [selectedKey]: {
          ...selectedLog,
          state: newState,
          label: newState === 'High Arousal' ? 'High Arousal Session' : newState === 'Low Arousal' ? 'Low Arousal Focus' : 'Baseline Rest'
        }
      };
      try {
        localStorage.setItem('snn_learning_sessions', JSON.stringify(updated));
      } catch (e) {
        console.warn('Failed to save session log:', e);
      }
      return updated;
    });
  };

  const handleAddNote = (e) => {
    e.preventDefault();
    if (selectedIsFuture || !customNote.trim()) return;
    setProgressLogs(prev => {
      const updated = {
        ...prev,
        [selectedKey]: {
          ...selectedLog,
          notes: customNote.trim()
        }
      };
      try {
        localStorage.setItem('snn_learning_sessions', JSON.stringify(updated));
      } catch (e) {
        console.warn('Failed to save note:', e);
      }
      return updated;
    });
    setCustomNote('');
  };

  // State color badges
  const getStateColorClass = (st) => {
    switch (st) {
      case 'High Arousal':
      case 'Stressed': return 'bg-amber-500/20 text-amber-800 border-amber-500/40';
      case 'Low Arousal':
      case 'Focused': return 'bg-emerald-600/20 text-emerald-800 border-emerald-500/40';
      case 'Active Session': return 'bg-blue-600/20 text-blue-800 border-blue-500/40';
      case 'Upcoming': return 'bg-slate-300/40 text-slate-700 border-slate-300';
      case 'No Session':
      case 'Neutral':
      default: return 'bg-slate-400/20 text-slate-700 border-slate-400/40';
    }
  };

  const getDateDotClass = (st) => {
    switch (st) {
      case 'High Arousal':
      case 'Stressed': return 'bg-amber-500 shadow-[0_0_8px_rgba(217,119,6,0.6)]';
      case 'Low Arousal':
      case 'Focused': return 'bg-emerald-600 shadow-[0_0_8px_rgba(16,185,129,0.6)]';
      case 'Active Session': return 'bg-blue-600 shadow-[0_0_8px_rgba(37,99,235,0.6)]';
      case 'Neutral': return 'bg-slate-400';
      default: return 'bg-transparent';
    }
  };

  // Date label text for selected header
  const getSelectedLabelText = () => {
    const todayMidnight = new Date(todayState);
    todayMidnight.setHours(0,0,0,0);
    const selMidnight = new Date(selectedDate);
    selMidnight.setHours(0,0,0,0);
    const diff = Math.round((selMidnight - todayMidnight) / 86400000);
    const formattedDate = selectedDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    if (diff === 0) return `Today (${formattedDate})`;
    if (diff === -1) return `Yesterday (${formattedDate})`;
    if (diff === 1) return `Tomorrow (${formattedDate})`;
    return formattedDate;
  };

  return (
    <div className="bg-[var(--bg-surface)] rounded-2xl border border-[var(--border-subtle)] p-5 shadow-sm mt-6">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 mb-4 border-b border-[var(--border-subtle)]">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-blue-50 text-[var(--accent-cyan)]">
            <CalendarIcon size={22} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-heading font-bold text-base text-[var(--text-primary)]">Cognitive Progress & Health Calendar</h3>
              <span 
                className="px-2 py-0.5 text-[0.62rem] font-bold rounded-full bg-blue-600/10 text-blue-700 border border-blue-500/30 flex items-center gap-1"
                title="Automated 12:00 AM Midnight Rollover Active"
              >
                <Clock size={10} className="animate-spin" style={{ animationDuration: '8s' }} /> 12 AM Rollover Active
              </span>
            </div>
            <p className="text-xs text-[var(--text-muted)]">Track daily neural stress, focus sessions & SNN progress reports</p>
          </div>
        </div>

        {/* Quick Date Selectors */}
        <div className="flex items-center gap-1.5 bg-[var(--bg-subtle)] p-1 rounded-xl border border-[var(--border-subtle)] text-xs font-medium">
          <button 
            onClick={() => handleSelectQuickDate('yesterday')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              formatDateKey(selectedDate) === formatDateKey(new Date(todayState.getTime() - 86400000))
                ? 'bg-[var(--accent-cyan)] text-white font-bold'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            Yesterday
          </button>
          <button 
            onClick={() => handleSelectQuickDate('today')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              formatDateKey(selectedDate) === formatDateKey(todayState)
                ? 'bg-[var(--accent-cyan)] text-white font-bold'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            Today (Live Session)
          </button>
          <button 
            onClick={() => handleSelectQuickDate('tomorrow')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              formatDateKey(selectedDate) === formatDateKey(new Date(todayState.getTime() + 86400000))
                ? 'bg-[var(--accent-cyan)] text-white font-bold'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            Tomorrow
          </button>
        </div>
      </div>

      {/* Grid Container */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Side: Calendar Grid */}
        <div className="lg:col-span-7 bg-[var(--bg-subtle)]/40 p-4 rounded-xl border border-[var(--border-subtle)]">
          {/* Month Header Navigation */}
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-heading font-bold text-sm text-[var(--text-primary)]">
              {monthNames[month]} {year}
            </h4>
            <div className="flex items-center gap-1">
              <button 
                onClick={handlePrevMonth}
                className="p-1.5 rounded-lg hover:bg-[var(--bg-surface)] text-[var(--text-secondary)] border border-transparent hover:border-[var(--border-subtle)]"
              >
                <ChevronLeft size={18} />
              </button>
              <button 
                onClick={handleNextMonth}
                className="p-1.5 rounded-lg hover:bg-[var(--bg-surface)] text-[var(--text-secondary)] border border-transparent hover:border-[var(--border-subtle)]"
              >
                <ChevronRight size={18} />
              </button>
            </div>
          </div>

          {/* Days of Week Header */}
          <div className="grid grid-cols-7 text-center text-[0.7rem] font-bold text-[var(--text-muted)] uppercase mb-2">
            <span>Sun</span>
            <span>Mon</span>
            <span>Tue</span>
            <span>Wed</span>
            <span>Thu</span>
            <span>Fri</span>
            <span>Sat</span>
          </div>

          {/* Days Grid */}
          <div className="grid grid-cols-7 gap-1.5">
            {/* Blank offset days */}
            {Array.from({ length: firstDayOfWeek }).map((_, idx) => (
              <div key={`blank-${idx}`} className="h-11 rounded-lg opacity-20" />
            ))}

            {/* Actual Month Days */}
            {Array.from({ length: daysInMonth }).map((_, idx) => {
              const dayNum = idx + 1;
              const dateObj = new Date(year, month, dayNum);
              const key = formatDateKey(dateObj);
              const log = progressLogs[key];
              const isSelected = key === selectedKey;
              const isTodayDate = key === formatDateKey(todayState);
              const isFuture = isFutureDate(dateObj, todayState);

              return (
                <button
                  key={key}
                  onClick={() => setSelectedDate(dateObj)}
                  className={`relative h-11 rounded-xl flex flex-col items-center justify-between p-1.5 transition-all text-xs ${
                    isSelected 
                      ? 'ring-2 ring-[var(--accent-cyan)] bg-[var(--bg-surface)] font-bold shadow-md' 
                      : isFuture
                        ? 'opacity-60 bg-transparent text-[var(--text-muted)] border border-dashed border-[var(--border-subtle)] hover:opacity-100 hover:bg-[var(--bg-surface)]/50'
                        : 'hover:bg-[var(--bg-surface)]/80 text-[var(--text-primary)] border border-transparent'
                  } ${
                    isTodayDate 
                      ? 'border-2 border-blue-500 shadow-[0_0_12px_rgba(0,98,255,0.35)] bg-blue-50/20 font-black' 
                      : ''
                  }`}
                >
                  {/* Today Badge / Indicator */}
                  <div className="flex items-center justify-between w-full">
                    <span className={`${isTodayDate ? 'text-blue-700 font-black' : ''}`}>{dayNum}</span>
                    {isTodayDate && (
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-ping" title="Active Live Session" />
                    )}
                    {isFuture && (
                      <Lock size={10} className="text-[var(--text-muted)] opacity-70" title="Future Date (Locked)" />
                    )}
                  </div>

                  {log?.state ? (
                    <span className={`w-2 h-2 rounded-full mb-0.5 ${getDateDotClass(log.state)}`} title={`${log.state} (${log.label})`} />
                  ) : (
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-200 opacity-20 mb-0.5" />
                  )}
                </button>
              );
            })}
          </div>

          {/* Calendar Status Legend */}
          <div className="flex flex-wrap items-center justify-around gap-2 mt-4 pt-3 border-t border-[var(--border-subtle)] text-xs text-[var(--text-secondary)] font-medium">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-[0_0_6px_rgba(217,119,6,0.6)]" />
              <span>High Arousal (Amber)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 shadow-[0_0_6px_rgba(16,185,129,0.6)]" />
              <span>Low Arousal (Green)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-400" />
              <span>Baseline (Grey)</span>
            </div>
            <div className="flex items-center gap-1.5 font-semibold text-blue-700">
              <span className="w-3 h-3 rounded border-2 border-blue-500 bg-blue-50/40" />
              <span>Active Session Outline</span>
            </div>
          </div>
        </div>

        {/* Right Side: Selected Date Progress Report */}
        <div className="lg:col-span-5 flex flex-col justify-between bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)]">
          <div>
            {/* Header for Selected Date */}
            <div className="flex items-center justify-between pb-3 border-b border-[var(--border-subtle)] mb-3">
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[0.68rem] uppercase font-bold tracking-wider text-[var(--accent-cyan)]">
                    Progress Report Keeper
                  </span>
                  {selectedIsFuture && (
                    <span className="px-1.5 py-0.5 text-[0.62rem] font-bold rounded bg-amber-100 text-amber-800 border border-amber-300 flex items-center gap-1">
                      <Lock size={10} /> Locked
                    </span>
                  )}
                </div>
                <h4 className="font-heading font-bold text-sm text-[var(--text-primary)]">
                  {getSelectedLabelText()}
                </h4>
              </div>
              <span className={`px-2.5 py-1 text-xs font-bold rounded-full border ${getStateColorClass(selectedLog.state)}`}>
                {selectedLog.state}
              </span>
            </div>

            {/* Quick Metrics Cards Grid */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              <div className="p-2.5 rounded-xl bg-[var(--bg-subtle)] border border-[var(--border-subtle)]">
                <span className="text-[0.68rem] text-[var(--text-muted)] block mb-0.5">Study Duration</span>
                <span className="font-mono font-bold text-sm text-[var(--text-primary)]">{selectedLog.sessionHours}</span>
              </div>
              <div className="p-2.5 rounded-xl bg-[var(--bg-subtle)] border border-[var(--border-subtle)]">
                <span className="text-[0.68rem] text-[var(--text-muted)] block mb-0.5">SNN Health Score</span>
                <span className="font-mono font-bold text-sm text-blue-600">
                  {selectedLog.snnScore !== '--' ? `${selectedLog.snnScore}%` : '--'}
                </span>
              </div>
              <div className="p-2.5 rounded-xl bg-[var(--bg-subtle)] border border-[var(--border-subtle)]">
                <span className="text-[0.68rem] text-[var(--text-muted)] block mb-0.5">Avg Heart Rate</span>
                <span className="font-mono font-bold text-sm text-amber-600">
                  {selectedLog.avgHeartRate ? `${selectedLog.avgHeartRate} BPM` : '--'}
                </span>
              </div>
              <div className="p-2.5 rounded-xl bg-[var(--bg-subtle)] border border-[var(--border-subtle)]">
                <span className="text-[0.68rem] text-[var(--text-muted)] block mb-0.5">Stress Spikes</span>
                <span className="font-mono font-bold text-sm text-red-600">
                  {selectedLog.stressSpikes ? `${selectedLog.stressSpikes} Episodes` : '0'}
                </span>
              </div>
            </div>

            {/* Situation Summary & Daily Log */}
            <div className="mb-4">
              <span className="text-xs font-bold text-[var(--text-secondary)] uppercase tracking-wider block mb-1.5">
                Session Log & Observations
              </span>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed bg-[var(--bg-subtle)]/70 p-3 rounded-lg border border-[var(--border-subtle)]">
                {selectedLog.notes}
              </p>
            </div>

            {/* Mark or Override State Buttons */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-[var(--text-secondary)] uppercase tracking-wider">
                  Log Date Status
                </span>
                {selectedIsFuture && (
                  <span className="text-[0.68rem] text-amber-700 font-medium">Future dates cannot be edited yet</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  disabled={selectedIsFuture}
                  onClick={() => handleUpdateState('High Arousal')}
                  className={`flex-1 py-1.5 text-xs font-semibold rounded-lg border transition-all ${
                    selectedIsFuture 
                      ? 'opacity-40 cursor-not-allowed bg-[var(--bg-subtle)] text-[var(--text-muted)] border-[var(--border-subtle)]'
                      : (selectedLog.state === 'High Arousal' || selectedLog.state === 'Stressed')
                        ? 'bg-amber-500 text-white border-amber-600 shadow-sm'
                        : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-amber-700 border-[var(--border-subtle)]'
                  }`}
                >
                  High Arousal (Amber)
                </button>
                <button
                  disabled={selectedIsFuture}
                  onClick={() => handleUpdateState('Low Arousal')}
                  className={`flex-1 py-1.5 text-xs font-semibold rounded-lg border transition-all ${
                    selectedIsFuture 
                      ? 'opacity-40 cursor-not-allowed bg-[var(--bg-subtle)] text-[var(--text-muted)] border-[var(--border-subtle)]'
                      : (selectedLog.state === 'Low Arousal' || selectedLog.state === 'Focused')
                        ? 'bg-emerald-600 text-white border-emerald-600 shadow-sm'
                        : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-emerald-700 border-[var(--border-subtle)]'
                  }`}
                >
                  Low Arousal (Green)
                </button>
                <button
                  disabled={selectedIsFuture}
                  onClick={() => handleUpdateState('Baseline')}
                  className={`flex-1 py-1.5 text-xs font-semibold rounded-lg border transition-all ${
                    selectedIsFuture 
                      ? 'opacity-40 cursor-not-allowed bg-[var(--bg-subtle)] text-[var(--text-muted)] border-[var(--border-subtle)]'
                      : (selectedLog.state === 'Baseline' || selectedLog.state === 'Neutral' || selectedLog.state === 'No Session')
                        ? 'bg-slate-500 text-white border-slate-600 shadow-sm'
                        : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-slate-600 border-[var(--border-subtle)]'
                  }`}
                >
                  Baseline (Grey)
                </button>
              </div>
            </div>
          </div>

          {/* Add Note Form */}
          <form onSubmit={handleAddNote} className="pt-3 border-t border-[var(--border-subtle)]">
            <div className="flex items-center gap-2">
              <input
                type="text"
                disabled={selectedIsFuture}
                value={customNote}
                onChange={(e) => setCustomNote(e.target.value)}
                placeholder={selectedIsFuture ? "Editing disabled for future dates..." : "Add custom clinical observation or EEG note..."}
                className="flex-1 px-3 py-1.5 text-xs bg-[var(--bg-subtle)] border border-[var(--border-subtle)] rounded-lg text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)] disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <button
                type="submit"
                disabled={selectedIsFuture}
                className="px-3 py-1.5 bg-[var(--accent-cyan)] text-white font-semibold text-xs rounded-lg hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Save
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
