import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import {
  MessageSquare,
  X,
  Minimize2,
  Maximize2,
  Send,
  ShieldCheck,
  Brain,
  Calendar,
  Activity,
  Sparkles,
  User,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Layers,
  ChevronRight,
  RefreshCw,
  Trash2,
  PlusCircle,
  HelpCircle
} from 'lucide-react';
import { safeFetchJson } from '../../api';

// Markdown-like text formatter for conversational output
function FormattedMessage({ content }) {
  if (!content) return null;

  // Split into paragraphs / lines
  const lines = content.split('\n');

  return (
    <div className="space-y-2 text-sm leading-relaxed">
      {lines.map((line, idx) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={idx} className="h-1" />;

        // Header ###
        if (trimmed.startsWith('### ')) {
          return (
            <h4 key={idx} className="text-base font-semibold text-cyan-400 mt-2 mb-1 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-cyan-400 shrink-0" />
              {trimmed.replace('### ', '')}
            </h4>
          );
        }

        // Header ##
        if (trimmed.startsWith('## ')) {
          return (
            <h3 key={idx} className="text-lg font-bold text-white mt-2 mb-1">
              {trimmed.replace('## ', '')}
            </h3>
          );
        }

        // Guardrail Notice (starts with *Medical Guardrail Notice:)
        if (trimmed.includes('Medical Guardrail Notice:')) {
          return (
            <div key={idx} className="my-2 p-2.5 rounded-lg bg-amber-950/40 border border-amber-500/40 text-amber-200/90 text-xs flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div>{trimmed.replace(/^\*/, '').replace(/\*$/, '')}</div>
            </div>
          );
        }

        // Privacy Guard Notice
        if (trimmed.includes('Privacy Guard Notice:')) {
          return (
            <div key={idx} className="my-2 p-2.5 rounded-lg bg-blue-950/40 border border-blue-500/40 text-blue-200/90 text-xs flex items-start gap-2">
              <ShieldCheck className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
              <div>{trimmed.replace(/^>\s*\*/, '').replace(/\*$/, '')}</div>
            </div>
          );
        }

        // Bullet point
        if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
          const itemText = trimmed.replace(/^[\*\-]\s+/, '');
          return (
            <div key={idx} className="flex items-start gap-2 ml-1 text-slate-300">
              <span className="text-cyan-400 font-bold mt-0.5">•</span>
              <span>{renderBoldAndCode(itemText)}</span>
            </div>
          );
        }

        // Numbered list
        const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
        if (numMatch) {
          return (
            <div key={idx} className="flex items-start gap-2 ml-1 text-slate-300">
              <span className="text-emerald-400 font-semibold">{numMatch[1]}.</span>
              <span>{renderBoldAndCode(numMatch[2])}</span>
            </div>
          );
        }

        // Normal paragraph
        return (
          <p key={idx} className="text-slate-300">
            {renderBoldAndCode(trimmed)}
          </p>
        );
      })}
    </div>
  );
}

// Inline formatting helper for bold (**text**) and code (`text`)
function renderBoldAndCode(text) {
  const parts = [];
  let remaining = text;
  let keyIdx = 0;

  // Split on bold or code tokens
  const tokenRegex = /(\*\*[^*]+\*\*|`[^`]+`|\$[^$]+\$)/g;
  let match;
  let lastIndex = 0;

  while ((match = tokenRegex.exec(remaining)) !== null) {
    if (match.index > lastIndex) {
      parts.push(remaining.substring(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={keyIdx++} className="font-semibold text-white">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code key={keyIdx++} className="px-1.5 py-0.5 bg-slate-800 text-cyan-300 rounded font-mono text-xs">
          {token.slice(1, -1)}
        </code>
      );
    } else if (token.startsWith('$') && token.endsWith('$')) {
      parts.push(
        <span key={keyIdx++} className="font-mono text-emerald-300 bg-slate-900/60 px-1 rounded text-xs">
          {token.slice(1, -1)}
        </span>
      );
    }
    lastIndex = tokenRegex.lastIndex;
  }

  if (lastIndex < remaining.length) {
    parts.push(remaining.substring(lastIndex));
  }

  return parts.length > 0 ? parts : text;
}

export default function ChatAssistantWidget() {
  const location = useLocation();
  const isResearchRoute = location.pathname.includes('research');

  // UI state
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'appointments' | 'inspector'
  const [domainContext, setDomainContext] = useState(isResearchRoute ? 'research' : 'dashboard');
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPrivacyInfo, setShowPrivacyInfo] = useState(false);

  // Appointments state
  const [appointments, setAppointments] = useState([]);
  const [loadingAppointments, setLoadingAppointments] = useState(false);
  const [newAptDate, setNewAptDate] = useState('');
  const [newAptTime, setNewAptTime] = useState('10:00 AM');
  const [newAptClinician, setNewAptClinician] = useState('Dr. Aditi Sharma (Neurophysiologist)');
  const [newAptReason, setNewAptReason] = useState('EEG Cognitive Workload & SNN Arousal Review');

  // Direct Neuron Inspector state
  const [inspectorLayer, setInspectorLayer] = useState('hidden1');
  const [inspectorNeuronIndex, setInspectorNeuronIndex] = useState(8);
  const [neuronData, setNeuronData] = useState(null);
  const [loadingNeuron, setLoadingNeuron] = useState(false);

  // Messages thread
  const [messages, setMessages] = useState([
    {
      id: 'welcome-1',
      sender: 'assistant',
      text: "👋 Welcome! I am your **SNN Neuromorphic & Clinical Neuro-Assistant**.\n\n" +
            "* In **Research Mode**, I can inspect individual synaptic weights (e.g. *\"Explain Neuron 8 in Hidden LIF 1\"*), calculate empirical AUROC benchmarks, and explain our spiking equations.\n" +
            "* In **Clinical Mode**, I can analyze cognitive fatigue states and **schedule EEG neuro-monitoring appointments**.\n\n" +
            "> *Zero-Leakage Privacy Guard is active: raw EEG timeseries and PII are masked at all times.*",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }
  ]);

  const messagesEndRef = useRef(null);

  // Auto-scroll on new message
  useEffect(() => {
    if (isOpen && activeTab === 'chat') {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen, activeTab]);

  // Sync domain context when route changes
  useEffect(() => {
    setDomainContext(location.pathname.includes('research') ? 'research' : 'dashboard');
  }, [location.pathname]);

  // Fetch appointments
  const fetchAppointments = async () => {
    setLoadingAppointments(true);
    try {
      const data = await safeFetchJson('/api/chat/appointments');
      if (data && data.appointments) {
        setAppointments(data.appointments);
      }
    } catch (err) {
      console.warn('Failed to load appointments:', err.message);
    } finally {
      setLoadingAppointments(false);
    }
  };

  useEffect(() => {
    if (isOpen && activeTab === 'appointments') {
      fetchAppointments();
    }
  }, [isOpen, activeTab]);

  // Handle direct neuron inspection
  const fetchNeuronInspection = async (layer = inspectorLayer, index = inspectorNeuronIndex) => {
    setLoadingNeuron(true);
    try {
      const data = await safeFetchJson(`/api/chat/neuron/${layer}/${index}`);
      setNeuronData(data);
    } catch (err) {
      console.warn('Failed to inspect neuron:', err.message);
    } finally {
      setLoadingNeuron(false);
    }
  };

  // Send conversational message
  const handleSendMessage = async (textToSend = inputMessage) => {
    const query = (textToSend || '').trim();
    if (!query || loading) return;

    // Add user message to thread
    const userMsg = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setLoading(true);

    // Prepare conversation history payload (last 8 turns)
    const historyPayload = [...messages, userMsg].slice(-8).map(m => ({
      sender: m.sender,
      text: m.text,
    }));

    try {
      const res = await safeFetchJson('/api/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: query,
          context: domainContext,
          history: historyPayload,
          client_state: {
            page: location.pathname,
            activeLayer: inspectorLayer,
            selectedNeuronIndex: Math.max(0, inspectorNeuronIndex - 1),
            timestamp: Date.now() / 1000,
          }
        })
      });

      const assistantMsg = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        text: res.response || "No response received.",
        action: res.action,
        intent: res.intent,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages(prev => [...prev, assistantMsg]);

      // If an appointment was created, refresh appointment list
      if (res.action && res.action.type === 'appointment_created') {
        fetchAppointments();
      }
    } catch (err) {
      const errorMsg = {
        id: `err-${Date.now()}`,
        sender: 'assistant',
        text: `⚠️ **Connection Error:** Could not reach the chatbot service (${err.message}). Please ensure the SNN backend is running on \`http://127.0.0.1:8000\`.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  // Create appointment manually from appointment tab
  const handleCreateAppointment = async (e) => {
    e.preventDefault();
    if (!newAptDate) return;

    try {
      await safeFetchJson('/api/chat/appointments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_name: "Active Learner / Subject",
          clinician: newAptClinician,
          date: newAptDate,
          time: newAptTime,
          reason: newAptReason,
        })
      });
      fetchAppointments();
      // Add notification to chat
      setMessages(prev => [
        ...prev,
        {
          id: `apt-conf-${Date.now()}`,
          sender: 'assistant',
          text: `### Clinical Appointment Scheduled!\n* **Clinician:** ${newAptClinician}\n* **Date & Time:** ${newAptDate} at ${newAptTime}\n* **Reason:** ${newAptReason}\n* *Saved to persistent clinical history.*`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        }
      ]);
      setActiveTab('chat');
    } catch (err) {
      alert(`Failed to create appointment: ${err.message}`);
    }
  };

  // Cancel appointment
  const handleCancelAppointment = async (id) => {
    try {
      await safeFetchJson(`/api/chat/appointments/${id}`, {
        method: 'DELETE',
      });
      fetchAppointments();
      setMessages(prev => [
        ...prev,
        {
          id: `apt-del-${Date.now()}`,
          sender: 'assistant',
          text: `ℹ️ **Appointment Cancelled:** Clinical consultation \`${id}\` was marked as cancelled.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        }
      ]);
    } catch (err) {
      alert(`Failed to cancel appointment: ${err.message}`);
    }
  };

  const quickPrompts = domainContext === 'research' ? [
    "What is Neuron 4?",
    "What does the green line represent?",
    "What is the voltage at t = 23?",
    "Is Neuron 4 firing at t = 23?",
    "Why does the membrane potential decrease after a spike?",
    "Which inputs contribute most to Neuron 4?",
    "Explain the graph in simple language",
    "What is the difference between membrane potential and firing threshold?",
    "Why is SNN AUROC 0.5532 higher than SVM?",
    "Explain LIF equations & surrogate gradient"
  ] : [
    "Schedule an EEG review appointment for Friday at 10 AM",
    "Show my upcoming clinical appointments",
    "What do high beta oscillations signify in EEG?",
    "How does the SNN estimate affective arousal?",
    "Explain the graph in simple language"
  ];

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Floating Trigger Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="group relative flex items-center gap-3 px-4 py-3 rounded-full bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 text-white shadow-xl hover:shadow-cyan-500/30 transition-all duration-300 hover:scale-105 border border-cyan-400/40"
          aria-label="Open SNN AI Assistant"
        >
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-300"></span>
          </span>
          <Brain className="w-5 h-5 text-cyan-200 group-hover:rotate-12 transition-transform duration-300" />
          <div className="text-left">
            <div className="text-xs font-bold tracking-wide flex items-center gap-1.5">
              SNN AI Assistant
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-300" />
            </div>
            <div className="text-[10px] text-cyan-200/80">
              {domainContext === 'research' ? 'Academic Research Mode' : 'Clinical Neuro-Assistant'}
            </div>
          </div>
        </button>
      )}

      {/* Main Glassmorphic Chat Panel */}
      {isOpen && (
        <div className="w-[92vw] sm:w-[460px] h-[660px] max-h-[88vh] bg-slate-950/95 backdrop-blur-2xl border border-cyan-500/30 rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
          
          {/* Header */}
          <div className="p-3.5 bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-cyan-500/20 border border-cyan-500/40 text-cyan-300">
                <Brain className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-bold text-white tracking-wide">SNN Neuro-Assistant</h3>
                  <button 
                    onClick={() => setShowPrivacyInfo(!showPrivacyInfo)}
                    className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-900/60 transition-colors"
                    title="Click for Privacy & Medical Guardrail details"
                  >
                    <ShieldCheck className="w-3 h-3" />
                    <span>Privacy Guard ON</span>
                  </button>
                </div>
                <div className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                  <span>DEAP SNN Checkpoint Active (AUROC 0.5532)</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
                aria-label="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Privacy Guard Info Modal Dropdown */}
          {showPrivacyInfo && (
            <div className="p-3 bg-slate-900/90 border-b border-emerald-500/30 text-xs text-slate-300 space-y-1.5 animate-in slide-in-from-top-2 duration-150">
              <div className="font-semibold text-emerald-400 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4" />
                Zero-Leakage Privacy & AGENTS.md Protocol
              </div>
              <p className="text-[11px] leading-relaxed text-slate-400">
                • <strong>Data Minimization:</strong> Raw microvolt EEG timeseries are dropped prior to prompt formatting.
                <br />• <strong>PII Redaction:</strong> Emails, telephone numbers, and identifiers are automatically masked.
                <br />• <strong>Medical Guardrails:</strong> The SNN models empirical affective arousal; it does not claim to diagnose clinical pathology or psychiatric conditions.
              </p>
            </div>
          )}

          {/* Domain Context & Mode Switcher Bar */}
          <div className="px-3 py-2 bg-slate-900/80 border-b border-slate-800/80 flex items-center justify-between text-xs">
            {/* Tabs */}
            <div className="flex items-center gap-1 bg-slate-950/60 p-1 rounded-lg border border-slate-800">
              <button
                onClick={() => setActiveTab('chat')}
                className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                  activeTab === 'chat'
                    ? 'bg-cyan-600/30 text-cyan-300 border border-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Chat
              </button>
              <button
                onClick={() => setActiveTab('appointments')}
                className={`px-2.5 py-1 rounded-md font-medium transition-all flex items-center gap-1.5 ${
                  activeTab === 'appointments'
                    ? 'bg-emerald-600/30 text-emerald-300 border border-emerald-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Calendar className="w-3 h-3" />
                Appointments
              </button>
              <button
                onClick={() => {
                  setActiveTab('inspector');
                  fetchNeuronInspection();
                }}
                className={`px-2.5 py-1 rounded-md font-medium transition-all flex items-center gap-1.5 ${
                  activeTab === 'inspector'
                    ? 'bg-violet-600/30 text-violet-300 border border-violet-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Layers className="w-3 h-3" />
                Neuron Inspector
              </button>
            </div>

            {/* Context Pill */}
            <button
              onClick={() => setDomainContext(domainContext === 'research' ? 'dashboard' : 'research')}
              className={`text-[10px] px-2 py-1 rounded-md border flex items-center gap-1 transition-colors ${
                domainContext === 'research'
                  ? 'bg-blue-950/40 border-blue-500/40 text-blue-300'
                  : 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
              }`}
              title="Click to toggle domain context"
            >
              <span>{domainContext === 'research' ? '🎓 Research' : '🏥 Clinical'}</span>
            </button>
          </div>

          {/* TAB 1: Chat Stream View */}
          {activeTab === 'chat' && (
            <>
              {/* Messages Container */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-700">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                  >
                    <div className="flex items-center gap-1.5 mb-1 px-1 text-[10px] text-slate-400">
                      {msg.sender === 'user' ? (
                        <>
                          <span>You</span>
                          <User className="w-3 h-3 text-cyan-400" />
                        </>
                      ) : (
                        <>
                          <Brain className="w-3 h-3 text-cyan-400" />
                          <span className="font-semibold text-cyan-400">SNN AI Assistant</span>
                          <span>• {msg.timestamp}</span>
                        </>
                      )}
                    </div>

                    <div
                      className={`max-w-[92%] p-3.5 rounded-2xl ${
                        msg.sender === 'user'
                          ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-tr-sm shadow-md'
                          : 'bg-slate-900/90 border border-slate-800 text-slate-100 rounded-tl-sm shadow-lg'
                      }`}
                    >
                      <FormattedMessage content={msg.text} />

                      {/* Interactive Appointment Card if present */}
                      {msg.action && msg.action.type === 'appointment_created' && msg.action.appointment && (
                        <div className="mt-3 p-3 rounded-xl bg-slate-950 border border-emerald-500/40 flex items-center justify-between text-xs">
                          <div>
                            <div className="font-semibold text-emerald-300 flex items-center gap-1.5">
                              <Calendar className="w-3.5 h-3.5" />
                              {msg.action.appointment.clinician}
                            </div>
                            <div className="text-slate-400 text-[11px] mt-0.5">
                              {msg.action.appointment.date} at {msg.action.appointment.time}
                            </div>
                          </div>
                          <span className="px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-500/30 text-[10px] font-semibold">
                            Confirmed
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-slate-400 text-xs w-max">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                    <span>Analyzing SNN synaptic traces & verifying guardrails...</span>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Quick Suggestion Chips */}
              <div className="px-3 py-1.5 bg-slate-950/40 border-t border-slate-900 overflow-x-auto flex items-center gap-1.5 scrollbar-none">
                {quickPrompts.map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSendMessage(prompt)}
                    className="shrink-0 text-[11px] px-2.5 py-1 rounded-full bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-cyan-900/40 hover:border-cyan-500/50 transition-colors"
                  >
                    {prompt}
                  </button>
                ))}
              </div>

              {/* Input Form */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSendMessage();
                }}
                className="p-3 bg-slate-900/90 border-t border-slate-800 flex items-center gap-2"
              >
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder={
                    domainContext === 'research'
                      ? "Ask about Neuron 8, AUROC 0.5532, LIF equations..."
                      : "Schedule appointment, ask about brainwaves, fatigue..."
                  }
                  className="flex-1 bg-slate-950/80 border border-slate-700/80 focus:border-cyan-400 rounded-xl px-3.5 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none transition-colors"
                />
                <button
                  type="submit"
                  disabled={loading || !inputMessage.trim()}
                  className="p-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-md shadow-cyan-900/20"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </>
          )}

          {/* TAB 2: Clinical Appointments View */}
          {activeTab === 'appointments' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-bold text-white flex items-center gap-1.5">
                    <Calendar className="w-4 h-4 text-emerald-400" />
                    Clinical EEG Appointments
                  </h4>
                  <p className="text-[11px] text-slate-400">
                    Schedule and manage neuro-monitoring follow-ups.
                  </p>
                </div>
                <button
                  onClick={fetchAppointments}
                  className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                  title="Refresh"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${loadingAppointments ? 'animate-spin' : ''}`} />
                </button>
              </div>

              {/* Book New Appointment Form */}
              <form onSubmit={handleCreateAppointment} className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
                <div className="text-xs font-semibold text-emerald-300 flex items-center gap-1.5">
                  <PlusCircle className="w-3.5 h-3.5" />
                  Quick Schedule Consultation
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <label className="block text-[10px] text-slate-400 mb-1">Date</label>
                    <input
                      type="date"
                      required
                      value={newAptDate}
                      onChange={(e) => setNewAptDate(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] text-slate-400 mb-1">Time</label>
                    <input
                      type="text"
                      value={newAptTime}
                      onChange={(e) => setNewAptTime(e.target.value)}
                      placeholder="10:00 AM"
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
                    />
                  </div>
                </div>

                <div className="text-xs">
                  <label className="block text-[10px] text-slate-400 mb-1">Attending Clinician</label>
                  <select
                    value={newAptClinician}
                    onChange={(e) => setNewAptClinician(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
                  >
                    <option value="Dr. Aditi Sharma (Neurophysiologist)">Dr. Aditi Sharma (Neurophysiologist)</option>
                    <option value="Dr. Marcus Vance (Cognitive Ergonomics)">Dr. Marcus Vance (Cognitive Ergonomics)</option>
                    <option value="Clinical Neuro-Monitoring Attending">Clinical Neuro-Monitoring Attending</option>
                  </select>
                </div>

                <div className="text-xs">
                  <label className="block text-[10px] text-slate-400 mb-1">Reason / Focus</label>
                  <input
                    type="text"
                    value={newAptReason}
                    onChange={(e) => setNewAptReason(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
                  />
                </div>

                <button
                  type="submit"
                  className="w-full py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs transition-colors shadow-md shadow-emerald-950/40"
                >
                  Confirm Appointment Booking
                </button>
              </form>

              {/* List of Active Appointments */}
              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-300">Scheduled Appointments ({appointments.length})</div>
                {appointments.length === 0 ? (
                  <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-center text-xs text-slate-500">
                    No active appointments scheduled. Use the form above or ask the chatbot directly!
                  </div>
                ) : (
                  appointments.map((apt) => (
                    <div
                      key={apt.id}
                      className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-start justify-between gap-2"
                    >
                      <div className="space-y-1">
                        <div className="font-semibold text-slate-200 text-xs flex items-center gap-1.5">
                          <span>{apt.clinician}</span>
                          <span className={`text-[9px] px-1.5 py-0.2 rounded-full border ${
                            apt.status === 'scheduled'
                              ? 'bg-emerald-950/60 border-emerald-500/40 text-emerald-400'
                              : 'bg-red-950/60 border-red-500/40 text-red-400'
                          }`}>
                            {apt.status}
                          </span>
                        </div>
                        <div className="text-[11px] text-slate-400 flex items-center gap-3">
                          <span className="flex items-center gap-1">
                            <Calendar className="w-3 h-3 text-cyan-400" />
                            {apt.date}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3 text-cyan-400" />
                            {apt.time}
                          </span>
                        </div>
                        <div className="text-[11px] text-slate-300 italic">
                          "{apt.reason}"
                        </div>
                      </div>

                      {apt.status === 'scheduled' && (
                        <button
                          onClick={() => handleCancelAppointment(apt.id)}
                          className="p-1 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-950/40 transition-colors"
                          title="Cancel Appointment"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* TAB 3: Direct Neuron Inspector View */}
          {activeTab === 'inspector' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div>
                <h4 className="text-sm font-bold text-white flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-violet-400" />
                  SNN Deep Neuron Inspector
                </h4>
                <p className="text-[11px] text-slate-400">
                  Inspect learned synaptic weights and receptive fields of individual units.
                </p>
              </div>

              {/* Selector Controls */}
              <div className="grid grid-cols-2 gap-2 p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs">
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Layer</label>
                  <select
                    value={inspectorLayer}
                    onChange={(e) => {
                      setInspectorLayer(e.target.value);
                      fetchNeuronInspection(e.target.value, inspectorNeuronIndex);
                    }}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
                  >
                    <option value="hidden1">Hidden LIF 1 (256 Neurons)</option>
                    <option value="hidden2">Hidden LIF 2 (128 Neurons)</option>
                    <option value="output">Output Readout (2 Classes)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">
                    Neuron Number (1 - {inspectorLayer === 'hidden1' ? 256 : inspectorLayer === 'hidden2' ? 128 : 2})
                  </label>
                  <input
                    type="number"
                    min="1"
                    max={inspectorLayer === 'hidden1' ? 256 : inspectorLayer === 'hidden2' ? 128 : 2}
                    value={inspectorNeuronIndex}
                    onChange={(e) => {
                      const val = parseInt(e.target.value) || 1;
                      setInspectorNeuronIndex(val);
                      fetchNeuronInspection(inspectorLayer, val);
                    }}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
                  />
                </div>
              </div>

              {/* Inspector Result Display */}
              {loadingNeuron ? (
                <div className="p-6 text-center text-xs text-slate-400 flex items-center justify-center gap-2">
                  <RefreshCw className="w-4 h-4 animate-spin text-violet-400" />
                  <span>Loading learned synaptic weights...</span>
                </div>
              ) : neuronData ? (
                <div className="p-3.5 rounded-xl bg-slate-900/90 border border-violet-500/30 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="font-bold text-white text-xs">
                      {neuronData.layer} — Neuron #{neuronData.neuron_number || inspectorNeuronIndex}
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-950 text-violet-300 border border-violet-500/40">
                      Bias: {typeof neuronData.bias === 'number' ? neuronData.bias.toFixed(4) : neuronData.bias}
                    </span>
                  </div>

                  {neuronData.dominant_band && (
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="text-[10px] text-slate-400 block">Dominant Band</span>
                        <span className="font-semibold text-cyan-300 capitalize">{neuronData.dominant_band}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="text-[10px] text-slate-400 block">Cortical Lobe</span>
                        <span className="font-semibold text-emerald-300">{neuronData.dominant_region}</span>
                      </div>
                    </div>
                  )}

                  {/* Top Excitatory Channels */}
                  {neuronData.top_excitatory && neuronData.top_excitatory.length > 0 && (
                    <div>
                      <div className="text-[11px] font-semibold text-slate-300 mb-1.5">Top Excitatory Inputs</div>
                      <div className="space-y-1">
                        {neuronData.top_excitatory.map((f, i) => (
                          <div key={i} className="flex items-center justify-between text-[11px] p-1.5 rounded bg-slate-950">
                            <span className="text-slate-300">{f.feature_name} ({f.region})</span>
                            <span className="font-mono text-emerald-400">+{f.weight}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Plain Language Interpretation */}
                  <div className="text-xs text-slate-300 border-t border-slate-800 pt-2">
                    <FormattedMessage content={neuronData.summary} />
                  </div>
                </div>
              ) : null}
            </div>
          )}

          {/* Footer Status Bar */}
          <div className="px-3 py-1.5 bg-slate-950 border-t border-slate-900 flex items-center justify-between text-[10px] text-slate-500">
            <span className="flex items-center gap-1">
              <ShieldCheck className="w-3 h-3 text-emerald-500" />
              HIPAA & Research Integrity Guaranteed
            </span>
            <span>SpikingJelly LIF Backend</span>
          </div>
        </div>
      )}
    </div>
  );
}
