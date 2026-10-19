import { useState, useEffect } from 'react';
import { Upload, File, X, Sparkles, Activity } from 'lucide-react';
import GlowButton from './ui/GlowButton';

const API_BASE = import.meta.env.VITE_API_URL || "";

export default function FileUpload({ onUploadSuccess }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');
  const [demoSamples, setDemoSamples] = useState([]);
  const [loadingDemo, setLoadingDemo] = useState(false);
  const getAuthHeaders = () => ({});

  useEffect(() => {
    fetchDemoSamples();
  }, []);

  const fetchDemoSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/demo-samples`);
      if (res.ok) {
        const data = await res.json();
        setDemoSamples(data.samples || []);
      }
    } catch (err) {
      console.warn("Failed to fetch demo samples:", err);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileSelect = (file) => {
    const lower = file.name.toLowerCase();
    if (!lower.endsWith('.edf') && !lower.endsWith('.csv')) {
      setError('Please select an EDF or CSV file');
      return;
    }
    if (file.size > 200 * 1024 * 1024) {
      setError('File size must be less than 200MB');
      return;
    }
    setSelectedFile(file);
    setError('');
  };

  const handleFileInput = (e) => {
    if (e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await res.json();
      setSelectedFile(null);
      if (onUploadSuccess) {
        onUploadSuccess(data.upload_id);
      }
    } catch (err) {
      setError(err.message || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  const handleLoadDemoSample = (uploadId) => {
    setLoadingDemo(true);
    if (onUploadSuccess) {
      onUploadSuccess(uploadId);
    }
    setLoadingDemo(false);
  };

  const handleRemove = () => {
    setSelectedFile(null);
    setError('');
  };

  return (
    <div className="file-upload-container">
      <h3 className="upload-title">Upload EDF or CSV File for Analysis</h3>
      <div
        className={`upload-dropzone ${dragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {selectedFile ? (
          <div className="file-selected">
            <File className="file-icon" />
            <div className="file-info">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
            </div>
            <button className="btn-remove" onClick={handleRemove} disabled={uploading}>
              <X size={20} />
            </button>
          </div>
        ) : (
          <>
            <Upload className="upload-icon" />
            <p className="upload-text">
              Drag and drop an EDF/CSV file here, or <label className="upload-label">browse<input type="file" accept=".edf,.EDF,.csv,.CSV" onChange={handleFileInput} className="file-input" /></label>
            </p>
            <p className="upload-hint">Maximum file size: 200MB</p>
          </>
        )}
      </div>

      {error && <div className="upload-error">{error}</div>}

      {selectedFile && (
        <button
          className="btn-upload"
          onClick={handleUpload}
          disabled={uploading}
        >
          {uploading ? 'Processing...' : 'Upload & Analyze'}
        </button>
      )}

      {/* Synthetic Demo Datasets Section */}
      <div className="demo-samples-section mt-6 pt-6 border-t border-[var(--border-color)]">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="text-blue-600" size={18} />
          <h4 className="font-heading font-semibold text-sm text-[var(--text-primary)]">
            Explore Pre-configured Demo Datasets
          </h4>
        </div>
        <p className="text-xs text-[var(--text-secondary)] mb-4">
          Select a synthetic clinical case study to immediately view full SNN wave decomposition, spectral FFT, and PDF export without uploading custom files.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {demoSamples.map((sample) => (
            <div
              key={sample.upload_id}
              className="demo-sample-card p-3 rounded-lg border border-[var(--border-color)] bg-[var(--bg-glass)] hover:border-blue-500/50 transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-blue-700 font-mono flex items-center gap-1">
                    <Activity size={12} />
                    {sample.filename}
                  </span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-800 font-medium">
                    {sample.duration}s
                  </span>
                </div>
                <h5 className="font-medium text-xs text-[var(--text-primary)] mb-1">
                  {sample.name}
                </h5>
                <p className="text-[11px] text-[var(--text-secondary)] line-clamp-2 mb-3">
                  {sample.summary}
                </p>
              </div>
              <GlowButton
                variant="secondary"
                onClick={() => handleLoadDemoSample(sample.upload_id)}
                disabled={loadingDemo}
                className="w-full text-xs py-1.5"
              >
                Load Sample Case
              </GlowButton>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
