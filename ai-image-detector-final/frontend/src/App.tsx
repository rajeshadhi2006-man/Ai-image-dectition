import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Shield,
  CheckCircle,
  AlertTriangle,
  Zap,
  Clock,
  Sun,
  Moon,
  Database,
  HelpCircle,
  Trash2,
  X
} from 'lucide-react';
import './App.css';
import DemoModal from './DemoModal';

// Connection to Hosted Hugging Face API
const rawApiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_BASE_URL = rawApiUrl.endsWith('/') ? rawApiUrl.slice(0, -1) : rawApiUrl;

interface ScanResult {
  filename: string;
  is_ai_generated: boolean;
  confidence: number;
  global_score: number;
  prediction_label: string;
  timestamp: string;
  image_url?: string;
  forensics?: {
    ela_score: number;
    noise_variance?: number;
    is_noise_suspicious?: boolean;
    fft_analysis?: {
      peak_ratio: number;
      has_checkerboard: boolean;
    };
    structural_analysis?: {
      entropy: number;
      structural_variance_coeff: number;
      is_inconsistent: boolean;
    };
    forensic_probability: number;
    sensor_match: boolean;
    final_integrity?: number;
    branch_scores?: {
      domain: number;
      sensor: number;
      structure: number;
    };
    noise_profile?: {
      skewness?: number;
      kurtosis?: number;
      variance?: number;
      high_freq_energy?: number;
      noise_score?: number;
      is_natural_sensor?: boolean;
      gaussian_fit?: number;
      cfa_match?: number;
    };
  };
  metadata?: {
    data: Record<string, string>;
    verdict: string;
    confidence: number;
    is_ai_generated: boolean;
    has_metadata: boolean;
    has_camera_info: boolean;
    is_edited?: boolean;
    footprint_score?: number;
    footprint_breakdown?: {
      metadata: number;
      source: number;
      file_structure: number;
      sensor: number;
      artifacts: number;
    };
  };
  ml_analysis?: {
    is_ai_generated: boolean;
    confidence: number;
    raw_score: number;
  };
}

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showDemo, setShowDemo] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [activeTab, setActiveTab] = useState<'ml' | 'metadata' | 'forensics'>('ml');
  const [history, setHistory] = useState<ScanResult[]>([]);
  const [flash, setFlash] = useState<'red' | 'green' | null>(null);
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    return (localStorage.getItem('theme') as 'light' | 'dark') || 'dark';
  });
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const loadLocalHistory = () => {
    const saved = localStorage.getItem('scan_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        setHistory([]);
      }
    }
  };

  const saveToLocalHistory = (newItem: ScanResult) => {
    const updated = [newItem, ...history].slice(0, 10);
    setHistory(updated);
    localStorage.setItem('scan_history', JSON.stringify(updated));
  };

  const clearHistory = () => {
    if (window.confirm("Delete all scan history? This cannot be undone.")) {
      setHistory([]);
      localStorage.removeItem('scan_history');
    }
  };

  const deleteHistoryItem = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    const updated = history.filter((_, i) => i !== index);
    setHistory(updated);
    localStorage.setItem('scan_history', JSON.stringify(updated));
  };

  useEffect(() => {
    loadLocalHistory();
  }, []);

  const handleFile = (selectedFile: File) => {
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(selectedFile);
      setResult(null);
      setShowReport(false);
      setActiveTab('ml');
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setShowReport(false);
    setActiveTab('ml');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const speak = (text: string) => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 0.95;
        window.speechSynthesis.speak(utterance);
      };

      speak("Virtual scanning in progress.");

      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        }
      });
      const data = response.data as ScanResult;

      const normalizedResult: ScanResult = {
        ...data,
        timestamp: data.timestamp || new Date().toLocaleString(),
        prediction_label: data.is_ai_generated ? 'AI GENERATED' : 'REAL IMAGE'
      };

      setResult(normalizedResult);
      saveToLocalHistory(normalizedResult);
      setFlash(normalizedResult.is_ai_generated ? 'red' : 'green');
      setTimeout(() => setFlash(null), 1000);

      if (normalizedResult.is_ai_generated) {
        speak("alert, alert artificial intelligence generated image");
      } else {
        speak("safe the image is real");
      }
      setLoading(false);
      setShowReport(true);

    } catch (error: any) {
      console.error('Detailed Analysis Error:', error);
      let errorMsg = 'Unknown error occurred.';

      if (error.response) {
        // The server responded with a status code that falls out of the range of 2xx
        errorMsg = `Server Error (${error.response.status}): ${typeof error.response.data === 'object' ? JSON.stringify(error.response.data) : error.response.data}`;
      } else if (error.request) {
        // The request was made but no response was received
        errorMsg = `No response from server. This usually means the backend is down or there's a CORS block. Target URL: ${API_BASE_URL}`;
      } else {
        // Something happened in setting up the request that triggered an Error
        errorMsg = `Request setup failed: ${error.message}`;
      }

      alert(`Critical Connection Error!\n\n${errorMsg}\n\nTechnical Details:\n- Endpoint: ${API_BASE_URL}/predict\n- Method: POST\n- Content: Multipart/FormData`);
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="bg-blobs">
        <div className="blob-1"></div>
        <div className="blob-2"></div>
      </div>

      <header className="fade-in">
        <div className="logo-container">
          <div className="logo-icon">
            <Shield className="text-white" size={24} />
          </div>
          <div className="logo-text">SentinelVision</div>
        </div>
        <div className="nav-actions">
          <div className="theme-toggle" onClick={toggleTheme}>
            {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          </div>
          <button className="demo-btn" onClick={() => setShowDemo(true)}>
            <HelpCircle size={18} /> Help
          </button>
        </div>
      </header>

      <main className="fade-in">
        {!showReport ? (
          <div className="scanner-stage">
            <div className="hero">
              <h1>Sentinel <span className="text-gradient">Vision</span></h1>
              <p>Advanced Neural Forensics & AI Generation Detection</p>
            </div>

            <div className="main-content">
              <div className="left-panel">
                <div
                  className={`drop-zone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => !loading && fileInputRef.current?.click()}
                >
                  <div className="scanning-line"></div>
                  {!file ? (
                    <div className="upload-prompt">
                      <input
                        ref={fileInputRef}
                        type="file"
                        onChange={onFileChange}
                        style={{ display: 'none' }}
                        accept="image/*"
                      />
                      <div className="upload-icon">
                        <Upload size={32} />
                      </div>
                      <div>
                        <h3>Deep Forensic Scan</h3>
                        <p>Click or drag to connect with remote analyzer</p>
                      </div>
                    </div>
                  ) : (
                    <div className={`preview-wrapper ${loading ? 'loading' : ''}`}>
                      <img src={preview!} alt="Preview" />
                      {loading && (
                        <>
                          <div className="scanning-overlay" />
                          <div className="virtual-grid-scan" />
                          <div className="scan-status-badge">
                            <div className="scan-status-dot" />
                            VIRTUAL SCANNING ACTIVE
                          </div>
                        </>
                      )}
                      <div className="preview-actions">
                        <button
                          className="analyze-btn"
                          onClick={analyzeImage}
                          disabled={loading}
                        >
                          {loading ? "Processing Forensic Data..." : "Analyze Image"}
                        </button>
                        <button
                          onClick={reset}
                          disabled={loading}
                          className="glass-panel change-btn"
                        >
                          New Case
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="right-panel">
                <motion.div
                  className="glass-panel history-sidebar"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <div className="sidebar-header">
                    <div className="sidebar-title">
                      <Clock size={18} />
                      <h3>Scan History</h3>
                    </div>
                    {history.length > 0 && (
                      <button className="clear-history-btn" onClick={clearHistory} title="Clear All">
                        <Trash2 size={14} /> Clear
                      </button>
                    )}
                  </div>

                  <div className="history-list">
                    {history.length > 0 ? history.map((item, index) => (
                      <div key={index} className="history-item" onClick={() => { setResult(item); setShowReport(true); }}>
                        <img
                          src={item.image_url || `https://via.placeholder.com/150?text=Scan`}
                          alt="Scan"
                          className="history-thumb"
                        />
                        <div className="history-details">
                          <div className="history-name">{item.filename}</div>
                          <div className="history-meta">
                            <span className={`status-dot ${item.is_ai_generated ? 'ai' : 'real'}`}></span>
                            {item.prediction_label} • {item.timestamp}
                          </div>
                        </div>
                        <button
                          className="delete-item-btn"
                          onClick={(e) => deleteHistoryItem(e, index)}
                          title="Delete"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    )) : (
                      <div className="empty-history">
                        <p>No recent detections</p>
                      </div>
                    )}
                  </div>
                </motion.div>
              </div>
            </div>
          </div>
        ) : (
          <div className="report-stage">
            <AnimatePresence>
              {result && (
                <motion.div
                  className="analysis-container"
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 50 }}
                >

                  <div className="report-header-actions">
                    <button className="new-scan-btn" onClick={reset}>
                      <Upload size={18} /> New Scan
                    </button>
                  </div>

                  <div className="summary-signals-grid">
                    <div
                      className={`summary-box clickable ${activeTab === 'ml' ? 'active' : ''} ${result.is_ai_generated ? 'ai-theme' : 'real-theme'}`}
                      onClick={() => setActiveTab('ml')}
                    >
                      <Zap size={24} />
                      <span className="summary-label">ML Score</span>
                      <span className="summary-value">
                        {(() => {
                          const displayScore = result.global_score !== undefined
                            ? result.global_score
                            : (result.confidence !== undefined ? result.confidence * 100 : 0);
                          return `${displayScore.toFixed(1)}%`;
                        })()} {result.is_ai_generated ? 'AI GENERATED' : 'REAL IMAGE'}
                      </span>
                    </div>

                    <div
                      className={`summary-box clickable ${activeTab === 'metadata' ? 'active' : ''} ${(result.metadata?.is_ai_generated || !result.metadata?.has_metadata) ? 'ai-theme' : 'real-theme'}`}
                      onClick={() => setActiveTab('metadata')}
                    >
                      <Shield size={24} />
                      <span className="summary-label">Footprint Score</span>
                      <span className="summary-value">
                        {result.metadata?.footprint_score !== undefined ? `${result.metadata.footprint_score.toFixed(0)}/100` : 'N/A'}
                      </span>
                    </div>

                    <div
                      className={`summary-box clickable ${activeTab === 'forensics' ? 'active' : ''}`}
                      onClick={() => setActiveTab('forensics')}
                    >
                      <Database size={24} />
                      <span className="summary-label">Forensic Lab Score</span>
                      <span className="summary-value">
                        {result.forensics?.final_integrity !== undefined ? `${result.forensics.final_integrity.toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                  </div>

                  <div className="single-dashboard">
                    <AnimatePresence mode="wait">
                      {activeTab === 'ml' && (
                        <motion.div
                          key="ml-tab"
                          id="ml-analysis"
                          className={`analysis-card ml-card ${result.is_ai_generated ? 'ai-theme' : 'real-theme'}`}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 20 }}
                        >
                          <div className="card-header">
                            <div className="header-badge">
                              <Zap size={20} />
                              <span>ML Score</span>
                            </div>
                            <div className="model-name">Hugging Face API</div>
                          </div>
                          <div className="verdict-card-big modern-card">
                            <div className="final-verdict-display">
                              <div className="final-verdict-icon">
                                {result.is_ai_generated ? <AlertTriangle size={64} /> : <CheckCircle size={64} />}
                              </div>
                              <h2 className="final-verdict-title">
                                {result.is_ai_generated ? "AI GENERATED" : "REAL IMAGE"}
                              </h2>
                              <div className="final-verdict-divider"></div>
                              <p className="final-verdict-sub">
                                {result.is_ai_generated
                                  ? "Artificial Signals Detected: Logic chain indicates synthetic generation"
                                  : "Verified Physical Source: All signals confirm authentic origin"}
                              </p>
                            </div>
                          </div>
                          <div className="verdict-section">
                            <div className="verdict-icon">
                              {result.is_ai_generated ? <AlertTriangle size={48} /> : <CheckCircle size={48} />}
                            </div>
                            <h3 className="verdict-text">{result.prediction_label}</h3>
                            <div className="confidence-bar">
                              <div className="bar-fill" style={{ width: `${(result.confidence * 100)}%` }}></div>
                            </div>
                            <p className="confidence-text">
                              {(result.confidence * 100).toFixed(1)}% Confidence |
                              Raw Score: {result.ml_analysis?.raw_score !== undefined ? result.ml_analysis.raw_score.toFixed(4) : 'N/A'}
                            </p>
                          </div>
                          <div className="metrics-grid">
                            <div className="metric"><Shield size={16} /><span>ML Score</span><strong>{result.ml_analysis ? `${(result.ml_analysis.confidence * 100).toFixed(1)}%` : 'N/A'}</strong></div>
                            <div className="metric"><Zap size={16} /><span>Raw Score</span><strong>{result.ml_analysis?.raw_score !== undefined ? result.ml_analysis.raw_score.toFixed(4) : 'N/A'}</strong></div>
                            <div className="metric"><Database size={16} /><span>Digital Forensic Lab</span><strong>{result.metadata?.has_metadata ? (result.metadata?.verdict || 'Available') : 'Unknown'}</strong></div>
                          </div>
                        </motion.div>
                      )}

                      {activeTab === 'metadata' && (
                        <motion.div
                          key="meta-tab"
                          id="metadata-analysis"
                          className="metadata-card analysis-card"
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 20 }}
                        >
                          <div className="card-header">
                            <div className="header-badge"><Database size={20} /><span>FOOT PRINT SCORE</span></div>
                            <div className="model-name">EXIF Data Structure</div>
                          </div>
                          <div className="verdict-section">
                            <div className="verdict-icon">{result.metadata?.is_ai_generated ? <AlertTriangle size={48} /> : <CheckCircle size={48} />}</div>
                            <h3 className="verdict-text">{result.metadata?.verdict || 'Unknown'}</h3>
                            <div className="confidence-bar"><div className="bar-fill" style={{ width: `${(result.metadata?.confidence || 0) * 100}%` }}></div></div>
                            <p className="confidence-text">{((result.metadata?.confidence || 0) * 100).toFixed(1)}% Metadata Confidence</p>
                          </div>
                          <div className="metadata-content">
                            {result.metadata?.has_metadata ? (
                              <>
                                <div className="metadata-summary">
                                  <p>{result.metadata.has_camera_info ? '✓ Authenticity criteria met' : '⚠ Missing critical camera metadata'}</p>
                                  {result.metadata.is_edited && <p className="warning-text">⚠ Software Editing Detected: Source signature modified</p>}
                                  <p>Found {Object.keys(result.metadata.data).length} metadata fields</p>
                                </div>

                                {result.metadata.footprint_breakdown && (
                                  <div className="footprint-analysis-grid">
                                    <div className="breakdown-item-box modern-card">
                                      <span className="chip-label">Metadata Quality</span>
                                      <div className="chip-progress-bg"><div className="chip-progress-fill" style={{ width: `${(result.metadata.footprint_breakdown.metadata / 20) * 100}%` }}></div></div>
                                      <span className="chip-value">{result.metadata.footprint_breakdown.metadata}/20</span>
                                    </div>
                                    <div className="breakdown-item-box modern-card">
                                      <span className="chip-label">Source Authenticity</span>
                                      <div className="chip-progress-bg"><div className="chip-progress-fill" style={{ width: `${(result.metadata.footprint_breakdown.source / 20) * 100}%` }}></div></div>
                                      <span className="chip-value">{result.metadata.footprint_breakdown.source}/20</span>
                                    </div>
                                    <div className="breakdown-item-box modern-card">
                                      <span className="chip-label">File Structure</span>
                                      <div className="chip-progress-bg"><div className="chip-progress-fill" style={{ width: `${(result.metadata.footprint_breakdown.file_structure / 20) * 100}%` }}></div></div>
                                      <span className="chip-value">{result.metadata.footprint_breakdown.file_structure}/20</span>
                                    </div>
                                    <div className="breakdown-item-box modern-card">
                                      <span className="chip-label">Sensor Integrity</span>
                                      <div className="chip-progress-bg"><div className="chip-progress-fill" style={{ width: `${(result.metadata.footprint_breakdown.sensor / 20) * 100}%` }}></div></div>
                                      <span className="chip-value">{result.metadata.footprint_breakdown.sensor}/20</span>
                                    </div>
                                    <div className="breakdown-item-box modern-card">
                                      <span className="chip-label">Artifact Resistance</span>
                                      <div className="chip-progress-bg"><div className="chip-progress-fill" style={{ width: `${(result.metadata?.footprint_breakdown?.artifacts / 20) * 100}%` }}></div></div>
                                      <span className="chip-value">{result.metadata?.footprint_breakdown?.artifacts}/20</span>
                                    </div>
                                    <div className={`total-footprint-box modern-card ${(result.metadata?.footprint_score || 0) < 50 ? 'ai-theme' : 'real-theme'}`}>
                                      <span>Total Footprint Integrity</span>
                                      <strong>{result.metadata?.footprint_score}/100</strong>
                                    </div>
                                  </div>
                                )}

                                <div className="metadata-grid">
                                  {Object.entries(result.metadata?.data).map(([key, value]) => (
                                    <div key={key} className="metadata-item">
                                      <span className="meta-label">{key}</span>
                                      <span className="meta-value">{String(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              </>
                            ) : (
                              <div className="no-metadata-warning">
                                <AlertTriangle size={32} />
                                <div><h4>Digital Footprint Analysis</h4><p>Missing EXIF data is a strong indicator of synthetic generation.</p></div>
                              </div>
                            )}
                          </div>
                        </motion.div>
                      )}

                      {activeTab === 'forensics' && result.forensics && (
                        <motion.div
                          key="forensics-tab"
                          id="forensic-analysis"
                          className="forensic-card analysis-card"
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 20 }}
                        >
                          <div className="card-header">
                            <div className="header-badge"><Shield size={20} /><span>Digital Forensic Lab</span></div>
                            <div className="model-name">DFFT & Noise Fingerprint</div>
                          </div>
                          <div className="verdict-section">
                            <div className={`verdict-icon ${(result.forensics.fft_analysis?.has_checkerboard) ? 'ai-text' : 'real-text'}`}>
                              {result.forensics.fft_analysis?.has_checkerboard ? <AlertTriangle size={48} /> : <CheckCircle size={48} />}
                            </div>
                            <h3 className={`verdict-text ${(result.forensics.fft_analysis?.has_checkerboard) ? 'ai-text' : 'real-text'}`}>
                              {result.forensics.fft_analysis?.has_checkerboard ? "SUSPICIOUS" : "CONSISTENT"}
                            </h3>
                            <div className="confidence-bar forensic-bar-tri">
                              <div className="bar-wrapper"><span className="mini-label">Domain</span><div className="bar-bg"><div className="bar-fill domain-fill" style={{ width: `${result.forensics.branch_scores?.domain || 0}%` }}></div></div></div>
                              <div className="bar-wrapper"><span className="mini-label">Sensor</span><div className="bar-bg"><div className="bar-fill sensor-fill" style={{ width: `${result.forensics.branch_scores?.sensor || 0}%` }}></div></div></div>
                              <div className="bar-wrapper"><span className="mini-label">Structure</span><div className="bar-bg"><div className="bar-fill structure-fill" style={{ width: `${result.forensics.branch_scores?.structure || 0}%` }}></div></div></div>
                            </div>
                          </div>

                          <div className="forensic-diagnostic-log">
                            <div className="log-item">
                              <span className="log-label">Final Integrity</span>
                              <span className="log-value">{result.forensics.final_integrity?.toFixed(2)}%</span>
                            </div>
                            <div className="log-item">
                              <span className="log-label">Fourier Transform</span>
                              <span className={`log-value ${result.forensics.fft_analysis?.has_checkerboard ? 'ai-text' : 'real-text'}`}>
                                {result.forensics.fft_analysis?.has_checkerboard ? "PATTERN DETECTED" : "NATURAL SPECTRUM"}
                              </span>
                            </div>
                            <div className="log-item">
                              <span className="log-label">Structural Variance</span>
                              <span className={`log-value ${result.forensics.structural_analysis?.is_inconsistent ? 'ai-text' : 'real-text'}`}>
                                {result.forensics.structural_analysis?.is_inconsistent ? "ANOMALOUS" : "VERIFIED"}
                              </span>
                            </div>
                            <div className="log-item">
                              <span className="log-label">Noise Profile</span>
                              <span className={`log-value ${result.forensics.noise_profile?.is_natural_sensor ? 'real-text' : 'ai-text'}`}>
                                {result.forensics.noise_profile?.is_natural_sensor ? "NATURAL" : "SYNTHETIC/STRETCHED"}
                              </span>
                            </div>
                            <div className="log-item">
                              <span className="log-label">CFA Interpolation</span>
                              <span className="log-value">{(result.forensics.noise_profile?.cfa_match || 0) * 100}% Match</span>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {(() => {
                      const isDefinitivelyReal = !result.is_ai_generated;

                      return (
                        <motion.div
                          className={`verdict-card-big analysis-card ${isDefinitivelyReal ? 'real-theme' : 'ai-theme'}`}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                        >
                          <div className="card-header">
                            <div className="header-badge"><Shield size={20} /><span>Final Analysis Verdict</span></div>
                            <div className="model-name">Forensic + ML Verification</div>
                          </div>
                          <div className="final-verdict-display">
                            <h2 className="final-verdict-title">{isDefinitivelyReal ? "REAL IMAGE" : "AI GENERATED"}</h2>
                            <div className="final-verdict-divider"></div>
                            <p className="final-verdict-sub">
                              {isDefinitivelyReal ? "Verified Physical Source: All signals confirm authentic origin" : "Artificial Signals Detected: Logic chain indicates synthetic generation"}
                            </p>
                          </div>
                        </motion.div>
                      );
                    })()}
                  </div>

                  <div className="json-raw-section">
                    <details>
                      <summary>Forensic JSON Payload</summary>
                      <pre>{JSON.stringify(result, null, 2)}</pre>
                    </details>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </main>

      <DemoModal showDemo={showDemo} setShowDemo={setShowDemo} />
      {flash && <div className={`screen-flash ${flash}`} />}
    </div>
  );
};

export default App;
