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
  Moon
} from 'lucide-react';
import './App.css';
import DemoModal from './DemoModal';

// Connection to Hosted Hugging Face API
const API_BASE_URL = '';

interface ScanResult {
  filename: string;
  is_ai_generated: boolean;
  confidence: number;
  prediction_label: string;
  timestamp: string;
  image_url?: string;
}

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showDemo, setShowDemo] = useState(false);
  const [history, setHistory] = useState<ScanResult[]>([]);
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

  // Local storage used for history since hosted API doesn't support /history endpoint
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
    }
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

      // Some APIs might not return timestamp/prediction_label in exact format
      // We normalize here to ensure the UI stays stable
      const normalizedResult: ScanResult = {
        ...data,
        timestamp: data.timestamp || new Date().toLocaleString(),
        prediction_label: data.prediction_label || (data.is_ai_generated ? 'Artificial' : 'Authentic')
      };

      setResult(normalizedResult);
      saveToLocalHistory(normalizedResult);

      setTimeout(() => {
        if (normalizedResult.is_ai_generated) {
          speak("alert it is artificial image genrated image");
        } else {
          speak("safe the image is real");
        }
      }, 1000);

    } catch (error: any) {
      console.error('Analysis error detail:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown connection error';
      alert(`Connection failed: ${errorMessage}. Please ensure the Hugging Face space is active and check your internet connection.`);
    } finally {
      // Small delay to let the virtual scanning effect be visible
      setTimeout(() => {
        setLoading(false);
      }, 2500);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
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
            <Zap size={16} /> Help
          </button>
        </div>
      </header>

      <main className="main-content">
        <div className="layout-grid">
          <div className="left-panel">
            <div className="upload-card">
              {!file ? (
                <div
                  className={`drop-zone ${dragActive ? 'dragging' : ''}`}
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
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
                      onClick={analyzeImage}
                      disabled={loading}
                      className="glass-panel analyze-btn"
                    >
                      Analyze the image
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

            <AnimatePresence>
              {result && (
                <motion.div
                  className="analysis-container"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                >
                  <div className="single-dashboard">
                    <motion.div
                      className={`analysis-card ml-card ${result.is_ai_generated ? 'ai-theme' : 'real-theme'}`}
                    >
                      <div className="card-header">
                        <div className="header-badge">
                          <Zap size={20} />
                          <span>Remote Node Analysis</span>
                        </div>
                        <div className="model-name">Hugging Face API</div>
                      </div>

                      <div className="verdict-section">
                        <div className="verdict-icon">
                          {result.is_ai_generated ? <AlertTriangle size={48} /> : <CheckCircle size={48} />}
                        </div>
                        <h3 className="verdict-text">{result.prediction_label}</h3>
                        <div className="confidence-bar">
                          <div className="bar-fill" style={{ width: `${(result.confidence * 100)}%` }}></div>
                        </div>
                        <p className="confidence-text">{(result.confidence * 100).toFixed(1)}% Forensic Confidence</p>
                      </div>

                      <div className="metrics-grid">
                        <div className="metric">
                          <Shield size={16} />
                          <span>Status</span>
                          <strong>Verified</strong>
                        </div>
                        <div className="metric">
                          <Zap size={16} />
                          <span>Source</span>
                          <strong>HF Cloud</strong>
                        </div>
                      </div>
                    </motion.div>
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

          <div className="right-panel">
            <motion.div
              className="glass-panel history-sidebar"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="sidebar-header">
                <Clock size={18} />
                <h3>Scan History</h3>
              </div>

              <div className="history-list">
                {history.length > 0 ? history.map((item, index) => (
                  <div key={index} className="history-item" onClick={() => setResult(item)}>
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
      </main>

      <DemoModal showDemo={showDemo} setShowDemo={setShowDemo} />
    </div>
  );
};

export default App;
