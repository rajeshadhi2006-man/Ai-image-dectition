import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface DemoModalProps {
    showDemo: boolean;
    setShowDemo: (show: boolean) => void;
}

const DemoModal: React.FC<DemoModalProps> = ({ showDemo, setShowDemo }) => {
    return (
        <AnimatePresence>
            {showDemo && (
                <motion.div
                    className="demo-overlay"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={() => setShowDemo(false)}
                >
                    <motion.div
                        className="demo-modal"
                        initial={{ scale: 0.9, y: 20 }}
                        animate={{ scale: 1, y: 0 }}
                        exit={{ scale: 0.9, y: 20 }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="demo-header">
                            <h2>Expert Detection Guide</h2>
                            <button className="close-demo" onClick={() => setShowDemo(false)}>&times;</button>
                        </div>
                        <div className="demo-steps">
                            <div className="demo-step">
                                <div className="step-num">1</div>
                                <div className="step-content">
                                    <h3>Upload Image</h3>
                                    <p>Drag any JPEG or PNG file into the drop zone or click to browse.</p>
                                </div>
                            </div>
                            <div className="demo-step">
                                <div className="step-num">2</div>
                                <div className="step-content">
                                    <h3>Neural Scan</h3>
                                    <p>Click "Analyze Image". Our CNN Neural Network will perform a deep pixel scan for AI artifacts.</p>
                                </div>
                            </div>
                            <div className="demo-step">
                                <div className="step-num">3</div>
                                <div className="step-content">
                                    <h3>Review Result</h3>
                                    <p>Examine the verdict and confidence scores. Authentic images show high real-world coherence.</p>
                                </div>
                            </div>
                            <div className="demo-step">
                                <div className="step-num">4</div>
                                <div className="step-content">
                                    <h3>Log Activity</h3>
                                    <p>Your previous detections are saved in the sidebar for historical forensic tracking.</p>
                                </div>
                            </div>
                        </div>
                        <button className="demo-close-btn" onClick={() => setShowDemo(false)}>Initialize Sentinel</button>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default DemoModal;
