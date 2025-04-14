import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) return resolve(); // already loaded

    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(`Failed to load: ${src}`);
    document.body.appendChild(script);
  });
}

async function bootstrap() {
  try {
    await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.min.js');
    await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js');
    await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js');

    console.log("✅ All MediaPipe scripts loaded");

    ReactDOM.createRoot(document.getElementById('root')).render(<App />);
  } catch (err) {
    console.error("❌ MediaPipe load failed:", err);
  }
}

bootstrap();

import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

