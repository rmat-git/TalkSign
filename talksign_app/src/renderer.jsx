import React from 'react';
import { createRoot } from 'react-dom/client';
// === NEW: Corrected path to resolve 'index.css' from the project root context ===
import '/src/index.css'; 

import App from './app.jsx'; 

const container = document.getElementById("root");

// Ensure the container exists before attempting to render.
if (container) {
  // Create a React root and render the App component.
  const root = createRoot(container);
  root.render(<App />);
}