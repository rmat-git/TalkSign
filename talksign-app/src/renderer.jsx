import React, { useState, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';

// --- IMPORT ELECTRON IPC ---
const { ipcRenderer } = window.require('electron');

const API_URL = "http://localhost:5000";

// --- ICONS ---
const MaleIcon = () => (<svg width="20" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="7" r="4" /><path d="M6 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2" /></svg>);
const FemaleIcon = () => (<svg width="20" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="7" r="4" /><path d="M12 11v10" /><path d="M9 11l-3 10h12l-3-10" /></svg>);
const MinimizeIcon = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="4 14 10 14 10 20" /><polyline points="20 10 14 10 14 4" /><line x1="14" y1="10" x2="21" y2="3" /><line x1="3" y1="21" x2="10" y2="14" /></svg>);
const ExpandIcon = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 3 21 3 21 9" /><polyline points="9 21 3 21 3 15" /><line x1="21" y1="3" x2="14" y2="10" /><line x1="3" y1="21" x2="10" y2="14" /></svg>);

// --- THEME ICONS ---
const SunIcon = () => (<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>);
const MoonIcon = () => (<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>);

// --- HELPERS ---
const KeyCap = ({ label }) => (
  <span className="inline-flex items-center justify-center min-w-[24px] h-6 px-1.5 bg-white border border-gray-300 rounded shadow-[0_1px_0_0_rgba(0,0,0,0.1)] text-[10px] font-bold text-gray-700 uppercase tracking-wider font-sans select-none mx-0.5">
    {label}
  </span>
);

const ShortcutRow = ({ label, keys }) => (
  <div className="flex justify-between items-center py-1">
    <span className="text-[10px] font-medium opacity-60 uppercase tracking-tight">{label}</span>
    <div className="flex items-center">
      {keys.map((k, i) => (
        <React.Fragment key={i}>
          <KeyCap label={k} />
        </React.Fragment>
      ))}
    </div>
  </div>
);

const App = () => {
  // --- STATE ---
  const [isCompact, setIsCompact] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

  const [status, setStatus] = useState({
    mode: "alphabet", prediction: "...", confidence: 0.0, sentence: "",
    hands_detected: false, is_cooldown: false, llm_processing: false 
  });
  const [isVCamActive, setIsVCamActive] = useState(false);
  const [devices, setDevices] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [settings, setSettings] = useState({
    textEnabled: true, textSize: 40, textColor: "#FFFFFF", textPosition: 0, 
    textEffect: "none", ttsEnabled: true, voiceType: "F"
  });
  const [videoUrl, setVideoUrl] = useState(`${API_URL}/video_feed`);

  // --- ACTIONS ---
  const toggleCompactMode = () => {
    const newMode = !isCompact;
    setIsCompact(newMode);
    ipcRenderer.send('app:compact-mode', newMode);
  };

  useEffect(() => {
    fetch(`${API_URL}/cameras`).then(res => res.json()).then(data => { setDevices(data); if (data.length > 0) setSelectedCameraId(data[0].id); }).catch(err => console.error(err));
  }, []);
  useEffect(() => {
    const interval = setInterval(() => { fetch(`${API_URL}/status`).then(res => res.json()).then(data => setStatus(data)).catch(() => {}); }, 100);
    return () => clearInterval(interval);
  }, []);
  const sendCommand = async (endpoint, method = "POST", body = null) => { try { const opts = { method }; if (body) { opts.headers = { 'Content-Type': 'application/json' }; opts.body = JSON.stringify(body); } const res = await fetch(`${API_URL}${endpoint}`, opts); return await res.json(); } catch (err) { console.error(err); } };
  const setMode = () => { const newMode = status.mode === "alphabet" ? "word" : "alphabet"; sendCommand(`/control/set_mode/${newMode}`); };
  const clearSentence = () => sendCommand('/control/clear_sentence');
  const handleBackspace = () => sendCommand('/control/backspace'); 
  const performCameraSwitch = (newId) => { setSelectedCameraId(newId); setVideoUrl(""); sendCommand('/control/switch_camera', 'POST', { camera_id: newId }).then(() => { setTimeout(() => setVideoUrl(`${API_URL}/video_feed?t=${new Date().getTime()}`), 100); }); };
  const handleCameraSelect = (e) => { performCameraSwitch(parseInt(e.target.value)); };
  const toggleVCam = () => { sendCommand('/control/toggle_vcam'); setIsVCamActive(prev => !prev); };
  const updateSetting = (key, value) => { setSettings(prev => ({ ...prev, [key]: value })); sendCommand('/control/update_settings', 'POST', { [key]: value }); };
  
  const handleKeyDown = useCallback((event) => {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;
    if (event.altKey && event.key.toLowerCase() === 'c') { event.preventDefault(); if (devices.length > 1) { const currentIndex = devices.findIndex(d => d.id === selectedCameraId); const nextIndex = (currentIndex + 1) % devices.length; const nextDeviceId = devices[nextIndex].id; performCameraSwitch(nextDeviceId); } return; }
    if (event.altKey && event.key.toLowerCase() === 'm') { event.preventDefault(); updateSetting('ttsEnabled', !settings.ttsEnabled); return; }
    switch (event.key) { case 'Enter': event.preventDefault(); sendCommand('/control/trigger_gemini'); break; case 'Tab': event.preventDefault(); setMode(); break; case 'Backspace': handleBackspace(); break; case 'Escape': clearSentence(); break; default: break; }
  }, [status.mode, devices, selectedCameraId, settings.ttsEnabled]); 

  useEffect(() => { window.addEventListener('keydown', handleKeyDown); return () => window.removeEventListener('keydown', handleKeyDown); }, [handleKeyDown]);

  const getStateColor = () => { if (!status.hands_detected) return "bg-red-500"; if (status.is_cooldown) return "bg-yellow-500"; return "bg-green-500"; };
  const getLLMColor = () => status.llm_processing ? "bg-yellow-500 animate-bounce" : "bg-green-500";

  // --- DYNAMIC STYLES ---
  const mainBgClass = isDarkMode ? "bg-gray-900" : "bg-[#F5F5F0]";
  const panelBgClass = isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200";
  const headerBgClass = isDarkMode ? "border-gray-700" : "border-gray-100";
  
  const mainTextClass = isDarkMode ? "text-white" : "text-[#111]";
  const subTextClass = isDarkMode ? "text-gray-400" : "text-gray-500";
  const labelTextClass = isDarkMode ? "text-gray-300" : "text-gray-700";

  const inputClass = isDarkMode 
    ? "bg-gray-700 border-gray-600 text-white hover:border-gray-500" 
    : "bg-white border-gray-300 text-black hover:border-gray-400";

  // --- DYNAMIC CSS FOR SLIDERS ---
  const sliderStyle = `
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      height: 16px;
      width: 16px;
      border-radius: 50%;
      background: ${isDarkMode ? '#ffffff' : '#000000'} !important;
      cursor: pointer;
      margin-top: -6px; 
    }
    input[type="range"]::-webkit-slider-runnable-track {
      width: 100%;
      height: 4px;
      cursor: pointer;
      background: ${isDarkMode ? '#4b5563' : '#e5e7eb'};
      border-radius: 9999px;
    }
  `;

  // --- RENDER ---
  return (
    <div className={`h-screen w-screen overflow-hidden box-border font-['Rubik'] transition-colors duration-300 flex items-center justify-center px-4 
      ${isCompact ? "bg-gray-900 border-2 border-blue-500" : `${mainBgClass} ${mainTextClass}`}`}>
      
      {/* INJECT DYNAMIC SLIDER CSS */}
      <style>{sliderStyle}</style>

      {/* ========================== */}
      {/* COMPACT MODE OVERLAY UI */}
      {/* ========================== */}
      {isCompact && (
        <div className="flex items-center gap-6 w-full text-white select-none">
           <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${getStateColor()}`}></div>
              <span className="text-xs font-bold uppercase tracking-widest text-gray-400">{status.hands_detected ? "Active" : "No Hand"}</span>
           </div>
           
           <div className="h-4 w-px bg-gray-700"></div>
           <span className="text-xs font-bold uppercase text-blue-400">{status.mode} Mode</span>
           <span className="ml-4 text-sm font-mono text-yellow-300 truncate max-w-[150px]">{status.sentence || "Listening..."}</span>

           <div className="flex gap-3 ml-auto mr-4">
              <div className="flex items-center gap-1 opacity-70"><KeyCap label="TAB" /><span className="text-[9px] font-bold">MODE</span></div>
              <div className="flex items-center gap-1 opacity-70"><KeyCap label="ENT" /><span className="text-[9px] font-bold">AI</span></div>
              <div className="flex items-center gap-1 opacity-70"><KeyCap label="ESC" /><span className="text-[9px] font-bold">CLR</span></div>
           </div>

           <button onClick={toggleCompactMode} className="p-2 bg-gray-700 hover:bg-gray-600 rounded text-white border border-gray-600 shadow-sm" title="Expand View">
             <ExpandIcon />
           </button>
        </div>
      )}

      {/* ========================== */}
      {/* NORMAL MODE UI */}
      {/* ========================== */}
      {!isCompact && (
        <div className="flex gap-6 h-full py-6"> {/* Added h-full and vertical padding to container */}
          
          {/* --- LEFT COLUMN --- */}
          {/* FIX: Added h-full to make Left Column stretch fully */}
          <div className="flex flex-col gap-3 h-full"> 
            
            {/* HEADER */}
            <div className="flex justify-between items-end w-[693px] pb-1">
                <h1 className="text-4xl font-light tracking-tight">
                    TALK<span className="font-bold">SIGN</span>
                </h1>
                <div className="flex items-center gap-6 pb-2">
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${getStateColor()}`}></div>
                      <span className={`text-[10px] font-bold tracking-widest uppercase ${subTextClass}`}>
                          {status.hands_detected ? "Hands Active" : "No Hands"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${getLLMColor()}`}></div>
                      <span className={`text-[10px] font-bold tracking-widest uppercase ${subTextClass}`}>
                          {status.llm_processing ? "Processing..." : "Gemini"}
                      </span>
                    </div>
                </div>
            </div>

            {/* VIDEO AREA */}
            <div className="w-[693px] h-[390px] bg-black shadow-sm relative overflow-hidden flex items-center justify-center shrink-0 rounded-lg">
              {videoUrl && <img src={videoUrl} alt="Stream" className="w-full h-full object-contain" />}
            </div>

            {/* CONTROLS ROW */}
            <div className="flex justify-between items-center h-12 w-[693px]">
                <div className="flex gap-2">
                    {/* CAMERA SELECT */}
                    <div className="relative w-40">
                        <select value={selectedCameraId} onChange={handleCameraSelect}
                            className={`w-full text-xs font-medium uppercase tracking-wide px-3 py-2.5 shadow-sm appearance-none rounded-md focus:outline-none cursor-pointer truncate pr-6 border ${inputClass}`}>
                            {devices.map((device) => (
                                <option key={device.id} value={device.id}>{device.name}</option>
                            ))}
                        </select>
                        <div className="absolute right-2 top-1/2 transform -translate-y-1/2 pointer-events-none opacity-60">
                            <svg width="8" height="6" viewBox="0 0 10 6" fill="none"><path d="M1 1L5 5L9 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                        </div>
                    </div>

                    {/* MODE BUTTON */}
                    <button onClick={setMode} className={`w-32 text-xs font-medium uppercase tracking-wide py-2.5 shadow-sm transition-colors rounded-md border ${inputClass} hover:opacity-90`}>
                      {status.mode}
                    </button>

                    {/* MINIMIZE BUTTON */}
                    <button 
                      onClick={toggleCompactMode} 
                      className={`w-10 flex items-center justify-center shadow-sm transition-colors rounded-md border ${inputClass} hover:opacity-90`}
                      title="Minimize to Overlay"
                    >
                      <MinimizeIcon />
                    </button>
                </div>

                {/* START BUTTON */}
                <button onClick={toggleVCam} className={`w-40 text-white text-xs font-bold uppercase tracking-wide py-2.5 shadow-sm transition-colors rounded-md ${isVCamActive ? "bg-red-600 hover:bg-red-700" : "bg-[#00C853] hover:bg-[#00a844]"}`}>
                  {isVCamActive ? "STOP" : "START"}
                </button>
            </div>

            {/* SHORTCUTS PANEL */}
            {/* FIX: Changed h-32 to flex-1 so it stretches to align bottom */}
            <div className={`w-[693px] shadow-sm flex flex-col flex-1 rounded-lg overflow-hidden border ${panelBgClass}`}>
                <div className={`flex items-center px-6 h-8 border-b ${isDarkMode ? 'bg-gray-700/50 border-gray-700' : 'bg-gray-50/50 border-gray-100'}`}>
                    <span className={`text-[10px] font-bold uppercase tracking-widest ${subTextClass}`}>Keyboard Shortcuts</span>
                </div>
                <div className="flex-1 p-3 px-6 grid grid-cols-3 gap-x-8 gap-y-1 items-center">
                    <div className={`flex flex-col justify-center space-y-2 border-r pr-4 ${isDarkMode ? "border-gray-700" : "border-gray-100"}`}>
                        <ShortcutRow label="Clear All" keys={['Esc']} />
                        <ShortcutRow label="Delete Last" keys={['←']} />
                    </div>
                    <div className={`flex flex-col justify-center space-y-2 border-r pr-4 ${isDarkMode ? "border-gray-700" : "border-gray-100"}`}>
                        <ShortcutRow label="Switch Mode" keys={['Tab']} />
                        <ShortcutRow label="Process (AI)" keys={['Enter']} />
                    </div>
                    <div className="flex flex-col justify-center space-y-2">
                        <ShortcutRow label="Cycle Camera" keys={['Alt', 'C']} />
                        <ShortcutRow label="Toggle Mute" keys={['Alt', 'M']} />
                    </div>
                </div>
            </div>
          </div>

          {/* --- RIGHT COLUMN (Customization) --- */}
          <div className={`w-[300px] flex flex-col shadow-sm h-full rounded-lg overflow-hidden border ${panelBgClass}`}>
            
            {/* --- HEADER + THEME TOGGLE --- */}
            <div className={`h-14 flex items-center justify-between px-6 border-b shrink-0 ${headerBgClass}`}>
               <h2 className="text-lg font-bold uppercase tracking-wider">CUSTOMIZATION</h2>
               
               {/* THEME TOGGLE BUTTON (Clear BG, Color Icon) */}
               <button 
                 onClick={() => setIsDarkMode(!isDarkMode)} 
                 className={`p-1.5 rounded-md transition-colors ${isDarkMode ? "text-white hover:bg-gray-700" : "text-black hover:bg-gray-200"}`}
                 title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
               >
                 {isDarkMode ? <SunIcon /> : <MoonIcon />}
               </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6">
                
                {/* TEXT SUBTITLES */}
                <div className="space-y-6">
                    <h3 className={`text-[10px] font-bold uppercase tracking-widest ${subTextClass}`}>Text Subtitles</h3>
                    
                    {/* SHOW TEXT CHECKBOX */}
                    <div className="flex justify-between items-center">
                        <span className={`text-xs font-medium uppercase ${labelTextClass}`}>Show Text</span>
                        <input 
                          type="checkbox" 
                          checked={settings.textEnabled} 
                          onChange={(e) => updateSetting('textEnabled', e.target.checked)} 
                          className="h-4 w-4 rounded-sm cursor-pointer"
                          style={{ accentColor: isDarkMode ? 'white' : 'black' }}
                        />
                    </div>
                    
                    {/* SIZE SLIDER */}
                    <div>
                        <div className="flex justify-between mb-2"><span className={`text-xs font-medium uppercase ${labelTextClass}`}>Size</span><span className="text-xs font-bold">{settings.textSize}px</span></div>
                        <input 
                          type="range" min="10" max="100" 
                          value={settings.textSize} 
                          onChange={(e) => updateSetting('textSize', parseInt(e.target.value))} 
                          className="w-full h-1 appearance-none bg-transparent cursor-pointer"
                        />
                    </div>
                    
                    {/* POSITION SLIDER */}
                    <div>
                        <div className="flex justify-between mb-2"><span className={`text-xs font-medium uppercase ${labelTextClass}`}>Position</span><span className="text-xs font-bold">{settings.textPosition}%</span></div>
                        <input 
                          type="range" min="0" max="90" 
                          value={settings.textPosition} 
                          onChange={(e) => updateSetting('textPosition', parseInt(e.target.value))} 
                          className="w-full h-1 appearance-none bg-transparent cursor-pointer"
                        />
                    </div>
                    
                    {/* COLOR PICKER */}
                    <div>
                        <span className={`text-xs font-medium uppercase block mb-3 ${labelTextClass}`}>Color</span>
                        <div className="flex gap-3 justify-center">
                        {['#FFFFFF', '#000000', '#FF3B30', '#007AFF', '#FFD700'].map(c => (
                            <button key={c} onClick={() => updateSetting('textColor', c)} 
                            className={`w-8 h-8 rounded-md border shadow-sm ${settings.textColor === c ? (isDarkMode ? 'ring-2 ring-white ring-offset-2 ring-offset-gray-800' : 'ring-2 ring-black ring-offset-2 ring-offset-white') : 'border-gray-300'}`} style={{backgroundColor: c}} />
                        ))}
                        </div>
                    </div>
                </div>
                
                <hr className={`border-t my-8 ${isDarkMode ? "border-gray-700" : "border-gray-100"}`} />
                
                {/* TEXT-TO-SPEECH */}
                <div className="space-y-6">
                    <h3 className={`text-[10px] font-bold uppercase tracking-widest ${subTextClass}`}>Text-to-Speech</h3>
                    
                    {/* ENABLE TTS CHECKBOX */}
                    <div className="flex justify-between items-center">
                        <span className={`text-xs font-medium uppercase ${labelTextClass}`}>Enable TTS</span>
                        <input 
                          type="checkbox" 
                          checked={settings.ttsEnabled} 
                          onChange={(e) => updateSetting('ttsEnabled', e.target.checked)} 
                          className="h-4 w-4 rounded-sm cursor-pointer"
                          style={{ accentColor: isDarkMode ? 'white' : 'black' }}
                        />
                    </div>
                    
                    {/* VOICE MODEL BUTTONS */}
                    <div>
                          <span className={`text-xs font-medium uppercase block mb-3 ${labelTextClass}`}>Voice Model</span>
                          <div className="flex gap-2">
                            {['M', 'F'].map(type => (
                              <button key={type} onClick={() => updateSetting('voiceType', type)} 
                                className={`flex-1 flex items-center justify-center py-2 border transition-colors rounded-md 
                                ${settings.voiceType === type 
                                  ? (isDarkMode ? 'bg-white text-black border-white' : 'bg-black text-white border-black') 
                                  : (isDarkMode ? 'bg-gray-700 text-white border-gray-600 hover:bg-gray-600' : 'bg-white border-gray-100 hover:bg-gray-100 text-black')}`}
                              >
                                {type === 'M' ? <MaleIcon /> : <FemaleIcon />}
                              </button>
                            ))}
                          </div>
                    </div>
                </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
const root = createRoot(document.getElementById('root'));
root.render(<App />);