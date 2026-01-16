import React, { useState, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';

const API_URL = "http://localhost:5000";

// --- ICONS ---
const MaleIcon = () => (
  <svg width="20" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="7" r="4" />
    <path d="M6 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2" />
  </svg>
);

const FemaleIcon = () => (
  <svg width="20" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="7" r="4" />
    <path d="M12 11v10" />
    <path d="M9 11l-3 10h12l-3-10" />
  </svg>
);

// --- HELPER: KEYCAP (Subtle Radius) ---
const KeyCap = ({ label }) => (
  <span className="inline-flex items-center justify-center min-w-[24px] h-6 px-1.5 
    bg-white border border-gray-300 rounded shadow-[0_1px_0_0_rgba(0,0,0,0.1)] 
    text-[10px] font-bold text-gray-700 uppercase tracking-wider font-sans select-none mx-0.5">
    {label}
  </span>
);

// --- HELPER: SHORTCUT ROW ---
const ShortcutRow = ({ label, keys }) => (
  <div className="flex justify-between items-center py-1">
    <span className="text-[10px] font-medium text-gray-500 uppercase tracking-tight">{label}</span>
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
  const [status, setStatus] = useState({
    mode: "alphabet", prediction: "...", confidence: 0.0, sentence: "",
    hands_detected: false, is_cooldown: false, llm_processing: false 
  });
  const [isVCamActive, setIsVCamActive] = useState(false);
  const [devices, setDevices] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [settings, setSettings] = useState({
    textEnabled: true, textSize: 40, textColor: "#FFFFFF", textPosition: 10, 
    textEffect: "none", ttsEnabled: true, voiceType: "F"
  });
  const [videoUrl, setVideoUrl] = useState(`${API_URL}/video_feed`);

  // --- EFFECT: Fetch Cameras ---
  useEffect(() => {
    fetch(`${API_URL}/cameras`)
      .then(res => res.json())
      .then(data => {
        setDevices(data);
        if (data.length > 0) setSelectedCameraId(data[0].id);
      })
      .catch(err => console.error(err));
  }, []);

  // --- EFFECT: Polling ---
  useEffect(() => {
    const interval = setInterval(() => {
      fetch(`${API_URL}/status`)
        .then(res => res.json())
        .then(data => setStatus(data))
        .catch(() => {});
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // --- ACTIONS ---
  const sendCommand = async (endpoint, method = "POST", body = null) => {
    try {
      const opts = { method };
      if (body) {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(body);
      }
      const res = await fetch(`${API_URL}${endpoint}`, opts);
      return await res.json();
    } catch (err) { console.error(err); }
  };

  const setMode = () => {
    const newMode = status.mode === "alphabet" ? "word" : "alphabet";
    sendCommand(`/control/set_mode/${newMode}`);
  };
  const clearSentence = () => sendCommand('/control/clear_sentence');
  const handleBackspace = () => sendCommand('/control/backspace'); 

  const performCameraSwitch = (newId) => {
    setSelectedCameraId(newId);
    setVideoUrl(""); 
    sendCommand('/control/switch_camera', 'POST', { camera_id: newId }).then(() => {
        setTimeout(() => setVideoUrl(`${API_URL}/video_feed?t=${new Date().getTime()}`), 100); 
    });
  };

  const handleCameraSelect = (e) => {
    performCameraSwitch(parseInt(e.target.value));
  };

  const toggleVCam = () => {
    sendCommand('/control/toggle_vcam');
    setIsVCamActive(prev => !prev);
  };

  const updateSetting = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    sendCommand('/control/update_settings', 'POST', { [key]: value });
  };

  // --- KEYBOARD ---
  const handleKeyDown = useCallback((event) => {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;

    if (event.altKey && event.key.toLowerCase() === 'c') {
        event.preventDefault();
        if (devices.length > 1) {
            const currentIndex = devices.findIndex(d => d.id === selectedCameraId);
            const nextIndex = (currentIndex + 1) % devices.length;
            const nextDeviceId = devices[nextIndex].id;
            performCameraSwitch(nextDeviceId);
        }
        return;
    }

    if (event.altKey && event.key.toLowerCase() === 'm') {
        event.preventDefault();
        updateSetting('ttsEnabled', !settings.ttsEnabled);
        return;
    }

    switch (event.key) {
        case 'Enter':
            event.preventDefault();
            sendCommand('/control/trigger_gemini');
            break;
        case 'Tab':
            event.preventDefault(); 
            setMode();
            break;
        case 'Backspace':
            handleBackspace();
            break;
        case 'Escape':
            clearSentence();
            break;
        default:
            break;
    }
  }, [status.mode, devices, selectedCameraId, settings.ttsEnabled]); 

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const getStateColor = () => {
      if (!status.hands_detected) return "bg-red-500";
      if (status.is_cooldown) return "bg-yellow-500";
      return "bg-green-500";
  };
  const getLLMColor = () => status.llm_processing ? "bg-yellow-500 animate-bounce" : "bg-green-500";

  return (
    <div className="flex h-screen w-screen bg-[#F5F5F0] text-[#111] p-6 gap-6 overflow-hidden box-border justify-center font-['Rubik']">
      
      {/* --- LEFT COLUMN --- */}
      <div className="flex flex-col gap-3">
        {/* HEADER */}
        <div className="flex justify-between items-end w-[693px] pb-1">
            <h1 className="text-4xl font-light tracking-tight text-black">
                TALK<span className="font-bold">SIGN</span>
            </h1>
            <div className="flex items-center gap-6 pb-2">
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${getStateColor()}`}></div>
                  <span className="text-[10px] font-bold tracking-widest uppercase text-gray-500">
                      {status.hands_detected ? "Hands Active" : "No Hands"}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${getLLMColor()}`}></div>
                  <span className="text-[10px] font-bold tracking-widest uppercase text-gray-500">
                      {status.llm_processing ? "Processing..." : "Gemini"}
                  </span>
                </div>
            </div>
        </div>

        {/* VIDEO AREA */}
        <div className="w-[693px] h-[390px] bg-black shadow-sm relative overflow-hidden flex items-center justify-center shrink-0 rounded-lg">
          {videoUrl && <img src={videoUrl} alt="Stream" className="w-full h-full object-contain" />}
        </div>

        {/* CONTROLS */}
        <div className="flex justify-between items-center h-12 w-[693px]">
            <div className="flex gap-4">
                <div className="relative w-40">
                    <select value={selectedCameraId} onChange={handleCameraSelect}
                        className="w-full bg-white border border-gray-300 hover:border-gray-400 text-black text-xs font-medium uppercase tracking-wide px-3 py-2.5 shadow-sm appearance-none rounded-md focus:outline-none focus:border-black cursor-pointer truncate pr-6">
                        {devices.map((device) => (
                            <option key={device.id} value={device.id}>{device.name}</option>
                        ))}
                    </select>
                    <div className="absolute right-2 top-1/2 transform -translate-y-1/2 pointer-events-none opacity-60">
                        <svg width="8" height="6" viewBox="0 0 10 6" fill="none"><path d="M1 1L5 5L9 1" stroke="black" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    </div>
                </div>
                <button onClick={setMode} className="w-40 bg-white border border-gray-300 hover:bg-gray-50 text-black text-xs font-medium uppercase tracking-wide py-2.5 shadow-sm transition-colors rounded-md">
                  {status.mode}
                </button>
            </div>
            <button onClick={toggleVCam} className={`w-40 text-white text-xs font-bold uppercase tracking-wide py-2.5 shadow-sm transition-colors rounded-md ${isVCamActive ? "bg-red-600 hover:bg-red-700" : "bg-[#00C853] hover:bg-[#00a844]"}`}>
              {isVCamActive ? "STOP" : "START"}
            </button>
        </div>

        {/* SHORTCUTS PANEL */}
        <div className="bg-white border border-gray-200 w-[693px] shadow-sm flex flex-col h-32 rounded-lg overflow-hidden">
            <div className="flex items-center px-6 h-8 border-b border-gray-100 bg-gray-50/50">
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Keyboard Shortcuts</span>
            </div>
            <div className="flex-1 p-3 px-6 grid grid-cols-3 gap-x-8 gap-y-1">
                <div className="flex flex-col justify-center space-y-1 border-r border-gray-100 pr-4">
                    <ShortcutRow label="Clear All" keys={['Esc']} />
                    <ShortcutRow label="Delete Last" keys={['←']} />
                </div>
                <div className="flex flex-col justify-center space-y-1 border-r border-gray-100 pr-4">
                    <ShortcutRow label="Switch Mode" keys={['Tab']} />
                    <ShortcutRow label="Process (AI)" keys={['Enter']} />
                </div>
                <div className="flex flex-col justify-center space-y-1">
                    <ShortcutRow label="Cycle Camera" keys={['Alt', 'C']} />
                    <ShortcutRow label="Toggle Mute" keys={['Alt', 'M']} />
                </div>
            </div>
        </div>
      </div>

      {/* --- RIGHT COLUMN --- */}
      <div className="w-[300px] flex flex-col bg-white border border-gray-200 shadow-sm h-full rounded-lg overflow-hidden">
        <div className="h-14 flex items-center px-6 border-b border-gray-100 shrink-0">
           <h2 className="text-lg font-bold uppercase tracking-wider text-black">CUSTOMIZATION</h2>
        </div>
        
        <div className="flex-1 overflow-y-auto p-6">
            
            {/* TEXT SUBTITLES */}
            <div className="space-y-6">
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Text Subtitles</h3>
                
                <div className="flex justify-between items-center">
                    <span className="text-xs font-medium text-gray-700 uppercase">Show Text</span>
                    <input type="checkbox" checked={settings.textEnabled} onChange={(e) => updateSetting('textEnabled', e.target.checked)} className="accent-black h-4 w-4 rounded-sm" />
                </div>
                
                <div>
                    <div className="flex justify-between mb-2"><span className="text-xs font-medium text-gray-700 uppercase">Size</span><span className="text-xs font-bold">{settings.textSize}px</span></div>
                    <input type="range" min="10" max="100" value={settings.textSize} onChange={(e) => updateSetting('textSize', parseInt(e.target.value))} className="w-full h-1 bg-gray-100 rounded-lg appearance-none cursor-pointer accent-black" />
                </div>
                
                <div>
                     <div className="flex justify-between mb-2"><span className="text-xs font-medium text-gray-700 uppercase">Position</span><span className="text-xs font-bold">{settings.textPosition}%</span></div>
                    <input type="range" min="0" max="90" value={settings.textPosition} onChange={(e) => updateSetting('textPosition', parseInt(e.target.value))} className="w-full h-1 bg-gray-100 rounded-lg appearance-none cursor-pointer accent-black" />
                </div>
                
                <div>
                    <span className="text-xs font-medium text-gray-700 uppercase block mb-3">Color</span>
                    {/* CENTERED COLORS */}
                    <div className="flex gap-3 justify-center">
                    {['#FFFFFF', '#000000', '#FF3B30', '#007AFF', '#FFD700'].map(c => (
                        <button key={c} onClick={() => updateSetting('textColor', c)} 
                        className={`w-8 h-8 rounded-md border border-gray-200 shadow-sm ${settings.textColor === c ? 'ring-2 ring-black ring-offset-2' : ''}`} style={{backgroundColor: c}} />
                    ))}
                    </div>
                </div>
            </div>

            {/* SEPARATOR */}
            <hr className="border-t border-gray-100 my-8" />

            {/* TEXT-TO-SPEECH */}
            <div className="space-y-6">
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Text-to-Speech</h3>
                
                <div className="flex justify-between items-center">
                    <span className="text-xs font-medium text-gray-700 uppercase">Enable TTS</span>
                    <input type="checkbox" checked={settings.ttsEnabled} onChange={(e) => updateSetting('ttsEnabled', e.target.checked)} className="accent-black h-4 w-4 rounded-sm" />
                </div>
                
                <div>
                     <span className="text-xs font-medium text-gray-700 uppercase block mb-3">Voice Model</span>
                     <div className="flex gap-2">
                       {['M', 'F'].map(type => (
                         <button key={type} onClick={() => updateSetting('voiceType', type)} 
                         className={`flex-1 flex items-center justify-center py-2 border border-gray-100 transition-colors rounded-md ${settings.voiceType === type ? 'bg-black text-white border-black' : 'bg-white text-black hover:bg-gray-100'}`}>
                            {/* UPDATED: USES ICONS INSTEAD OF TEXT */}
                            {type === 'M' ? <MaleIcon /> : <FemaleIcon />}
                         </button>
                       ))}
                     </div>
                </div>
            </div>

        </div>
      </div>
    </div>
  );
};
const root = createRoot(document.getElementById('root'));
root.render(<App />);