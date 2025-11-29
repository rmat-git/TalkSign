import React from 'react';
// Using inline SVG icons for window controls
const MinimizeIcon = (props) => (
  <svg viewBox="0 0 10 10" className="w-4 h-4" {...props}>
    <path fill="currentColor" d="M2 4.5H8V5.5H2V4.5Z" />
  </svg>
);

const MaximizeIcon = (props) => (
  <svg viewBox="0 0 10 10" className="w-4 h-4" {...props}>
    <path fill="currentColor" d="M1 1L9 1L9 9L1 9L1 1ZM2 2L8 2L8 8L2 8L2 2Z" />
  </svg>
);

const RestoreIcon = (props) => (
  <svg viewBox="0 0 10 10" className="w-4 h-4" {...props}>
    <path fill="currentColor" d="M3 1L9 1L9 7M1 3L7 3L7 9L1 9L1 3ZM4 2L8 2L8 6L7 6L7 4L4 4L4 2Z" />
  </svg>
);

const CloseIcon = (props) => (
  <svg viewBox="0 0 10 10" className="w-4 h-4" {...props}>
    <path fill="currentColor" d="M2 2L8 8M8 2L2 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

// New Icons for Customization
const SettingsIcon = (props) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 text-gray-800" {...props}>
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1.51-1V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
    </svg>
);

const MicIcon = (props) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 text-gray-800" {...props}>
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
        <line x1="12" y1="19" x2="12" y2="23"></line>
        <line x1="8" y1="23" x2="16" y2="23"></line>
    </svg>
);

// WindowControls component remains the same
const WindowControls = ({ onMinimize, onMaximize, isMaximized, onClose }) => {
  const baseClasses = "w-10 h-8 flex items-center justify-center transition-colors duration-150 text-gray-700";
  const closeClasses = "hover:bg-red-500 hover:text-white";
  const MaximizeRestoreIcon = isMaximized ? RestoreIcon : MaximizeIcon;

  return (
    <div className="flex space-x-0">
      <button
        onClick={onMinimize}
        className={`${baseClasses} hover:bg-gray-200`}
        aria-label="Minimize"
      >
        <MinimizeIcon />
      </button>
      <button
        onClick={onMaximize}
        className={`${baseClasses} hover:bg-gray-200`}
        aria-label={isMaximized ? "Restore Down" : "Maximize"}
      >
        <MaximizeRestoreIcon />
      </button>
      <button
        onClick={onClose}
        className={`${baseClasses} ${closeClasses}`}
        aria-label="Close"
      >
        <CloseIcon />
      </button>
    </div>
  );
};

// --- NEW COMPONENT: Customization Card ---
const CustomizationCard = ({ showOnScreenText, setShowOnScreenText, textSize, setTextSize, textPosition, setTextPosition }) => {
    // Note: The draft image uses 'B' for Position which is ambiguous, using a slider for percentage position
    const handleSliderChange = (setter) => (e) => setter(Number(e.target.value));
    
    return (
        <div className="p-6 bg-white rounded-xl shadow-lg border border-gray-100">
            <div className="flex items-center space-x-2 mb-4">
                <SettingsIcon />
                <h2 className="text-xl font-bold text-gray-800">Customization</h2>
            </div>

            <div className="space-y-4">
                <div className="flex items-center justify-between">
                    <span className="text-gray-700">On-Screen Text</span>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" checked={showOnScreenText} onChange={(e) => setShowOnScreenText(e.target.checked)} className="sr-only peer" />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                        <span className="ml-3 text-sm font-medium text-gray-900">{showOnScreenText ? 'ON' : 'OFF'}</span>
                    </label>
                </div>
                
                <div>
                    <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-700">Text Size</span>
                        <span className="font-semibold text-gray-900">{textSize}</span>
                    </div>
                    <input type="range" min="10" max="48" value={textSize} onChange={handleSliderChange(setTextSize)} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg" />
                </div>
                
                <div>
                    <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-700">Position (from top)</span>
                        <span className="font-semibold text-gray-900">{textPosition}%</span>
                    </div>
                    <input type="range" min="0" max="100" value={textPosition} onChange={handleSliderChange(setTextPosition)} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg" />
                </div>
                
                <div className="flex justify-between items-center pt-2">
                    <button className="text-gray-600 hover:text-blue-600 transition-colors">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5"><path d="M21.5 2v6h-6"/><path d="M21.5 13a9 9 0 1 1-6.5-12"/></svg>
                    </button>
                    <button className="px-4 py-2 bg-green-500 text-white font-semibold rounded-lg hover:bg-green-600 transition-colors">
                        Save Changes
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- NEW COMPONENT: Text-to-Speech Card ---
const TextToSpeechCard = ({ textToSpeechEnabled, setTextToSpeechEnabled, ttsVoice, setTtsVoice }) => {
    return (
        <div className="p-6 bg-white rounded-xl shadow-lg border border-gray-100">
            <div className="flex items-center space-x-2 mb-4">
                <MicIcon />
                <h2 className="text-xl font-bold text-gray-800">Text to Speech</h2>
            </div>
            
            <div className="space-y-4">
                <div className="flex items-center justify-between">
                    <span className="text-gray-700">Text to Speech</span>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" checked={textToSpeechEnabled} onChange={(e) => setTextToSpeechEnabled(e.target.checked)} className="sr-only peer" />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                        <span className="ml-3 text-sm font-medium text-gray-900">{textToSpeechEnabled ? 'ON' : 'OFF'}</span>
                    </label>
                </div>
                
                <div>
                    <label htmlFor="tts-voice" className="block text-sm font-medium text-gray-700 mb-1">Voice</label>
                    <select
                        id="tts-voice"
                        value={ttsVoice}
                        onChange={(e) => setTtsVoice(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    >
                        <option value="M">Male (Default)</option>
                        <option value="F">Female</option>
                        <option value="C">Child</option>
                    </select>
                </div>

                <div className="flex justify-end items-center pt-2">
                    <button className="px-4 py-2 bg-green-500 text-white font-semibold rounded-lg hover:bg-green-600 transition-colors">
                        Save Changes
                    </button>
                </div>
            </div>
        </div>
    );
};


const App = () => {
  const [isClosed, setIsClosed] = React.useState(false);
  const [isMaximized, setIsMaximized] = React.useState(false);
  const [isMinimized, setIsMinimized] = React.useState(false);
  
  const [stream, setStream] = React.useState(null);
  const [videoAspect, setVideoAspect] = React.useState(null);
  const [grantedResolution, setGrantedResolution] = React.useState(null);
  
  const [availableCameras, setAvailableCameras] = React.useState([]);
  const [selectedCameraId, setSelectedCameraId] = React.useState('');

  const [isModelRunning, setIsModelRunning] = React.useState(false);
  
  // === NEW STATE FOR PREDICTION & UI CUSTOMIZATION ===
  const [predictedSign, setPredictedSign] = React.useState('...'); 
  const canvasRef = React.useRef(null);
  
  // UI Customization State
  const [showOnScreenText, setShowOnScreenText] = React.useState(true);
  const [textSize, setTextSize] = React.useState(32); // Increased default for visibility
  const [textPosition, setTextPosition] = React.useState(80); // 80% from top (bottom area)
  const [textToSpeechEnabled, setTextToSpeechEnabled] = React.useState(true);
  const [ttsVoice, setTtsVoice] = React.useState('M'); 
  // ======================================

  const videoRef = React.useRef(null);
  const isStreaming = !!stream;

  const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));

  const handleLoadedMetadata = (event) => {
    const video = event.target;
    const width = video.videoWidth;
    const height = video.videoHeight;
    const commonDivisor = gcd(width, height);
    
    const ratioString = `${width / commonDivisor}:${height / commonDivisor} (${width}x${height})`;
    setVideoAspect(ratioString);
    console.log(`Video metadata loaded. Raw Resolution: ${width}x${height}, Aspect Ratio: ${ratioString}`);
  };
  
  const getCameraList = React.useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter(device => device.kind === 'videoinput');
      
      setAvailableCameras(cameras);

      if (cameras.length > 0 && !selectedCameraId) {
        setSelectedCameraId(cameras[0].deviceId);
      }
    } catch (error) {
      console.error('Error enumerating devices:', error);
    }
  }, [selectedCameraId]);

  // EFFECT 1: Load cameras on mount
  React.useEffect(() => {
    // Permission check and initial camera list load
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(stream => {
            stream.getTracks().forEach(track => track.stop());
            getCameraList();
        })
        .catch(() => {
            getCameraList();
        });

    navigator.mediaDevices.addEventListener('devicechange', getCameraList);

    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', getCameraList);
    };
  }, [getCameraList]);

  // EFFECT 2: Stream Management
  React.useEffect(() => {
    let currentStream = null;

    const startStream = async (deviceId) => {
        setStream(null);
        setVideoAspect(null);
        setGrantedResolution(null);
        setIsModelRunning(false);

        if (!deviceId) return;
        
        const ASPECT_RATIO_16_9 = 1.777777778; 

        try {
            const constraints = {
                video: {
                    deviceId: { exact: deviceId },
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                    aspectRatio: { exact: ASPECT_RATIO_16_9 } 
                },
                audio: false
            };
            
            const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            currentStream = mediaStream;
            setStream(mediaStream);

            if (mediaStream.getVideoTracks().length > 0) {
              const track = mediaStream.getVideoTracks()[0];
              const settings = track.getSettings();
              
              setGrantedResolution(`${settings.width}x${settings.height} @ ${settings.frameRate} FPS`);
              
              console.log('✅ GRANTED VIDEO SETTINGS (Check this for max quality and 16:9 ratio):', settings);
            }

            console.log(`Webcam stream auto-started for device: ${deviceId}`);

        } catch (error) {
            if (error.name === "OverconstrainedError") {
                console.warn('OverconstrainedError: Failed to get 16:9 aspect ratio with preferred resolution. Trying without 16:9 constraint.');
                
                try {
                    const fallbackConstraints = {
                        video: {
                            deviceId: { exact: deviceId },
                            width: { ideal: 1920 },
                            height: { ideal: 1080 },
                            frameRate: { max: 15 } 
                        },
                        audio: false
                    };
                    
                    const mediaStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
                    currentStream = mediaStream;
                    setStream(mediaStream);
                    
                    if (mediaStream.getVideoTracks().length > 0) {
                      const track = mediaStream.getVideoTracks()[0];
                      const settings = track.getSettings();
                      setGrantedResolution(`${settings.width}x${settings.height} @ ${settings.frameRate} FPS (Fallback)`);
                      console.log('✅ GRANTED VIDEO SETTINGS (Fallback):', settings);
                    }
                    return; 
                } catch (fallbackError) {
                     console.error('Error accessing media devices for fallback stream:', fallbackError);
                }
            }
            
            console.error('Final stream start failed:', error);
            setStream(null);
            setVideoAspect("Error");
            setGrantedResolution("Failed/Unsupported (Check console)");
        }
    };

    if (selectedCameraId) {
        startStream(selectedCameraId);
    }

    return () => {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
    };
    
  }, [selectedCameraId]);

  // EFFECT 3: Attach stream to video element
  React.useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(e => {
        console.error("Video playback failed:", e);
      });
    }
  }, [stream]);

  // Final unmount cleanup
  React.useEffect(() => {
    return () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    };
  }, [stream]);
  
  // === PREDICTION LOGIC START ===
  const FLASK_API_URL = 'http://127.0.0.1:5000/api/predict';

  const captureAndPredict = React.useCallback(async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas || video.paused || video.ended || video.readyState < 2) {
          return;
      }

      // 1. Capture the frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      
      // Draw the video frame onto the canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height); 

      // Get the image data as a Base64 string (image/jpeg for speed)
      const imageDataURL = canvas.toDataURL('image/jpeg', 0.8);
      // Remove the prefix before sending
      const rawBase64 = imageDataURL.split(',')[1];


      // 2. Send to Flask API
      try {
          const response = await fetch(FLASK_API_URL, {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify({ image: rawBase64 }),
          });

          if (response.ok) {
              const data = await response.json();
              
              // === Client-side Console Output ===
              console.log(`Translation: ${data.result} (Conf: ${data.confidence.toFixed(2)})`);
              // ===================================
              
              setPredictedSign(data.result); 

              // *** NEW: Implement Text-to-Speech logic here ***
              if (textToSpeechEnabled && data.result !== '...') {
                  if ('speechSynthesis' in window) {
                      const utterance = new SpeechSynthesisUtterance(data.result);
                      // In a real app, you'd select the voice based on ttsVoice state
                      window.speechSynthesis.speak(utterance);
                  } else {
                      // Only warn once in a real implementation
                      console.warn("Text-to-Speech not supported in this browser.");
                  }
              }
              // ************************************************
          } else {
              const errorData = await response.json();
              setPredictedSign(`API Error: ${errorData.error || response.statusText}`);
              console.error("API Error:", errorData);
          }
      } catch (error) {
          console.error('Prediction API call failed (Is Flask running?):', error);
          setPredictedSign('Network Error');
      }
  }, [FLASK_API_URL, textToSpeechEnabled]); // textToSpeechEnabled added as a dependency

  // EFFECT 4: Prediction Loop (runs when model is toggled ON)
  React.useEffect(() => {
      let predictionIntervalId;

      if (isModelRunning && isStreaming) {
          // Set interval for prediction. 100ms = 10 FPS
          predictionIntervalId = setInterval(captureAndPredict, 100); 
      } else {
          setPredictedSign('...'); 
      }

      return () => {
          if (predictionIntervalId) {
              clearInterval(predictionIntervalId);
          }
      };
  }, [isModelRunning, isStreaming, captureAndPredict]); 
  // === PREDICTION LOGIC END ===

  const toggleModelStatus = () => {
      if (!isStreaming) {
          console.warn("Cannot start model: Webcam is not streaming.");
          return;
      }
      setIsModelRunning(prev => !prev);
  };

  const handleMinimize = () => {
    setIsMinimized(true);
    console.log('Window minimized. Displaying minimised bar.');
  };

  const handleMaximize = () => {
    setIsMaximized(prev => !prev);
    console.log(`Window is now ${isMaximized ? 'restored' : 'maximized'}.`);
  };

  const handleClose = () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        setStream(null);
    }
    setIsModelRunning(false);
    setIsClosed(true);
    console.log('Window closed. Stream and Model stopped.');
  };
  
  // *** NEW: Placeholder Test Function for the TEST button in the draft ***
  const handleTest = () => {
      console.log("TEST button clicked. In a real application, this might trigger a connectivity or latency test.");
      alert('Test button clicked! Check console for more details.');
  };
  // **********************************************************************


  if (isClosed) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white p-4">
        <p className="text-lg">
          Application is closed. <button onClick={() => setIsClosed(false)} className="underline text-blue-400 hover:text-blue-200 transition-colors font-medium">Click here to reopen.</button>
        </p>
      </div>
    );
  }

  if (isMinimized) {
    return (
      <div className="fixed bottom-0 left-1/2 transform -translate-x-1/2 w-80 bg-blue-600 text-white p-2 rounded-t-lg shadow-2xl flex justify-between items-center z-50 cursor-pointer">
          <span className="font-semibold" onClick={() => setIsMinimized(false)}>TalkSign Application (Minimized)</span>
          <button
              onClick={() => setIsMinimized(false)}
              className="p-1 rounded-full hover:bg-blue-700 transition-colors"
              aria-label="Restore"
          >
              <MaximizeIcon className="w-4 h-4 text-white" />
          </button>
      </div>
    );
  }

  const mainWrapperClasses = isMaximized
    ? "min-h-screen bg-white flex flex-col antialiased transition-all duration-300"
    : "min-h-screen bg-gray-50 flex flex-col antialiased transition-all duration-300";

  const mainContentLayoutClasses = isMaximized
    ? "max-w-full mx-auto w-full px-6 py-8 transition-all duration-300"
    : "max-w-7xl mx-auto w-full px-6 py-8 transition-all duration-300";

  // *** NEW: Calculate prediction text style based on state ***
  const predictedTextStyle = {
    fontSize: `${textSize}px`,
    top: `${textPosition}%`,
    transform: `translateY(-${textPosition}%)`,
  };
  // ************************************************************

  return (
    <div className={mainWrapperClasses}>
      {/* HEADER: Always fixed at top */}
      <header className="sticky top-0 z-10 bg-white shadow-md w-full">
        <div className="max-w-full mx-auto flex items-center justify-between px-6 py-3">
          
          {/* App Title */}
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-extrabold text-blue-600">
              TalkSign
            </h1>
          </div>
          
          {/* Window Controls - Now functional via props */}
          <WindowControls
            onMinimize={handleMinimize}
            onMaximize={handleMaximize}
            isMaximized={isMaximized}
            onClose={handleClose}
          />
        </div>
      </header>

      {/* MAIN CONTENT AREA: Stretches to fill the remaining vertical space. */}
      <main className={`flex-grow ${mainContentLayoutClasses}`}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* LEFT COLUMN: Main Video/Interaction Area */}
          <div className="md:col-span-2 space-y-6">
            
            {/* 1. Main Output Area */}
            <div className="p-6 bg-white rounded-xl shadow-lg border border-gray-100">
              <div className="w-full bg-gray-900 rounded-lg flex items-center justify-center text-white font-mono text-lg shadow-inner overflow-hidden relative aspect-video"> 
                
                {isStreaming ? (
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        onLoadedMetadata={handleLoadedMetadata}
                        className="w-full h-full object-cover rounded-lg transform scale-x-[-1]"
                    />
                ) : (
                    <p className="text-xl text-gray-400 text-center p-4 min-h-64 flex items-center justify-center">
                        {availableCameras.length > 0 ? (
                            'Webcam is loading or requires permission.'
                        ) : (
                            'No cameras found or permission denied. Please check your system settings.'
                        )}
                    </p>
                )}
                {/* === NEW: Hidden Canvas for Frame Capture === */}
                <canvas ref={canvasRef} className="hidden"></canvas>
                {/* =========================================== */}

                {/* Overlay text for Sign Recognition output - UPDATED POSITIONING & STYLES */}
                {isModelRunning && isStreaming && showOnScreenText && (
                    // Applying custom dynamic styles for size and position
                    <div 
                        className="absolute w-full px-6 flex items-center justify-center pointer-events-none"
                        style={predictedTextStyle}
                    >
                        <span className="p-2 bg-black bg-opacity-70 rounded-lg text-white font-extrabold text-center select-none shadow-xl">
                            {predictedSign}
                        </span>
                    </div>
                )}
                
                {/* Overlay Webcam Selector and TEST Button - Based on draft image */}
                <div className="absolute bottom-4 left-4 flex items-center space-x-2">
                    {/* Webcam Select Group (Simplified to match the visual draft layout) */}
                    <div className="relative">
                        <select
                            id="webcam-select-overlay"
                            className="bg-gray-800 bg-opacity-70 text-white text-sm py-1 pl-8 pr-3 border border-gray-700 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500"
                            value={selectedCameraId}
                            onChange={(e) => setSelectedCameraId(e.target.value)}
                        >
                            {availableCameras.length === 0 ? (
                              <option disabled value="">
                                Loading...
                              </option>
                            ) : (
                              availableCameras.map(camera => (
                                <option key={camera.deviceId} value={camera.deviceId}>
                                  {camera.label || `Camera (${camera.deviceId.substring(0, 8)}...)`}
                                </option>
                              ))
                            )}
                        </select>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 text-white absolute left-2 top-1/2 transform -translate-y-1/2 pointer-events-none">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                            <circle cx="12" cy="13" r="4"></circle>
                        </svg>
                    </div>

                    <button
                        onClick={handleTest}
                        className="px-4 py-1 bg-green-500 text-white font-semibold text-sm rounded-lg hover:bg-green-600 transition-colors"
                    >
                        TEST
                    </button>
                </div>
              </div>
            </div>
            
            {/* 2. Controls Area - Replacing with Status Log based on original code structure */}
            <div className="p-6 bg-white rounded-xl shadow-lg border border-gray-100">
                <h2 className="text-xl font-bold text-gray-800 mb-2">Model Status</h2>
                <div className="flex justify-between items-center">
                    <button
                        onClick={toggleModelStatus}
                        className={`w-48 px-6 py-2.5 font-semibold rounded-lg transition-colors duration-200 shadow-md ${
                            isModelRunning
                                ? 'bg-red-600 text-white hover:bg-red-700'
                                : 'bg-green-600 text-white hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed'
                        }`}
                        disabled={!isStreaming}
                    >
                        {isModelRunning ? 'Stop TalkSign' : 'Start TalkSign'}
                    </button>
                    <div className="text-sm space-y-1 text-gray-700">
                        <p>
                            Model Status:
                            <span className={`font-semibold ${isModelRunning ? 'text-blue-600' : 'text-orange-500'}`}>
                                {isModelRunning ? ' Running' : ' Ready'}
                            </span>
                        </p>
                        <p>
                            Granted Resolution: <span className={`font-semibold ${grantedResolution && grantedResolution.includes("Failed") ? 'text-red-500' : 'text-blue-500'}`}>{grantedResolution || 'N/A'}</span>
                        </p>
                        {videoAspect && (
                          <p>
                            Video Aspect/Res: <span className="font-semibold text-blue-500">{videoAspect}</span>
                          </p>
                        )}
                    </div>
                </div>
            </div>

          </div>

          {/* RIGHT COLUMN: Sidebar / Controls - Replacing original cards with new components */}
          <div className="md:col-span-1 space-y-6">
            <CustomizationCard
                showOnScreenText={showOnScreenText}
                setShowOnScreenText={setShowOnScreenText}
                textSize={textSize}
                setTextSize={setTextSize}
                textPosition={textPosition}
                setTextPosition={setTextPosition}
            />
            
            <TextToSpeechCard
                textToSpeechEnabled={textToSpeechEnabled}
                setTextToSpeechEnabled={setTextToSpeechEnabled}
                ttsVoice={ttsVoice}
                setTtsVoice={setTtsVoice}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;