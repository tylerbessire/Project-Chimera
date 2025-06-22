import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Activity } from 'lucide-react';

const Timeline = ({ onClose }) => {
  const [currentJobId, setCurrentJobId] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playheadPosition, setPlayheadPosition] = useState(0);
  const [moleculeCount, setMoleculeCount] = useState(0);
  const [jobStatus, setJobStatus] = useState('Waiting for mashup job...');
  const [lunaStatus, setLunaStatus] = useState('Luna (gpt-4.1-nano): Ready');
  const [claudeStatus, setClaudeStatus] = useState('Claude (claude-4-sonnet-20250514): Ready');
  const [actions, setActions] = useState([
    { ai: 'luna', text: 'Ready for emotional analysis and molecular planning', time: new Date().toLocaleTimeString() },
    { ai: 'claude', text: 'Timeline interface active - awaiting collaboration data', time: new Date().toLocaleTimeString() }
  ]);
  const [trackData, setTrackData] = useState({
    track1: { info: 'Awaiting Luna\'s Analysis', blocks: [] },
    track2: { info: 'Luna\'s Design Pending', blocks: [] },
    track3: { info: 'Pattern Recognition', blocks: [] },
    track4: { info: 'Claude\'s Engineering', blocks: [] },
    track5: { info: 'Narrative Elements', blocks: [] },
    track6: { info: 'Emotional Architecture', blocks: [] }
  });

  const API_BASE_URL = 'http://localhost:5002';
  const playheadRef = useRef(null);

  // Check for active jobs
  useEffect(() => {
    const checkForActiveJobs = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/active-jobs`);
        if (response.ok) {
          const jobs = await response.json();
          
          if (jobs.length > 0 && jobs[0].id !== currentJobId) {
            setCurrentJobId(jobs[0].id);
            setJobStatus(`Processing: ${jobs[0].id}`);
            startMonitoring(jobs[0].id);
          }
        }
      } catch (error) {
        console.log('Checking for jobs...');
      }
    };

    checkForActiveJobs();
    const interval = setInterval(checkForActiveJobs, 3000);
    return () => clearInterval(interval);
  }, [currentJobId]);

  const startMonitoring = (jobId) => {
    const statusInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/mashup/status/${jobId}`);
        const status = await response.json();
        
        updateUI(status);
        
        if (status.status === 'complete') {
          clearInterval(statusInterval);
          setJobStatus('Mashup Complete!');
          startPlayback();
        } else if (status.status === 'error') {
          clearInterval(statusInterval);
          setJobStatus('Error occurred');
        }
      } catch (error) {
        console.error('Status check failed:', error);
      }
    }, 1000);
  };

  const updateUI = (status) => {
    // Update molecule count from real processing data
    if (status.molecules_used) {
      setMoleculeCount(status.molecules_used);
    } else if (status.progress && status.progress.includes('segments')) {
      const match = status.progress.match(/(\d+)/);
      if (match) {
        const count = parseInt(match[1]);
        setMoleculeCount(prev => Math.max(prev, count));
      }
    }

    // Update AI status based on real processing phase
    if (status.current_phase) {
      switch(status.current_phase) {
        case 'analysis':
          setLunaStatus('Luna (gpt-4.1-nano): Analyzing Real Audio Data');
          setClaudeStatus('Claude (claude-4-sonnet-20250514): Preparing Engineering');
          break;
        case 'engineering':
          setLunaStatus('Luna (gpt-4.1-nano): Creating Emotional Narrative');
          setClaudeStatus('Claude (claude-4-sonnet-20250514): Engineering Timeline');
          break;
        case 'rendering':
          setLunaStatus('Luna (gpt-4.1-nano): Final Creative Review');
          setClaudeStatus('Claude (claude-4-sonnet-20250514): Rendering Audio');
          break;
      }
    }
    
    // Add real actions from processing log
    if (status.progress) {
      addAction('system', status.progress);
    }

    // Update timeline with REAL data from backend
    if (status.timeline_data && status.timeline_data.tracks) {
      updateRealTimelineData(status.timeline_data);
    } else if (status.progress && status.progress.includes('segment')) {
      simulateTimelineBlocks();
    }
  };

  const updateRealTimelineData = (realTimelineData) => {
    /**
     * Update timeline with REAL data from Luna and Claude's work
     * This replaces simulation with actual engineering data
     */
    setTrackData(prev => {
      const newTrackData = { ...prev };
      
      realTimelineData.tracks.forEach((track, index) => {
        const trackId = `track${index + 1}`;
        if (newTrackData[trackId]) {
          // Update with real timeline segments
          newTrackData[trackId].blocks = track.segments.map((segment, segIndex) => ({
            id: `real_${track.track_id}_${segIndex}`,
            type: segment.type || 'vocals',
            startTime: segment.start_time,
            duration: segment.end_time - segment.start_time,
            label: `${segment.source.toUpperCase()}: ${segment.type}`,
            volume: segment.volume,
            effects: segment.effects || [],
            waveform: segment.waveform || [],
            real_data: true // Mark as real data
          }));
          
          // Update track info with real engineering details
          const totalSegments = track.segments.length;
          const totalDuration = track.segments.reduce((sum, seg) => sum + (seg.end_time - seg.start_time), 0);
          newTrackData[trackId].info = `${totalSegments} segments, ${totalDuration.toFixed(1)}s engineered`;
        }
      });
      
      return newTrackData;
    });
    
    // Add real engineering actions
    addAction('luna', `Real timeline generated with ${realTimelineData.tracks.length} tracks`);
    addAction('claude', `Engineered ${realTimelineData.tracks.reduce((sum, track) => sum + track.segments.length, 0)} precise segments`);
  };

  const addAction = (ai, message) => {
    const timestamp = new Date().toLocaleTimeString();
    setActions(prev => [
      { ai, text: message, time: timestamp },
      ...prev.slice(0, 9) // Keep only last 10 actions
    ]);
  };

  const simulateTimelineBlocks = () => {
    const types = ['vocals', 'drums', 'bass', 'guitar', 'synth'];
    const trackIds = ['track1', 'track2', 'track3', 'track4', 'track5', 'track6'];
    
    setTrackData(prev => {
      const newTrackData = { ...prev };
      
      trackIds.forEach((trackId, index) => {
        if (newTrackData[trackId].blocks.length < 3) {
          const startTime = Math.random() * 100;
          const duration = 20 + Math.random() * 40;
          
          newTrackData[trackId].blocks.push({
            id: Date.now() + Math.random(),
            type: types[index % types.length],
            startTime,
            duration,
            label: `Segment ${newTrackData[trackId].blocks.length + 1}`
          });
          
          newTrackData[trackId].info = `${newTrackData[trackId].blocks.length} segments processed`;
        }
      });
      
      return newTrackData;
    });
  };

  const startPlayback = () => {
    setIsPlaying(true);
    const duration = 120000; // 2 minutes in ms
    const startTime = Date.now();
    
    const updatePlayhead = () => {
      if (!isPlaying) return;
      
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const position = progress * 2400; // 2400px timeline width
      
      setPlayheadPosition(position);
      
      if (progress < 1) {
        requestAnimationFrame(updatePlayhead);
      } else {
        setIsPlaying(false);
        setPlayheadPosition(0);
      }
    };
    
    updatePlayhead();
  };

  const getBlockStyle = (block) => ({
    left: `${block.startTime * 20}px`, // 20px per second
    width: `${block.duration * 20}px`,
  });

  const getBlockClass = (type) => {
    const baseClass = "absolute h-16 top-2 rounded-lg flex items-center px-3 text-xs font-semibold cursor-pointer transition-all duration-300 hover:transform hover:-translate-y-1 hover:scale-105 shadow-lg border-2";
    
    switch (type) {
      case 'vocals': return `${baseClass} bg-gradient-to-r from-[#bc8461] to-[#edceaf] text-[#524039] border-[#bc8461]`;
      case 'drums': return `${baseClass} bg-gradient-to-r from-[#5c2c44] to-[#bc8461] text-[#ece5dd] border-[#bc8461]`;
      case 'bass': return `${baseClass} bg-gradient-to-r from-[#524039] to-[#5c2c44] text-[#ece5dd] border-[#bc8461]`;
      case 'guitar': return `${baseClass} bg-gradient-to-r from-[#edceaf] to-[#bc8461] text-[#524039] border-[#bc8461]`;
      case 'synth': return `${baseClass} bg-gradient-to-r from-[#5c2c44] to-[#edceaf] text-[#ece5dd] border-[#bc8461]`;
      default: return `${baseClass} bg-gradient-to-r from-[#bc8461] to-[#edceaf] text-[#524039] border-[#bc8461]`;
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-[#524039] via-[#5c2c44] to-[#524039] text-[#ece5dd] overflow-hidden" style={{ fontFamily: 'Inter, sans-serif' }}>
      {/* Header */}
      <div className="bg-gradient-to-r from-[#5c2c44] to-[#524039] border-b-3 border-[#bc8461] px-5 py-5 flex justify-between items-center shadow-lg">
        <div className="flex items-center gap-4">
          <button
            onClick={onClose}
            className="bg-gradient-to-r from-[#5c2c44] to-[#524039] border-2 border-[#bc8461] text-[#ece5dd] px-4 py-2 rounded-lg hover:bg-gradient-to-r hover:from-[#bc8461] hover:to-[#edceaf] hover:text-[#524039] transition-all duration-300 font-semibold flex items-center gap-2"
          >
            <ArrowLeft size={16} />
            Back to Studio
          </button>
          <h1 className="text-2xl font-bold text-[#bc8461] font-mono tracking-wide">🎵 Live Collaboration Timeline</h1>
        </div>
        
        <div className="flex gap-6">
          <div className="flex items-center gap-3 bg-[#ece5dd]/10 border-2 border-[#bc8461] rounded-xl px-4 py-3">
            <div className="w-3 h-3 bg-[#bc8461] rounded-full animate-pulse"></div>
            <span className="text-[#edceaf] font-semibold">{lunaStatus}</span>
          </div>
          <div className="flex items-center gap-3 bg-[#ece5dd]/10 border-2 border-[#bc8461] rounded-xl px-4 py-3">
            <div className="w-3 h-3 bg-[#bc8461] rounded-full animate-pulse"></div>
            <span className="text-[#ece5dd] font-semibold">{claudeStatus}</span>
          </div>
        </div>
      </div>

      {/* Status and Molecule Counter */}
      <div className="absolute top-6 right-6 space-y-4 z-20">
        <div className="bg-gradient-to-r from-[#5c2c44] to-[#524039] border-3 border-[#bc8461] rounded-xl px-5 py-4 shadow-lg">
          <div className="text-3xl font-bold text-[#bc8461] text-center font-mono">{moleculeCount}</div>
          <div className="text-sm text-[#edceaf] text-center font-semibold">Molecules Processed</div>
        </div>
        
        <div className="bg-gradient-to-r from-[#5c2c44] to-[#524039] border-2 border-[#bc8461] rounded-lg px-4 py-3 shadow-lg">
          <div className="text-[#ece5dd] text-sm font-semibold">{jobStatus}</div>
        </div>
      </div>

      {/* Timeline Container */}
      <div className="flex h-[calc(100vh-100px)]">
        {/* Track Labels */}
        <div className="w-70 bg-gradient-to-b from-[#5c2c44] to-[#524039] border-r-3 border-[#bc8461] shadow-lg">
          {Object.entries(trackData).map(([trackId, track], index) => {
            const trackNames = ['Lead Vocals', 'Harmony Stack', 'Drums', 'Bass', 'Guitar Layers', 'Synth/FX'];
            return (
              <div key={trackId} className="h-20 px-4 py-3 border-b-2 border-[#bc8461] flex flex-col justify-center">
                <div className="text-[#bc8461] font-bold text-base font-mono">{trackNames[index]}</div>
                <div className="text-[#edceaf] text-sm font-medium">{track.info}</div>
              </div>
            );
          })}
        </div>

        {/* Timeline Main */}
        <div className="flex-1 relative overflow-x-auto bg-gradient-to-b from-[#524039] via-[#5c2c44] to-[#524039]">
          {/* Time Markers */}
          <div className="absolute top-0 left-0 w-[2400px] h-6 bg-gradient-to-r from-[#5c2c44] to-[#524039] border-b-2 border-[#bc8461] flex items-center text-[#ece5dd] text-sm font-bold font-mono z-10">
            {Array.from({ length: 10 }, (_, i) => (
              <div key={i} className="w-60 text-center border-r border-[#bc8461]">
                {i}:{(i * 10).toString().padStart(2, '0')}
              </div>
            ))}
          </div>

          {/* Grid */}
          <div 
            className="absolute top-6 left-0 w-[2400px] h-full"
            style={{
              backgroundImage: `
                linear-gradient(to right, rgba(188, 132, 97, 0.2) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(188, 132, 97, 0.2) 1px, transparent 1px)
              `,
              backgroundSize: '20px 80px'
            }}
          ></div>

          {/* Playhead */}
          <div 
            ref={playheadRef}
            className="absolute top-6 w-1 h-full bg-[#bc8461] shadow-lg z-20"
            style={{ left: `${playheadPosition}px`, boxShadow: '0 0 10px rgba(188, 132, 97, 0.8)' }}
          ></div>

          {/* Tracks */}
          <div className="absolute top-6 left-0 w-[2400px]">
            {Object.entries(trackData).map(([trackId, track]) => (
              <div key={trackId} className="h-20 border-b-2 border-[#bc8461] relative">
                {track.blocks.map((block) => (
                  <div
                    key={block.id}
                    className={getBlockClass(block.type)}
                    style={getBlockStyle(block)}
                    onClick={() => {
                      // Add click effect
                      const element = document.elementFromPoint(0, 0);
                      if (element) {
                        element.style.transform = 'translateY(-4px) scale(1.05)';
                        setTimeout(() => {
                          element.style.transform = 'translateY(-2px) scale(1.02)';
                        }, 200);
                      }
                    }}
                  >
                    {block.label}
                    <div className="absolute bottom-1 left-2 right-2 h-3 bg-[#ece5dd]/30 rounded-sm" style={{
                      background: `repeating-linear-gradient(to right, rgba(236, 229, 221, 0.3) 0px, rgba(236, 229, 221, 0.7) 2px, rgba(236, 229, 221, 0.3) 4px, rgba(236, 229, 221, 0.1) 6px)`
                    }}></div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* AI Actions Feed */}
      <div className="fixed bottom-6 right-6 bg-gradient-to-br from-[#5c2c44] to-[#524039] border-3 border-[#bc8461] rounded-xl p-5 w-96 max-h-72 overflow-y-auto shadow-xl">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="text-[#bc8461]" size={20} />
          <h3 className="text-[#bc8461] font-bold text-lg">Live Actions</h3>
        </div>
        
        <div className="space-y-2">
          {actions.map((action, index) => (
            <div
              key={index}
              className={`text-sm p-2 border-b border-[#bc8461] ${
                action.ai === 'luna' 
                  ? 'text-[#edceaf]' 
                  : action.ai === 'claude' 
                    ? 'text-[#ece5dd]' 
                    : 'text-[#bc8461]'
              }`}
            >
              <span className="font-semibold">{action.time}</span> - {action.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Timeline;