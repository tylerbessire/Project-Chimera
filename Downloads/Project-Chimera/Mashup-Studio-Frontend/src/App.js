import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Loader2, Play, Send, Music, Square, Plus, BarChart3, Library } from 'lucide-react';
import Timeline from './Timeline';
import SearchComponent from './SearchComponent';
import './index.css';

// --- Mock shadcn/ui components updated with new color scheme ---
const Card = ({ children, className }) => <div className={`bg-gradient-to-br from-[#5c2c44] to-[#524039] border-2 border-[#bc8461] rounded-lg p-6 shadow-lg ${className}`}>{children}</div>;
const CardHeader = ({ children, className }) => <div className={`mb-4 ${className}`}>{children}</div>;
const CardTitle = ({ children, className }) => <h2 className={`text-2xl font-bold text-[#bc8461] ${className}`}>{children}</h2>;
const CardContent = ({ children, className }) => <div>{children}</div>;
const Input = (props) => <input {...props} className={`w-full bg-[#524039] border-2 border-[#bc8461] text-[#ece5dd] rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#edceaf] focus:border-[#edceaf] ${props.className}`} />;
const Button = ({ children, ...props }) => <button {...props} className={`inline-flex items-center justify-center bg-gradient-to-r from-[#bc8461] to-[#edceaf] hover:from-[#edceaf] hover:to-[#bc8461] text-[#524039] font-bold py-2 px-4 rounded-md transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed ${props.className}`}>{children}</button>;
const Textarea = (props) => <textarea {...props} className={`w-full bg-[#524039] border-2 border-[#bc8461] text-[#ece5dd] rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#edceaf] focus:border-[#edceaf] ${props.className}`} />;

// Main App Component
function App() {
  const [songInputs, setSongInputs] = useState(['', '']);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobProgress, setJobProgress] = useState('');
  const [error, setError] = useState('');
  const [mashupData, setMashupData] = useState(null);
  const [revisionCommand, setRevisionCommand] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [showTimeline, setShowTimeline] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [availableSongs, setAvailableSongs] = useState([]);
  const [userSuggestions, setUserSuggestions] = useState('');
  const audioRef = useRef(null);

  const API_BASE_URL = 'http://localhost:5002'; // Updated to match Flask port

  // Load available songs on component mount
  useEffect(() => {
    const loadSongs = async () => {
      try {
        const { data } = await axios.get(`${API_BASE_URL}/api/songs`);
        setAvailableSongs(data.songs || []);
      } catch (err) {
        console.error('Failed to load songs:', err);
      }
    };
    loadSongs();
  }, []);

  useEffect(() => {
    if (jobId && (jobStatus === 'pending' || jobStatus === 'processing')) {
      const interval = setInterval(async () => {
        try {
          const { data } = await axios.get(`${API_BASE_URL}/api/mashup/status/${jobId}`);
          setJobStatus(data.status);
          setJobProgress(data.progress || '');

          // Show real engineering data
          if (data.current_phase) {
            setJobProgress(`${data.progress} (Phase: ${data.current_phase})`);
          }

          if (data.status === 'complete') {
            setMashupData(data.result);
            setJobId(null);
            
            // Show real completion message
            if (data.result?.real_engineering_complete) {
              setJobProgress('✅ Real mashup engineering complete!');
            }
          } else if (data.status === 'failed' || data.status === 'error') {
            setError(data.error || 'An unknown error occurred.');
            setJobId(null);
          }
        } catch (err) {
          setError('Failed to get job status.');
          setJobId(null);
        }
      }, 2000); // Check every 2 seconds for real-time updates

      return () => clearInterval(interval);
    }
  }, [jobId, jobStatus]);
  
  const handleSongInputChange = (index, value) => {
    const newInputs = [...songInputs];
    newInputs[index] = value;
    setSongInputs(newInputs);
  };

  const addSongInput = () => {
    setSongInputs([...songInputs, '']);
  };

  const handleSongAdded = (songInfo) => {
    // Reload songs when a new one is added
    const loadSongs = async () => {
      try {
        const { data } = await axios.get(`${API_BASE_URL}/api/songs`);
        setAvailableSongs(data.songs || []);
      } catch (err) {
        console.error('Failed to reload songs:', err);
      }
    };
    loadSongs();
    
    // Optionally auto-add to song inputs
    if (songInfo && songInfo.title) {
      setSongInputs(prev => {
        const newInputs = [...prev];
        const emptyIndex = newInputs.findIndex(input => !input.trim());
        if (emptyIndex !== -1) {
          newInputs[emptyIndex] = songInfo.title;
        } else {
          newInputs.push(songInfo.title);
        }
        return newInputs;
      });
    }
  };

  const createMashup = async () => {
    const songs = songInputs.map(s => ({ query: s })).filter(s => s.query.trim() !== '');
    if (songs.length < 2) {
      setError('Please provide at least two songs.');
      return;
    }

    setError('');
    setMashupData(null);
    setJobStatus('pending');
    setJobProgress('Submitting job...');

    try {
      // Use professional algorithm by default, fallback to AI if API keys available
      const { data } = await axios.post(`${API_BASE_URL}/api/mashup/create`, { 
        songs,
        algorithm: 'professional',
        style: 'energetic',
        user_suggestions: userSuggestions.trim() || null
      });
      setJobId(data.job_id);
    } catch (err) {
      setError('Failed to start mashup creation job.');
      setJobStatus(null);
    }
  };

  const reviseMashup = async () => {
    if (!revisionCommand.trim() || !mashupData) {
      setError('Please enter a revision command.');
      return;
    }

    setError('');
    setJobStatus('pending');
    setJobProgress('Submitting revision...');

    try {
      const { data } = await axios.post(`${API_BASE_URL}/api/mashup/revise`, {
        mashup_id: mashupData.mashup_id,
        current_recipe: mashupData.recipe,
        user_command: revisionCommand,
      });
      setMashupData(null);
      setRevisionCommand('');
      setJobId(data.job_id);
    } catch (err) {
      setError('Failed to start mashup revision job.');
      setJobStatus(null);
    }
  };
  
  const togglePlay = () => {
    if (audioRef.current) {
        if (isPlaying) {
            audioRef.current.pause();
        } else {
            audioRef.current.play();
        }
        setIsPlaying(!isPlaying);
    }
  };

  const isLoading = jobStatus === 'pending' || jobStatus === 'processing';

  // If timeline is open, show timeline component
  if (showTimeline) {
    return <Timeline onClose={() => setShowTimeline(false)} />;
  }

  return (
    <div className="bg-gradient-to-br from-[#524039] via-[#5c2c44] to-[#524039] min-h-screen text-[#ece5dd] p-4 sm:p-8 flex flex-col items-center" style={{ fontFamily: 'Inter, sans-serif' }}>
      <div className="w-full max-w-3xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-[#bc8461] to-[#edceaf]">
            Intelligent Mashup Studio
          </h1>
          <p className="text-[#edceaf] mt-2 font-medium">AI-Powered Music Mashups with Luna & Claude</p>
        </header>

        <main>
          {/* Search Component Toggle */}
          <div className="mb-6 text-center">
            <Button 
              onClick={() => setShowSearch(!showSearch)} 
              className="bg-gradient-to-r from-[#5c2c44] to-[#524039] hover:from-[#524039] hover:to-[#5c2c44] text-[#ece5dd] border-2 border-[#bc8461]"
            >
              <Library className="mr-2 h-4 w-4" />
              {showSearch ? 'Hide Search' : 'Search & Download Songs'}
            </Button>
          </div>

          {/* Search Component */}
          {showSearch && (
            <SearchComponent onSongAdded={handleSongAdded} />
          )}

          <Card className="mb-8">
            <CardHeader>
              <CardTitle>1. Create Your Mashup</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {songInputs.map((song, index) => (
                  <Input
                    key={index}
                    type="text"
                    placeholder={`Song ${index + 1} (e.g., "Artist - Title")`}
                    value={song}
                    onChange={(e) => handleSongInputChange(index, e.target.value)}
                    disabled={isLoading}
                  />
                ))}
                
                {/* User Suggestions Field */}
                <div className="mt-4">
                  <label className="block text-sm font-medium text-[#edceaf] mb-2">
                    Your Creative Direction (Optional)
                  </label>
                  <Textarea
                    placeholder="Tell Luna & Claude how you want this mashup to sound... (e.g., 'Start slow and build to an epic drop', 'Keep it chill throughout', 'Make it a dance floor banger')"
                    value={userSuggestions}
                    onChange={(e) => setUserSuggestions(e.target.value)}
                    disabled={isLoading}
                    rows={3}
                    className="text-sm"
                  />
                </div>
              </div>
              <div className="flex justify-between items-center mt-4">
                 <Button onClick={addSongInput} disabled={isLoading} className="bg-gradient-to-r from-[#5c2c44] to-[#524039] hover:from-[#524039] hover:to-[#5c2c44] text-[#ece5dd] border-2 border-[#bc8461]">
                    <Plus className="mr-2 h-4 w-4" /> Add Song
                 </Button>
                <div className="flex gap-3">
                  <Button onClick={() => setShowTimeline(true)} className="bg-gradient-to-r from-[#edceaf] to-[#bc8461] hover:from-[#bc8461] hover:to-[#edceaf]">
                    <BarChart3 className="mr-2 h-4 w-4" />
                    Live Timeline
                  </Button>
                  <Button onClick={createMashup} disabled={isLoading}>
                    {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Music className="mr-2 h-4 w-4" />}
                    Create Mashup
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {(isLoading || error || mashupData) && (
            <Card>
              <CardHeader>
                <CardTitle>2. Results & Revisions</CardTitle>
              </CardHeader>
              <CardContent>
                {isLoading && (
                  <div className="text-center p-8">
                    <Loader2 className="h-12 w-12 animate-spin text-purple-400 mx-auto" />
                    <p className="mt-4 text-lg font-semibold">{jobStatus ? jobStatus.charAt(0).toUpperCase() + jobStatus.slice(1) : 'Loading'}...</p>
                    <p className="text-gray-400">{jobProgress}</p>
                  </div>
                )}
                {error && <p className="text-red-500 font-bold bg-red-900/50 p-3 rounded-md">{error}</p>}
                {mashupData && (
                  <div className="space-y-6">
                    <div>
                      <h3 className="font-bold text-lg mb-2">Mashup Ready: {mashupData.recipe.mashup_title}</h3>
                      <div className="flex items-center gap-4 bg-gray-900 p-3 rounded-lg">
                        <Button onClick={togglePlay} className="rounded-full w-12 h-12 flex-shrink-0">
                          {isPlaying ? <Square /> : <Play />}
                        </Button>
                        <div className="w-full">
                           <audio ref={audioRef} src={`${API_BASE_URL}${mashupData.audio_url}`} onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)} onEnded={() => setIsPlaying(false)} controls className="w-full"></audio>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="font-bold text-lg mb-2">Enter Revision Request:</h3>
                       <Textarea
                        placeholder="e.g., Make the first chorus instrumental only..."
                        value={revisionCommand}
                        onChange={(e) => setRevisionCommand(e.target.value)}
                        disabled={isLoading}
                        rows={3}
                      />
                      <Button onClick={reviseMashup} disabled={isLoading} className="mt-2 w-full">
                        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
                        Submit Tweak
                      </Button>
                    </div>
                     <div>
                        <details>
                            <summary className="font-bold text-lg cursor-pointer hover:text-purple-400 transition-colors">View Current Recipe (JSON)</summary>
                            <pre className="bg-black p-4 rounded-md mt-2 text-xs overflow-x-auto">
                                {JSON.stringify(mashupData.recipe, null, 2)}
                            </pre>
                        </details>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
