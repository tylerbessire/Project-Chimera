import React, { useState } from 'react';
import axios from 'axios';
import { Search, Download, Music, Loader2 } from 'lucide-react';

// Mock shadcn/ui components with matching style
const Card = ({ children, className }) => <div className={`bg-gradient-to-br from-[#5c2c44] to-[#524039] border-2 border-[#bc8461] rounded-lg p-6 shadow-lg ${className}`}>{children}</div>;
const CardHeader = ({ children, className }) => <div className={`mb-4 ${className}`}>{children}</div>;
const CardTitle = ({ children, className }) => <h2 className={`text-2xl font-bold text-[#bc8461] ${className}`}>{children}</h2>;
const CardContent = ({ children, className }) => <div>{children}</div>;
const Input = (props) => <input {...props} className={`w-full bg-[#524039] border-2 border-[#bc8461] text-[#ece5dd] rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#edceaf] focus:border-[#edceaf] ${props.className}`} />;
const Button = ({ children, ...props }) => <button {...props} className={`inline-flex items-center justify-center bg-gradient-to-r from-[#bc8461] to-[#edceaf] hover:from-[#edceaf] hover:to-[#bc8461] text-[#524039] font-bold py-2 px-4 rounded-md transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed ${props.className}`}>{children}</button>;

const SearchComponent = ({ onSongAdded }) => {
  const [artist, setArtist] = useState('');
  const [title, setTitle] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [downloadingId, setDownloadingId] = useState(null);
  const [error, setError] = useState('');

  const API_BASE_URL = 'http://localhost:5002';

  const handleSearch = async () => {
    if (!artist.trim() && !title.trim()) {
      setError('Please enter an artist or title to search');
      return;
    }

    const query = `${artist.trim()} ${title.trim()}`.trim();
    setError('');
    setIsSearching(true);
    setSearchResults([]);

    try {
      const { data } = await axios.post(`${API_BASE_URL}/api/search`, {
        query: query,
        max_results: 5
      });

      // Poll for search results since it's a background job
      const checkSearchResults = async (jobId) => {
        const checkInterval = setInterval(async () => {
          try {
            const statusResponse = await axios.get(`${API_BASE_URL}/api/job/${jobId}`);
            const jobData = statusResponse.data;

            if (jobData.status === 'completed' && jobData.result) {
              setSearchResults(jobData.result.results || []);
              setIsSearching(false);
              clearInterval(checkInterval);
            } else if (jobData.status === 'failed') {
              setError(jobData.error || 'Search failed');
              setIsSearching(false);
              clearInterval(checkInterval);
            }
          } catch (err) {
            setError('Failed to get search results');
            setIsSearching(false);
            clearInterval(checkInterval);
          }
        }, 1000);
      };

      if (data.job_id) {
        checkSearchResults(data.job_id);
      }
    } catch (err) {
      setError('Search request failed');
      setIsSearching(false);
    }
  };

  const handleDownload = async (result) => {
    setDownloadingId(result.id);
    setError('');

    try {
      const { data } = await axios.post(`${API_BASE_URL}/api/download_and_analyze`, {
        video_id: result.id,
        custom_name: `${artist} - ${title}`.trim() || result.title
      });

      // Poll for download and analysis completion
      const checkDownload = async (jobId) => {
        const checkInterval = setInterval(async () => {
          try {
            const statusResponse = await axios.get(`${API_BASE_URL}/api/job/${jobId}`);
            const jobData = statusResponse.data;

            if (jobData.status === 'completed' && jobData.result) {
              // Song has been added to library
              if (onSongAdded) {
                onSongAdded(jobData.result.song_info);
              }
              setDownloadingId(null);
              clearInterval(checkInterval);
              
              // Clear search after successful download
              setSearchResults([]);
              setArtist('');
              setTitle('');
            } else if (jobData.status === 'failed') {
              setError(jobData.error || 'Download failed');
              setDownloadingId(null);
              clearInterval(checkInterval);
            }
          } catch (err) {
            setError('Failed to download song');
            setDownloadingId(null);
            clearInterval(checkInterval);
          }
        }, 1000);
      };

      if (data.job_id) {
        checkDownload(data.job_id);
      }
    } catch (err) {
      setError('Download request failed');
      setDownloadingId(null);
    }
  };

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>🎵 Search & Download Songs</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Search Form */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Input
              type="text"
              placeholder="Artist (e.g., Avicii)"
              value={artist}
              onChange={(e) => setArtist(e.target.value)}
              disabled={isSearching}
            />
            <Input
              type="text"
              placeholder="Title (e.g., Levels)"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={isSearching}
            />
            <Button 
              onClick={handleSearch} 
              disabled={isSearching}
              className="w-full"
            >
              {isSearching ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Search className="mr-2 h-4 w-4" />
              )}
              {isSearching ? 'Searching...' : 'Search'}
            </Button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-900/50 border border-red-500 text-red-200 p-3 rounded-md">
              {error}
            </div>
          )}

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-[#bc8461]">Search Results:</h3>
              {searchResults.map((result) => (
                <div 
                  key={result.id} 
                  className="bg-[#524039] border border-[#bc8461] rounded-lg p-4 flex items-center justify-between"
                >
                  <div className="flex-1">
                    <h4 className="font-semibold text-[#ece5dd]">{result.title}</h4>
                    <p className="text-[#edceaf] text-sm">
                      by {result.uploader} • {result.duration_string} • 
                      {result.view_count?.toLocaleString()} views
                    </p>
                    <div className="text-xs text-[#bc8461] mt-1">
                      Relevance: {Math.round(result.relevance_score)}%
                    </div>
                  </div>
                  <div className="ml-4">
                    <Button
                      onClick={() => handleDownload(result)}
                      disabled={downloadingId === result.id}
                      className="bg-gradient-to-r from-[#5c2c44] to-[#524039] hover:from-[#524039] hover:to-[#5c2c44] text-[#ece5dd] border-2 border-[#bc8461]"
                    >
                      {downloadingId === result.id ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Download className="mr-2 h-4 w-4" />
                      )}
                      {downloadingId === result.id ? 'Downloading...' : 'Download'}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Searching State */}
          {isSearching && (
            <div className="text-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-[#bc8461] mx-auto mb-2" />
              <p className="text-[#edceaf]">Searching YouTube for "{artist} {title}"...</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default SearchComponent;