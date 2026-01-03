import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Trash2,
  Play,
  Pause,
  Volume2,
  Wand2,
  Plus,
  Search,
  Filter,
  Download,
  Edit2,
  Check,
  X,
  Combine,
  Mic2,
  Music,
  Clock,
  FileAudio,
  Loader2,
  AlertCircle,
  Copy,
  MoreVertical,
  RefreshCw,
  FolderOpen,
  Sparkles,
  ChevronUp,
  ChevronDown,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import { cn, generateId, formatFileSize, formatDuration, downloadBlob } from '../../lib/utils';
import Layout from '../layout/Layout';
import Button from '../ui/Button';

interface VoiceFile {
  id: string;
  name: string;
  path: string;
  size: number;
  duration?: number;
  uploadedAt: number;
}

interface Voice {
  name: string;
  files: VoiceFile[];
  totalSize: number;
  createdAt: number;
}

interface AudioPlayerState {
  voiceName: string;
  fileId: string;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
}

const VoiceLibrary: React.FC = () => {
  const [voices, setVoices] = useState<Record<string, Voice>>({});
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedVoices, setSelectedVoices] = useState<Set<string>>(new Set());
  const [uploadingVoice, setUploadingVoice] = useState<string | null>(null);
  const [editingVoice, setEditingVoice] = useState<string | null>(null);
  const [newVoiceName, setNewVoiceName] = useState('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [audioPlayer, setAudioPlayer] = useState<AudioPlayerState | null>(null);
  const [testingVoice, setTestingVoice] = useState<string | null>(null);
  const [testText, setTestText] = useState('Hello, this is a test of the voice synthesis system.');
  const [combineMode, setCombineMode] = useState(false);
  const [selectedForCombine, setSelectedForCombine] = useState<Set<string>>(new Set());
  const [expandedVoice, setExpandedVoice] = useState<string | null>(null);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load voices from API
  const loadVoices = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch('/voices');
      const data = await response.json();
      const voiceList = Array.isArray(data) ? data : data.voices || [];
      
      // Fetch detailed info for each voice
      const voicesData: Record<string, Voice> = {};
      await Promise.all(
        voiceList.map(async (voiceName: string) => {
          try {
            const detailsResponse = await fetch(`/voices/${encodeURIComponent(voiceName)}/details`);
            if (detailsResponse.ok) {
              const details = await detailsResponse.json();
              voicesData[voiceName] = {
                name: voiceName,
                files: details.files.map((file: any) => ({
                  id: generateId(),
                  name: file.name,
                  path: file.path,
                  size: file.size,
                  duration: file.duration,
                  uploadedAt: file.modified * 1000,
                })),
                totalSize: details.total_size,
                createdAt: Date.now(),
              };
            } else {
              // Fallback if details endpoint fails
              voicesData[voiceName] = {
                name: voiceName,
                files: [],
                totalSize: 0,
                createdAt: Date.now(),
              };
            }
          } catch (error) {
            console.error(`Failed to load details for voice ${voiceName}:`, error);
            voicesData[voiceName] = {
              name: voiceName,
              files: [],
              totalSize: 0,
              createdAt: Date.now(),
            };
          }
        })
      );
      
      setVoices(voicesData);
    } catch (error) {
      toast.error('Failed to load voices');
      console.error(error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadVoices();
  }, [loadVoices]);

  // Audio playback
  useEffect(() => {
    if (!audioRef.current) {
      audioRef.current = new Audio();
      
      audioRef.current.addEventListener('timeupdate', () => {
        if (audioPlayer && audioRef.current) {
          setAudioPlayer(prev => prev ? {
            ...prev,
            currentTime: audioRef.current!.currentTime,
          } : null);
        }
      });
      
      audioRef.current.addEventListener('ended', () => {
        setAudioPlayer(prev => prev ? { ...prev, isPlaying: false } : null);
      });
      
      audioRef.current.addEventListener('loadedmetadata', () => {
        if (audioPlayer && audioRef.current) {
          setAudioPlayer(prev => prev ? {
            ...prev,
            duration: audioRef.current!.duration,
          } : null);
        }
      });
    }
  }, [audioPlayer]);

  // Upload voice files
  const handleUpload = async (voiceName: string, files: FileList) => {
    if (!voiceName.trim()) {
      toast.error('Please enter a voice name');
      return;
    }

    setUploadingVoice(voiceName);
    
    try {
      const uploadPromises = Array.from(files).map(async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`/voices?voice_name=${encodeURIComponent(voiceName)}`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Upload failed');
        }
        
        return response.json();
      });
      
      await Promise.all(uploadPromises);
      
      toast.success(`Uploaded ${files.length} file(s) to ${voiceName}`);
      await loadVoices();
      setShowUploadModal(false);
      setNewVoiceName('');
    } catch (error) {
      toast.error(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setUploadingVoice(null);
    }
  };

  // Delete voice
  const handleDeleteVoice = async (voiceName: string) => {
    if (!confirm(`Are you sure you want to delete the voice "${voiceName}"? This action cannot be undone.`)) {
      return;
    }
    
    try {
      const response = await fetch(`/voices/${encodeURIComponent(voiceName)}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error('Delete failed');
      }
      
      toast.success(`Voice "${voiceName}" deleted`);
      await loadVoices();
    } catch (error) {
      toast.error(`Failed to delete voice: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  // Test TTS with voice
  const handleTestVoice = async (voiceName: string) => {
    if (!testText.trim()) {
      toast.error('Please enter test text');
      return;
    }
    
    setTestingVoice(voiceName);
    
    try {
      const response = await fetch('/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: testText,
          voice_name: voiceName,
          language: 'en',
          output_format: 'wav',
        }),
      });
      
      if (!response.ok) {
        throw new Error('TTS generation failed');
      }
      
      const buffer = await response.arrayBuffer();
      const blob = new Blob([buffer], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
        setAudioPlayer({
          voiceName,
          fileId: 'test',
          isPlaying: true,
          currentTime: 0,
          duration: 0,
        });
      }
      
      toast.success('TTS generated successfully');
    } catch (error) {
      toast.error(`TTS failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setTestingVoice(null);
    }
  };

  // Play/pause audio
  const togglePlayPause = () => {
    if (!audioRef.current) return;
    
    if (audioPlayer?.isPlaying) {
      audioRef.current.pause();
      setAudioPlayer(prev => prev ? { ...prev, isPlaying: false } : null);
    } else {
      audioRef.current.play();
      setAudioPlayer(prev => prev ? { ...prev, isPlaying: true } : null);
    }
  };

  // Play specific voice file
  const handlePlayFile = async (voiceName: string, fileName: string) => {
    try {
      const response = await fetch(`/voices/${encodeURIComponent(voiceName)}/files/${encodeURIComponent(fileName)}`);
      if (!response.ok) {
        throw new Error('Failed to load audio file');
      }
      
      const buffer = await response.arrayBuffer();
      const blob = new Blob([buffer], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
        setAudioPlayer({
          voiceName,
          fileId: fileName,
          isPlaying: true,
          currentTime: 0,
          duration: 0,
        });
      }
    } catch (error) {
      toast.error('Failed to play audio file');
    }
  };

  // Download voice file
  const handleDownloadFile = async (voiceName: string, fileName: string) => {
    try {
      const response = await fetch(`/voices/${encodeURIComponent(voiceName)}/files/${encodeURIComponent(fileName)}`);
      if (!response.ok) {
        throw new Error('Download failed');
      }
      
      const buffer = await response.arrayBuffer();
      const blob = new Blob([buffer]);
      downloadBlob(blob, fileName);
      toast.success('File downloaded');
    } catch (error) {
      toast.error('Failed to download file');
    }
  };

  // Delete individual file
  const handleDeleteFile = async (voiceName: string, fileName: string) => {
    if (!confirm(`Delete ${fileName}?`)) return;
    
    try {
      const response = await fetch(
        `/voices/${encodeURIComponent(voiceName)}/files/${encodeURIComponent(fileName)}`,
        { method: 'DELETE' }
      );
      
      if (!response.ok) {
        throw new Error('Delete failed');
      }
      
      toast.success('File deleted');
      await loadVoices();
    } catch (error) {
      toast.error('Failed to delete file');
    }
  };

  // Filter voices
  const filteredVoices = Object.entries(voices).filter(([name]) =>
    name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Combine voices
  const handleCombineVoices = async () => {
    if (selectedForCombine.size < 2) {
      toast.error('Select at least 2 voices to combine');
      return;
    }
    
    const combinedName = prompt('Enter name for combined voice:', 'combined_voice');
    if (!combinedName) return;
    
    const sanitizedName = combinedName.replace(/[^a-zA-Z0-9_]/g, '_');
    
    try {
      const response = await fetch('/voices/combine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          voice_name: sanitizedName,
          source_voices: Array.from(selectedForCombine),
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to combine voices');
      }
      
      const result = await response.json();
      toast.success(`Combined ${result.files_copied} files into "${sanitizedName}"`);
      
      setCombineMode(false);
      setSelectedForCombine(new Set());
      await loadVoices();
    } catch (error) {
      toast.error(`Failed to combine voices: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  return (
    <Layout>
      <div className="h-full flex flex-col relative overflow-hidden aurora-bg noise">
        {/* Background orbs */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <motion.div
            animate={{
              x: [0, 100, 0],
              y: [0, -50, 0],
              scale: [1, 1.2, 1],
            }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="absolute -top-40 -right-40 w-96 h-96 rounded-full"
            style={{
              background: 'radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%)',
              filter: 'blur(40px)',
            }}
          />
          <motion.div
            animate={{
              x: [0, -80, 0],
              y: [0, 80, 0],
              scale: [1, 1.3, 1],
            }}
            transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
            className="absolute -bottom-20 -left-20 w-80 h-80 rounded-full"
            style={{
              background: 'radial-gradient(circle, rgba(168, 85, 247, 0.12) 0%, transparent 70%)',
              filter: 'blur(40px)',
            }}
          />
        </div>

        <div className="relative z-10 h-full flex flex-col p-8 max-w-7xl mx-auto w-full">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="flex items-center justify-between mb-6">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-xl bg-gradient-to-br from-accent to-purple-500">
                    <Music className="w-6 h-6 text-white" />
                  </div>
                  <h1 className="text-4xl font-bold gradient-text">Voice Library</h1>
                </div>
                <p className="text-text-tertiary text-lg">
                  Manage your voice collection and test synthesis
                </p>
              </div>
              
              <div className="flex items-center gap-3">
                <Button
                  variant="ghost"
                  onClick={loadVoices}
                  disabled={loading}
                >
                  <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
                  Refresh
                </Button>
                
                {combineMode ? (
                  <>
                    <Button
                      variant="secondary"
                      onClick={() => {
                        setCombineMode(false);
                        setSelectedForCombine(new Set());
                      }}
                    >
                      <X className="w-4 h-4" />
                      Cancel
                    </Button>
                    <Button
                      onClick={handleCombineVoices}
                      disabled={selectedForCombine.size < 2}
                    >
                      <Combine className="w-4 h-4" />
                      Combine ({selectedForCombine.size})
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      variant="secondary"
                      onClick={() => setCombineMode(true)}
                    >
                      <Combine className="w-4 h-4" />
                      Combine Voices
                    </Button>
                    <Button onClick={() => setShowUploadModal(true)}>
                      <Plus className="w-4 h-4" />
                      Add Voice
                    </Button>
                  </>
                )}
              </div>
            </div>

            {/* Search and filters */}
            <div className="flex items-center gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-tertiary" />
                <input
                  type="text"
                  placeholder="Search voices..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-12 pr-4 py-3 rounded-xl bg-bg-secondary/50 border border-white/5 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-accent/50"
                />
              </div>
              
              <div className="badge">
                <FileAudio className="w-3 h-3 mr-1" />
                {Object.keys(voices).length} voices
              </div>
            </div>
          </motion.div>

          {/* Voice grid */}
          <div className="flex-1 overflow-y-auto scrollbar-hide">
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <Loader2 className="w-8 h-8 animate-spin text-accent" />
              </div>
            ) : filteredVoices.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex flex-col items-center justify-center h-64 text-center"
              >
                <div className="w-16 h-16 rounded-2xl bg-bg-secondary/50 flex items-center justify-center mb-4">
                  <FolderOpen className="w-8 h-8 text-text-tertiary" />
                </div>
                <p className="text-text-secondary text-lg mb-2">No voices found</p>
                <p className="text-text-tertiary mb-6">
                  {searchQuery ? 'Try adjusting your search' : 'Upload your first voice to get started'}
                </p>
                {!searchQuery && (
                  <Button onClick={() => setShowUploadModal(true)}>
                    <Plus className="w-4 h-4" />
                    Add Voice
                  </Button>
                )}
              </motion.div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pb-4">
                {filteredVoices.map(([voiceName, voice], index) => (
                  <VoiceCard
                    key={voiceName}
                    voiceName={voiceName}
                    voice={voice}
                    index={index}
                    isSelected={selectedVoices.has(voiceName)}
                    isSelectedForCombine={selectedForCombine.has(voiceName)}
                    combineMode={combineMode}
                    isExpanded={expandedVoice === voiceName}
                    isPlaying={audioPlayer?.voiceName === voiceName && audioPlayer.isPlaying}
                    isTesting={testingVoice === voiceName}
                    currentlyPlayingFile={audioPlayer?.voiceName === voiceName ? audioPlayer.fileId : null}
                    onSelect={() => {
                      if (combineMode) {
                        setSelectedForCombine(prev => {
                          const newSet = new Set(prev);
                          if (newSet.has(voiceName)) {
                            newSet.delete(voiceName);
                          } else {
                            newSet.add(voiceName);
                          }
                          return newSet;
                        });
                      } else {
                        setSelectedVoices(prev => {
                          const newSet = new Set(prev);
                          if (newSet.has(voiceName)) {
                            newSet.delete(voiceName);
                          } else {
                            newSet.add(voiceName);
                          }
                          return newSet;
                        });
                      }
                    }}
                    onExpand={() => setExpandedVoice(expandedVoice === voiceName ? null : voiceName)}
                    onTest={() => {
                      setTestingVoice(voiceName);
                      handleTestVoice(voiceName);
                    }}
                    onDelete={() => handleDeleteVoice(voiceName)}
                    onPlayFile={(fileName) => handlePlayFile(voiceName, fileName)}
                    onDownloadFile={(fileName) => handleDownloadFile(voiceName, fileName)}
                    onDeleteFile={(fileName) => handleDeleteFile(voiceName, fileName)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Upload Modal */}
        <AnimatePresence>
          {showUploadModal && (
            <UploadModal
              onClose={() => {
                setShowUploadModal(false);
                setNewVoiceName('');
              }}
              onUpload={handleUpload}
              uploading={uploadingVoice !== null}
              voiceName={newVoiceName}
              setVoiceName={setNewVoiceName}
              existingVoices={Object.keys(voices)}
            />
          )}
        </AnimatePresence>

        {/* Test TTS Panel */}
        <AnimatePresence>
          {testingVoice && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="fixed bottom-8 right-8 w-96 card p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
                  <Wand2 className="w-5 h-5 text-accent" />
                  Test Voice
                </h3>
                <button
                  onClick={() => setTestingVoice(null)}
                  className="text-text-tertiary hover:text-text-primary"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <textarea
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Enter text to test..."
                className="w-full min-h-[100px] bg-bg-tertiary rounded-lg p-3 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/50 mb-4 resize-none"
              />
              
              <Button
                onClick={() => handleTestVoice(testingVoice)}
                loading={uploadingVoice === testingVoice}
                className="w-full"
              >
                <Sparkles className="w-4 h-4" />
                Generate Speech
              </Button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Audio Player */}
        <AnimatePresence>
          {audioPlayer && (
            <motion.div
              initial={{ opacity: 0, y: 100 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 100 }}
              className="fixed bottom-0 left-0 right-0 glass-border glass p-4"
            >
              <div className="max-w-7xl mx-auto flex items-center gap-4">
                <button
                  onClick={togglePlayPause}
                  className="w-12 h-12 rounded-xl bg-accent hover:bg-accent-hover flex items-center justify-center text-white transition-colors"
                >
                  {audioPlayer.isPlaying ? (
                    <Pause className="w-5 h-5" />
                  ) : (
                    <Play className="w-5 h-5 ml-0.5" />
                  )}
                </button>
                
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-text-primary capitalize">
                      {audioPlayer.voiceName.replace(/_/g, ' ')}
                    </span>
                    <span className="text-xs text-text-tertiary font-mono">
                      {formatDuration(audioPlayer.currentTime)} / {formatDuration(audioPlayer.duration)}
                    </span>
                  </div>
                  <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-accent to-purple-500"
                      style={{
                        width: `${(audioPlayer.currentTime / audioPlayer.duration) * 100 || 0}%`,
                      }}
                    />
                  </div>
                </div>
                
                <button
                  onClick={() => {
                    if (audioRef.current) {
                      audioRef.current.pause();
                      audioRef.current.src = '';
                    }
                    setAudioPlayer(null);
                  }}
                  className="text-text-tertiary hover:text-text-primary"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </Layout>
  );
};

// Voice Card Component
interface VoiceCardProps {
  voiceName: string;
  voice: Voice;
  index: number;
  isSelected: boolean;
  isSelectedForCombine: boolean;
  combineMode: boolean;
  isExpanded: boolean;
  isPlaying: boolean;
  isTesting: boolean;
  currentlyPlayingFile: string | null;
  onSelect: () => void;
  onExpand: () => void;
  onTest: () => void;
  onDelete: () => void;
  onPlayFile: (fileName: string) => void;
  onDownloadFile: (fileName: string) => void;
  onDeleteFile: (fileName: string) => void;
}

const VoiceCard: React.FC<VoiceCardProps> = ({
  voiceName,
  voice,
  index,
  isSelected,
  isSelectedForCombine,
  combineMode,
  isExpanded,
  isPlaying,
  isTesting,
  currentlyPlayingFile,
  onSelect,
  onExpand,
  onTest,
  onDelete,
  onPlayFile,
  onDownloadFile,
  onDeleteFile,
}) => {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className={cn(
        'card-hover p-6 relative group',
        (isSelected || isSelectedForCombine) && 'ring-2 ring-accent',
        isExpanded && 'lg:col-span-2'
      )}
    >
      {combineMode && (
        <div 
          className="absolute top-4 left-4 z-10 cursor-pointer"
          onClick={onSelect}
        >
          <div className={cn(
            'w-6 h-6 rounded-lg border-2 flex items-center justify-center transition-all',
            isSelectedForCombine
              ? 'bg-accent border-accent'
              : 'bg-bg-tertiary border-border'
          )}>
            {isSelectedForCombine && <Check className="w-4 h-4 text-white" />}
          </div>
        </div>
      )}

      <div className="flex items-start justify-between mb-4">
        <div 
          className="flex items-center gap-3 cursor-pointer flex-1"
          onClick={!combineMode ? onExpand : undefined}
        >
          <div className={cn(
            'w-12 h-12 rounded-xl flex items-center justify-center transition-all',
            isPlaying
              ? 'bg-gradient-to-br from-accent to-purple-500 shadow-lg shadow-accent/30'
              : 'bg-bg-tertiary group-hover:bg-bg-hover'
          )}>
            <Mic2 className={cn(
              'w-6 h-6',
              isPlaying ? 'text-white' : 'text-text-tertiary group-hover:text-text-secondary'
            )} />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-text-primary capitalize">
              {voiceName.replace(/_/g, ' ')}
            </h3>
            <div className="flex items-center gap-2 text-xs text-text-tertiary">
              <span>{voice.files.length || 0} files</span>
              {voice.totalSize > 0 && (
                <>
                  <span>•</span>
                  <span>{formatFileSize(voice.totalSize)}</span>
                </>
              )}
            </div>
          </div>
        </div>

        {!combineMode && (
          <div className="relative">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className="p-2 hover:bg-bg-hover rounded-lg transition-colors"
            >
              <MoreVertical className="w-4 h-4 text-text-tertiary" />
            </button>

            <AnimatePresence>
              {showMenu && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setShowMenu(false)}
                  />
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: -10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: -10 }}
                    className="absolute right-0 top-full mt-2 w-48 card p-2 z-20"
                  >
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete();
                        setShowMenu(false);
                      }}
                      className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-error hover:bg-error/10 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete Voice
                    </button>
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Expanded file list */}
      <AnimatePresence>
        {isExpanded && voice.files.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden mb-4"
          >
            <div className="border-t border-white/5 pt-4 space-y-2">
              {voice.files.map((file) => (
                <div
                  key={file.id}
                  className={cn(
                    'flex items-center gap-3 p-3 rounded-lg bg-bg-tertiary/50 hover:bg-bg-tertiary transition-colors',
                    currentlyPlayingFile === file.name && 'ring-1 ring-accent'
                  )}
                >
                  <button
                    onClick={() => onPlayFile(file.name)}
                    className="w-8 h-8 rounded-lg bg-accent/10 hover:bg-accent/20 flex items-center justify-center text-accent transition-colors"
                  >
                    {currentlyPlayingFile === file.name && isPlaying ? (
                      <Pause className="w-4 h-4" />
                    ) : (
                      <Play className="w-4 h-4 ml-0.5" />
                    )}
                  </button>
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-text-secondary truncate">{file.name}</p>
                    <div className="flex items-center gap-2 text-xs text-text-tertiary">
                      <span>{formatFileSize(file.size)}</span>
                      {file.duration && (
                        <>
                          <span>•</span>
                          <span>{formatDuration(file.duration)}</span>
                        </>
                      )}
                    </div>
                  </div>
                  
                  <button
                    onClick={() => onDownloadFile(file.name)}
                    className="p-2 hover:bg-bg-hover rounded-lg text-text-tertiary hover:text-text-primary transition-colors"
                    title="Download"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  
                  <button
                    onClick={() => onDeleteFile(file.name)}
                    className="p-2 hover:bg-bg-hover rounded-lg text-text-tertiary hover:text-error transition-colors"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!combineMode && (
        <div className="flex items-center gap-2 mt-4">
          <Button
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onTest();
            }}
            loading={isTesting}
            className="flex-1"
          >
            <Wand2 className="w-4 h-4" />
            Test
          </Button>
          
          <Button
            size="sm"
            variant="ghost"
            onClick={onExpand}
          >
            {isExpanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </Button>
        </div>
      )}

      {isPlaying && (
        <motion.div
          className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-accent to-purple-500"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      )}
    </motion.div>
  );
};

// Upload Modal Component
interface UploadModalProps {
  onClose: () => void;
  onUpload: (voiceName: string, files: FileList) => void;
  uploading: boolean;
  voiceName: string;
  setVoiceName: (name: string) => void;
  existingVoices: string[];
}

const UploadModal: React.FC<UploadModalProps> = ({
  onClose,
  onUpload,
  uploading,
  voiceName,
  setVoiceName,
  existingVoices,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFiles(e.target.files);
    }
  };

  const handleSubmit = () => {
    if (!voiceName.trim()) {
      toast.error('Please enter a voice name');
      return;
    }
    
    if (!selectedFiles || selectedFiles.length === 0) {
      toast.error('Please select at least one file');
      return;
    }
    
    onUpload(voiceName, selectedFiles);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="w-full max-w-2xl card p-8"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-text-primary flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-accent to-purple-500">
              <Upload className="w-6 h-6 text-white" />
            </div>
            Add New Voice
          </h2>
          <button
            onClick={onClose}
            className="text-text-tertiary hover:text-text-primary transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="space-y-6">
          {/* Voice name input */}
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Voice Name
            </label>
            <input
              type="text"
              value={voiceName}
              onChange={(e) => setVoiceName(e.target.value.replace(/[^a-zA-Z0-9_]/g, '_'))}
              placeholder="my_awesome_voice"
              className="w-full px-4 py-3 rounded-xl bg-bg-tertiary border border-white/5 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-accent/50"
            />
            <p className="text-xs text-text-tertiary mt-1">
              Use only letters, numbers, and underscores
            </p>
            
            {voiceName && existingVoices.includes(voiceName) && (
              <div className="flex items-center gap-2 mt-2 text-warning">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">This voice already exists. Files will be added to it.</span>
              </div>
            )}
          </div>

          {/* File drop zone */}
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Audio Files
            </label>
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={cn(
                'relative border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer',
                dragActive
                  ? 'border-accent bg-accent/5'
                  : 'border-white/10 hover:border-white/20 hover:bg-bg-tertiary/50'
              )}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".wav,.mp3"
                onChange={handleFileSelect}
                className="hidden"
              />
              
              <FileAudio className="w-16 h-16 mx-auto mb-4 text-text-tertiary" />
              
              {selectedFiles && selectedFiles.length > 0 ? (
                <div>
                  <p className="text-text-primary font-medium mb-2">
                    {selectedFiles.length} file(s) selected
                  </p>
                  <div className="space-y-1 mb-4">
                    {Array.from(selectedFiles).map((file, i) => (
                      <p key={i} className="text-sm text-text-tertiary">
                        {file.name} ({formatFileSize(file.size)})
                      </p>
                    ))}
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                    className="text-sm text-accent hover:text-accent-hover"
                  >
                    Choose different files
                  </button>
                </div>
              ) : (
                <div>
                  <p className="text-text-primary mb-2">
                    Drag and drop audio files here
                  </p>
                  <p className="text-text-tertiary text-sm mb-4">
                    or click to browse
                  </p>
                  <p className="text-text-tertiary text-xs">
                    Supports .wav and .mp3 files
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <Button
              variant="secondary"
              onClick={onClose}
              disabled={uploading}
              className="flex-1"
            >
              Cancel
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={!voiceName.trim() || !selectedFiles || uploading}
              loading={uploading}
              className="flex-1"
            >
              <Upload className="w-4 h-4" />
              Upload
            </Button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default VoiceLibrary;

