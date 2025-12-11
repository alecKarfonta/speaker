import React, { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Trash2, 
  Download, 
  Clock, 
  Volume2,
  ChevronRight,
  History,
  Sparkles
} from 'lucide-react';
import { cn, truncate } from '../../lib/utils';
import { useTTSStore, HistoryItem } from '../../stores/ttsStore';

// Audio waveform visualization
const AudioWaveform: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  const bars = 24;
  return (
    <div className="flex items-center justify-center gap-[2px] h-10 px-2">
      {[...Array(bars)].map((_, i) => (
        <motion.div
          key={i}
          className="w-[3px] rounded-full bg-gradient-to-t from-accent to-cyan"
          initial={{ height: 4 }}
          animate={isPlaying ? {
            height: [4, Math.random() * 32 + 8, 4],
            opacity: [0.5, 1, 0.5],
          } : { 
            height: Math.random() * 16 + 4,
            opacity: 0.3,
          }}
          transition={isPlaying ? {
            duration: 0.4,
            repeat: Infinity,
            delay: i * 0.03,
            ease: "easeInOut",
          } : { duration: 0.3 }}
        />
      ))}
    </div>
  );
};

const HistoryItemCard: React.FC<{
  item: HistoryItem;
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onDelete: () => void;
  onDownload: () => void;
}> = ({ item, isPlaying, onPlay, onPause, onDelete, onDownload }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className={cn(
        'history-item group cursor-pointer',
        isPlaying && 'history-item-active'
      )}
      onClick={() => isPlaying ? onPause() : onPlay()}
    >
      {/* Playing indicator line */}
      {isPlaying && (
        <motion.div
          layoutId="playingIndicator"
          className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-accent via-purple-500 to-pink-500 rounded-l-xl"
        />
      )}
      
      <div className="flex items-start gap-4">
        {/* Play button with ring animation */}
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={(e) => {
            e.stopPropagation();
            isPlaying ? onPause() : onPlay();
          }}
          className={cn(
            'relative w-12 h-12 rounded-xl flex items-center justify-center shrink-0',
            'transition-all duration-300',
            isPlaying 
              ? 'bg-gradient-to-br from-accent to-purple-500' 
              : 'bg-white/5 hover:bg-white/10'
          )}
        >
          {isPlaying && (
            <motion.div
              className="absolute inset-0 rounded-xl border-2 border-accent"
              animate={{ scale: [1, 1.3, 1], opacity: [0.8, 0, 0.8] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
          )}
          {isPlaying ? (
            <Pause className="w-5 h-5 text-white" />
          ) : (
            <Play className="w-5 h-5 text-text-tertiary group-hover:text-white ml-0.5" />
          )}
        </motion.button>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Text preview */}
          <motion.p 
            className={cn(
              'text-sm leading-relaxed transition-colors',
              isPlaying ? 'text-white' : 'text-text-secondary group-hover:text-text-primary'
            )}
            layout
          >
            {expanded ? item.text : truncate(item.text, 80)}
          </motion.p>
          
          {/* Waveform when playing */}
          <AnimatePresence>
            {isPlaying && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3"
              >
                <AudioWaveform isPlaying={isPlaying} />
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Meta info */}
          <div className="flex items-center gap-2 mt-3 flex-wrap">
            <span className="badge">
              <Volume2 className="w-3 h-3 mr-1.5 text-text-tertiary" />
              {item.voice}
            </span>
            <span className={cn(
              'badge',
              item.backend === 'glm-tts' ? 'badge-accent' : 'badge-purple'
            )}>
              {item.backend.toUpperCase()}
            </span>
            <span className="text-xs text-text-tertiary flex items-center gap-1 ml-auto">
              <Clock className="w-3 h-3" />
              {new Date(item.timestamp).toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit' 
              })}
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={(e) => {
              e.stopPropagation();
              onDownload();
            }}
            className="p-2 rounded-lg bg-white/5 hover:bg-accent/20 text-text-tertiary hover:text-accent transition-all"
          >
            <Download className="w-4 h-4" />
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="p-2 rounded-lg bg-white/5 hover:bg-red-500/20 text-text-tertiary hover:text-red-400 transition-all"
          >
            <Trash2 className="w-4 h-4" />
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

const HistoryPanel: React.FC = () => {
  const { history, removeFromHistory, clearHistory } = useTTSStore();
  const [playingId, setPlayingId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handlePlay = (item: HistoryItem) => {
    if (audioRef.current) audioRef.current.pause();
    
    const audio = new Audio(item.audioUrl);
    audio.onended = () => setPlayingId(null);
    audio.play();
    audioRef.current = audio;
    setPlayingId(item.id);
  };

  const handlePause = () => {
    audioRef.current?.pause();
    setPlayingId(null);
  };

  const handleDownload = (item: HistoryItem) => {
    const a = document.createElement('a');
    a.href = item.audioUrl;
    a.download = `speaker-${item.voice}-${Date.now()}.wav`;
    a.click();
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-bg-secondary via-bg-primary to-bg-primary relative">
      {/* Subtle top gradient */}
      <div className="absolute top-0 left-0 right-0 h-32 bg-gradient-to-b from-accent/5 to-transparent pointer-events-none" />
      
      {/* Header */}
      <div className="relative px-5 py-5 border-b border-white/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                <History className="w-5 h-5 text-white" />
              </div>
              {history.length > 0 && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-accent flex items-center justify-center text-xs font-bold text-white shadow-lg"
                >
                  {history.length}
                </motion.div>
              )}
            </div>
            <div>
              <h2 className="font-semibold text-white">History</h2>
              <p className="text-xs text-text-tertiary">
                {history.length} generation{history.length !== 1 ? 's' : ''}
              </p>
            </div>
          </div>
          {history.length > 0 && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={clearHistory}
              className="text-xs text-text-tertiary hover:text-red-400 px-3 py-1.5 rounded-lg hover:bg-red-500/10 transition-all"
            >
              Clear all
            </motion.button>
          )}
        </div>
      </div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-hide">
        {history.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center h-full text-center py-12"
          >
            <div className="relative mb-6">
              <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-bg-tertiary to-bg-secondary flex items-center justify-center">
                <Sparkles className="w-8 h-8 text-text-tertiary" />
              </div>
              <motion.div
                animate={{ 
                  scale: [1, 1.2, 1],
                  opacity: [0.3, 0.6, 0.3],
                }}
                transition={{ duration: 3, repeat: Infinity }}
                className="absolute inset-0 rounded-3xl bg-accent/20 blur-xl"
              />
            </div>
            <p className="text-base font-medium text-text-secondary mb-2">No generations yet</p>
            <p className="text-sm text-text-tertiary max-w-[200px]">
              Your voice creations will appear here
            </p>
          </motion.div>
        ) : (
          <AnimatePresence mode="popLayout">
            {history.map((item, index) => (
              <HistoryItemCard
                key={item.id}
                item={item}
                isPlaying={playingId === item.id}
                onPlay={() => handlePlay(item)}
                onPause={handlePause}
                onDelete={() => removeFromHistory(item.id)}
                onDownload={() => handleDownload(item)}
              />
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
};

export default HistoryPanel;
