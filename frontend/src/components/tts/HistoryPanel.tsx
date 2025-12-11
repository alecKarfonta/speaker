import React, { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Trash2, 
  Download, 
  Clock, 
  MoreHorizontal,
  Volume2,
  ChevronRight,
  Waves
} from 'lucide-react';
import { cn, truncate } from '../../lib/utils';
import { useTTSStore, HistoryItem } from '../../stores/ttsStore';

interface HistoryItemCardProps {
  item: HistoryItem;
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onDelete: () => void;
  onDownload: () => void;
}

const WaveformPlaceholder: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => (
  <div className="flex items-center gap-0.5 h-8">
    {[...Array(20)].map((_, i) => (
      <motion.div
        key={i}
        className="w-1 bg-accent/60 rounded-full"
        initial={{ height: 4 }}
        animate={isPlaying ? {
          height: [4, Math.random() * 24 + 8, 4],
        } : { height: Math.random() * 12 + 4 }}
        transition={isPlaying ? {
          duration: 0.5,
          repeat: Infinity,
          delay: i * 0.05,
        } : { duration: 0 }}
      />
    ))}
  </div>
);

const HistoryItemCard: React.FC<HistoryItemCardProps> = ({
  item,
  isPlaying,
  onPlay,
  onPause,
  onDelete,
  onDownload,
}) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className={cn(
        'group rounded-xl border transition-all duration-200',
        isPlaying 
          ? 'bg-accent/10 border-accent/30' 
          : 'bg-bg-secondary/50 border-border hover:border-border-strong hover:bg-bg-secondary'
      )}
    >
      {/* Main content */}
      <div className="p-3">
        <div className="flex items-start gap-3">
          {/* Play button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={isPlaying ? onPause : onPlay}
            className={cn(
              'w-10 h-10 rounded-lg flex items-center justify-center shrink-0',
              'transition-all duration-200',
              isPlaying 
                ? 'bg-accent text-white shadow-lg shadow-accent/25' 
                : 'bg-bg-tertiary text-text-secondary hover:bg-accent hover:text-white'
            )}
          >
            {isPlaying ? (
              <Pause className="w-4 h-4" />
            ) : (
              <Play className="w-4 h-4 ml-0.5" />
            )}
          </motion.button>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <p className={cn(
              'text-sm leading-snug',
              expanded ? 'text-text-primary' : 'text-text-primary truncate'
            )}>
              {expanded ? item.text : truncate(item.text, 60)}
            </p>
            
            {/* Waveform */}
            {isPlaying && (
              <div className="mt-2">
                <WaveformPlaceholder isPlaying={isPlaying} />
              </div>
            )}
            
            {/* Meta info */}
            <div className="flex items-center gap-2 mt-2 flex-wrap">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs bg-bg-tertiary text-text-secondary">
                <Volume2 className="w-3 h-3" />
                {item.voice}
              </span>
              <span className={cn(
                'px-2 py-0.5 rounded-md text-xs font-medium',
                item.backend === 'glm-tts' 
                  ? 'bg-accent/20 text-accent' 
                  : 'bg-purple-500/20 text-purple-400'
              )}>
                {item.backend.toUpperCase()}
              </span>
              <span className="text-xxs text-text-tertiary flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {new Date(item.timestamp).toLocaleTimeString([], { 
                  hour: '2-digit', 
                  minute: '2-digit' 
                })}
              </span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={onDownload}
              className="p-1.5 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-bg-tertiary transition-colors"
              title="Download"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={onDelete}
              className="p-1.5 rounded-lg text-text-tertiary hover:text-error hover:bg-error/10 transition-colors"
              title="Delete"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Expandable parameters section */}
      {item.params && Object.keys(item.params).length > 0 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full px-3 py-2 border-t border-border/50 flex items-center justify-between text-xs text-text-tertiary hover:text-text-secondary hover:bg-bg-tertiary/50 transition-colors"
        >
          <span>Parameters</span>
          <ChevronRight className={cn(
            'w-3 h-3 transition-transform',
            expanded && 'rotate-90'
          )} />
        </button>
      )}

      <AnimatePresence>
        {expanded && item.params && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-1">
              <div className="flex flex-wrap gap-1">
                {Object.entries(item.params).map(([key, value]) => (
                  <span 
                    key={key} 
                    className="inline-flex items-center px-1.5 py-0.5 rounded text-xxs bg-bg-tertiary text-text-tertiary font-mono"
                  >
                    {key}={typeof value === 'number' ? value.toFixed?.(2) || value : String(value)}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const HistoryPanel: React.FC = () => {
  const { history, removeFromHistory, clearHistory } = useTTSStore();
  const [playingId, setPlayingId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handlePlay = (item: HistoryItem) => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    
    const audio = new Audio(item.audioUrl);
    audio.onended = () => setPlayingId(null);
    audio.play();
    audioRef.current = audio;
    setPlayingId(item.id);
  };

  const handlePause = () => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    setPlayingId(null);
  };

  const handleDownload = (item: HistoryItem) => {
    const a = document.createElement('a');
    a.href = item.audioUrl;
    a.download = `tts-${item.voice}-${Date.now()}.wav`;
    a.click();
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-bg-secondary to-bg-primary">
      {/* Header */}
      <div className="px-4 py-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Waves className="w-4 h-4 text-white" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-text-primary">History</h2>
              <p className="text-xxs text-text-tertiary">{history.length} generations</p>
            </div>
          </div>
          {history.length > 0 && (
            <button
              onClick={clearHistory}
              className="text-xs text-text-tertiary hover:text-error transition-colors px-2 py-1 rounded hover:bg-error/10"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {history.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center h-full text-center py-12"
          >
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-bg-tertiary to-bg-secondary flex items-center justify-center mb-4 shadow-inner">
              <Clock className="w-8 h-8 text-text-tertiary" />
            </div>
            <p className="text-sm font-medium text-text-secondary mb-1">No generations yet</p>
            <p className="text-xs text-text-tertiary max-w-[200px]">
              Your generated audio will appear here for easy playback and download
            </p>
          </motion.div>
        ) : (
          <AnimatePresence mode="popLayout">
            {history.map((item) => (
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
