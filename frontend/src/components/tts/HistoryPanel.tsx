import React, { useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, Trash2, Download, Clock, X } from 'lucide-react';
import { cn, formatDuration, truncate } from '../../lib/utils';
import { useTTSStore, HistoryItem } from '../../stores/ttsStore';
import Button from '../ui/Button';

interface HistoryItemCardProps {
  item: HistoryItem;
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onDelete: () => void;
  onDownload: () => void;
}

const HistoryItemCard: React.FC<HistoryItemCardProps> = ({
  item,
  isPlaying,
  onPlay,
  onPause,
  onDelete,
  onDownload,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="history-item group"
    >
      <div className="flex items-start gap-3">
        {/* Play button */}
        <button
          onClick={isPlaying ? onPause : onPlay}
          className={cn(
            'w-10 h-10 rounded-lg flex items-center justify-center shrink-0',
            'transition-colors',
            isPlaying 
              ? 'bg-accent text-white' 
              : 'bg-bg-tertiary text-text-secondary hover:bg-accent hover:text-white'
          )}
        >
          {isPlaying ? (
            <Pause className="w-4 h-4" />
          ) : (
            <Play className="w-4 h-4 ml-0.5" />
          )}
        </button>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text-primary truncate-2 leading-snug">
            {item.text}
          </p>
          
          <div className="flex items-center gap-2 mt-1.5">
            <span className="badge-default">{item.voice}</span>
            <span className="badge-accent">{item.backend.toUpperCase()}</span>
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
            className="btn-icon p-1.5"
            title="Download"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onDelete}
            className="btn-icon p-1.5 hover:text-error"
            title="Delete"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Parameters (collapsed by default) */}
      <details className="mt-2">
        <summary className="text-xxs text-text-tertiary cursor-pointer hover:text-text-secondary">
          View parameters
        </summary>
        <div className="mt-1.5 text-xxs text-text-tertiary font-mono bg-bg-tertiary rounded px-2 py-1">
          {Object.entries(item.params || {}).map(([key, value]) => (
            <span key={key} className="inline-block mr-2">
              {key}={String(value)}
            </span>
          ))}
        </div>
      </details>
    </motion.div>
  );
};

const HistoryPanel: React.FC = () => {
  const { history, removeFromHistory, clearHistory } = useTTSStore();
  const [playingId, setPlayingId] = React.useState<string | null>(null);
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
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="panel-header px-4 pt-4">
        <h2 className="panel-title">History</h2>
        {history.length > 0 && (
          <button
            onClick={clearHistory}
            className="text-xs text-text-tertiary hover:text-error transition-colors"
          >
            Clear all
          </button>
        )}
      </div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-12 h-12 rounded-full bg-bg-tertiary flex items-center justify-center mb-3">
              <Clock className="w-6 h-6 text-text-tertiary" />
            </div>
            <p className="text-sm text-text-secondary">No generations yet</p>
            <p className="text-xs text-text-tertiary mt-1">
              Your generated audio will appear here
            </p>
          </div>
        ) : (
          <AnimatePresence mode="popLayout">
            <div className="space-y-2">
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
            </div>
          </AnimatePresence>
        )}
      </div>
    </div>
  );
};

export default HistoryPanel;

