import React, { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bookmark,
  Download,
  Trash2,
  Pencil,
  Check,
  X,
  Library,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import { downloadBlob, formatDuration } from '../../lib/utils';
import { deleteSfxBlob, getSfxBlob } from '../../services/sfxAudioDb';
import { ensureAudioWavBlob } from '../../services/sfxApi';
import { SfxLibraryItem, useSfxStore } from '../../stores/sfxStore';

function downloadFilename(name: string): string {
  const safe = name
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '_')
    .slice(0, 60);
  return `${safe || 'sound_effect'}.wav`;
}

interface LoadedLibraryItem extends SfxLibraryItem {
  audioUrl: string | null;
}

const SfxLibraryPanel: React.FC = () => {
  const { library, removeFromLibrary, renameLibraryItem } = useSfxStore();
  const [loaded, setLoaded] = useState<LoadedLibraryItem[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const urlsRef = useRef<string[]>([]);

  const revokeUrls = useCallback(() => {
    urlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    urlsRef.current = [];
  }, []);

  useEffect(() => {
    let cancelled = false;

    const hydrate = async () => {
      revokeUrls();
      const items: LoadedLibraryItem[] = [];
      for (const item of library) {
        const blob = await getSfxBlob(item.id);
        if (cancelled) return;
        if (blob) {
          const wavBlob = ensureAudioWavBlob(blob);
          const url = URL.createObjectURL(wavBlob);
          urlsRef.current.push(url);
          items.push({ ...item, audioUrl: url });
        } else {
          items.push({ ...item, audioUrl: null });
        }
      }
      if (!cancelled) setLoaded(items);
    };

    hydrate();
    return () => {
      cancelled = true;
      revokeUrls();
    };
  }, [library, revokeUrls]);

  const handleDownload = async (item: LoadedLibraryItem) => {
    const blob =
      (await getSfxBlob(item.id)) ||
      (item.audioUrl
        ? await fetch(item.audioUrl).then((r) => r.blob())
        : null);
    if (!blob) {
      toast.error('Could not load audio');
      return;
    }
    downloadBlob(blob, downloadFilename(item.name));
  };

  const handleDelete = async (id: string) => {
    await deleteSfxBlob(id);
    removeFromLibrary(id);
    toast.success('Removed from library');
  };

  const startRename = (item: LoadedLibraryItem) => {
    setEditingId(item.id);
    setEditName(item.name);
  };

  const commitRename = () => {
    if (editingId && editName.trim()) {
      renameLibraryItem(editingId, editName.trim());
    }
    setEditingId(null);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-5 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-gradient-to-br from-rose-500 to-orange-500 shadow-lg shadow-rose-500/20">
            <Library className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="font-semibold text-white">Saved Effects</h2>
            <p className="text-xs text-text-tertiary">
              {library.length} persisted locally
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {loaded.length === 0 ? (
          <div className="text-center py-12 px-4">
            <Bookmark className="w-10 h-10 text-text-tertiary mx-auto mb-3 opacity-50" />
            <p className="text-sm text-text-secondary">No saved effects yet</p>
            <p className="text-xs text-text-tertiary mt-1">
              Generate variants, pick your favorite, then save to library
            </p>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {loaded.map((item) => (
              <motion.div
                key={item.id}
                layout
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="glow-card p-4 rounded-xl border border-white/5 bg-bg-secondary/40"
              >
                {editingId === item.id ? (
                  <div className="flex items-center gap-2 mb-3">
                    <input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="flex-1 bg-bg-tertiary border border-white/10 rounded-lg px-3 py-1.5 text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-rose-500/50"
                      autoFocus
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') commitRename();
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                    />
                    <button
                      onClick={commitRename}
                      className="p-1.5 rounded-lg bg-green-500/20 text-green-400 hover:bg-green-500/30"
                    >
                      <Check className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setEditingId(null)}
                      className="p-1.5 rounded-lg bg-white/5 text-text-tertiary hover:bg-white/10"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="min-w-0">
                      <p className="font-medium text-white text-sm truncate">
                        {item.name}
                      </p>
                      <p className="text-xs text-text-tertiary line-clamp-2 mt-0.5">
                        {item.prompt}
                      </p>
                    </div>
                    <button
                      onClick={() => startRename(item)}
                      className="shrink-0 p-1.5 rounded-lg text-text-tertiary hover:text-white hover:bg-white/5"
                      title="Rename"
                    >
                      <Pencil className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )}

                <div className="mt-3 space-y-3">
                  {item.audioUrl ? (
                    <audio
                      src={item.audioUrl}
                      controls
                      preload="metadata"
                      className="w-full h-9 rounded-lg opacity-90"
                      onError={() => toast.error('Could not decode saved audio')}
                    />
                  ) : (
                    <p className="text-xs text-amber-400/90">Audio missing — delete and re-save</p>
                  )}

                  <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-text-tertiary">
                    {formatDuration(item.duration)} · {item.params.seconds}s clip
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => handleDownload(item)}
                      className="p-2 rounded-lg bg-white/5 text-text-secondary hover:bg-white/10 hover:text-white"
                      title="Download WAV"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="p-2 rounded-lg bg-white/5 text-text-tertiary hover:bg-red-500/20 hover:text-red-400"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
};

export default SfxLibraryPanel;
