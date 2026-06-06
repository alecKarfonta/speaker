import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface SfxParams {
  seconds: number;
  num_inference_steps: number;
  cfg_scale: number;
  sigma_shift: number;
  seed: number | null;
  batchCount: number;
}

export interface SfxLibraryItem {
  id: string;
  name: string;
  prompt: string;
  params: Omit<SfxParams, 'batchCount'>;
  duration: number;
  createdAt: number;
}

const DEFAULT_PARAMS: SfxParams = {
  seconds: 8,
  num_inference_steps: 100,
  cfg_scale: 4.0,
  sigma_shift: 5.0,
  seed: 0,
  batchCount: 1,
};

interface SfxState {
  prompt: string;
  setPrompt: (prompt: string) => void;
  params: SfxParams;
  setParam: <K extends keyof SfxParams>(key: K, value: SfxParams[K]) => void;
  resetParams: () => void;
  library: SfxLibraryItem[];
  addToLibrary: (item: SfxLibraryItem) => void;
  removeFromLibrary: (id: string) => void;
  renameLibraryItem: (id: string, name: string) => void;
  updateLibraryItem: (id: string, patch: Partial<SfxLibraryItem>) => void;
}

export const useSfxStore = create<SfxState>()(
  persist(
    (set) => ({
      prompt: '',
      setPrompt: (prompt) => set({ prompt }),
      params: DEFAULT_PARAMS,
      setParam: (key, value) =>
        set((state) => ({ params: { ...state.params, [key]: value } })),
      resetParams: () => set({ params: DEFAULT_PARAMS }),
      library: [],
      addToLibrary: (item) =>
        set((state) => ({ library: [item, ...state.library] })),
      removeFromLibrary: (id) =>
        set((state) => ({
          library: state.library.filter((item) => item.id !== id),
        })),
      renameLibraryItem: (id, name) =>
        set((state) => ({
          library: state.library.map((item) =>
            item.id === id ? { ...item, name } : item
          ),
        })),
      updateLibraryItem: (id, patch) =>
        set((state) => ({
          library: state.library.map((item) =>
            item.id === id ? { ...item, ...patch } : item
          ),
        })),
    }),
    {
      name: 'sfx-storage',
      partialize: (state) => ({
        prompt: state.prompt,
        params: state.params,
        library: state.library,
      }),
    }
  )
);

export function defaultSfxName(prompt: string): string {
  const trimmed = prompt.trim().replace(/\s+/g, ' ');
  if (!trimmed) return 'Sound effect';
  return trimmed.length > 48 ? `${trimmed.slice(0, 48)}…` : trimmed;
}
