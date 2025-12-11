import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Voice {
  id: string;
  name: string;
}

export interface TTSParams {
  // GLM-TTS params
  sampling: number;
  temperature: number;
  glm_top_p: number;
  min_token_text_ratio: number;
  max_token_text_ratio: number;
  beam_size: number;
  repetition_penalty: number;
  sample_method: 'ras' | 'topk';
  // XTTS params
  tau: number;
  gpt_cond_len: number;
  top_k: number;
  top_p: number;
  decoder_iterations: number;
  split_sentences: boolean;
  // Output
  output_format: 'wav' | 'raw';
}

export interface HistoryItem {
  id: string;
  text: string;
  voice: string;
  language: string;
  backend: 'glm-tts' | 'xtts';
  params: Partial<TTSParams>;
  audioUrl: string;
  timestamp: number;
  duration?: number;
}

interface TTSState {
  // Text input
  text: string;
  setText: (text: string) => void;
  
  // Voice selection
  selectedVoice: string;
  setSelectedVoice: (voice: string) => void;
  voices: Voice[];
  setVoices: (voices: Voice[]) => void;
  
  // Language
  language: string;
  setLanguage: (language: string) => void;
  
  // Backend
  backend: 'glm-tts' | 'xtts';
  setBackend: (backend: 'glm-tts' | 'xtts') => void;
  
  // Parameters
  params: TTSParams;
  setParam: <K extends keyof TTSParams>(key: K, value: TTSParams[K]) => void;
  resetParams: () => void;
  
  // History
  history: HistoryItem[];
  addToHistory: (item: HistoryItem) => void;
  removeFromHistory: (id: string) => void;
  clearHistory: () => void;
  
  // UI State
  isGenerating: boolean;
  setIsGenerating: (generating: boolean) => void;
  showAdvanced: boolean;
  setShowAdvanced: (show: boolean) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
}

const DEFAULT_PARAMS: TTSParams = {
  // GLM-TTS defaults
  sampling: 25,
  temperature: 1.0,
  glm_top_p: 0.8,
  min_token_text_ratio: 8,
  max_token_text_ratio: 30,
  beam_size: 1,
  repetition_penalty: 0.1,
  sample_method: 'ras',
  // XTTS defaults
  tau: 0.6,
  gpt_cond_len: 10,
  top_k: 10,
  top_p: 0.9,
  decoder_iterations: 30,
  split_sentences: true,
  // Output
  output_format: 'wav',
};

export const useTTSStore = create<TTSState>()(
  persist(
    (set) => ({
      // Text
      text: '',
      setText: (text) => set({ text }),
      
      // Voice
      selectedVoice: '',
      setSelectedVoice: (selectedVoice) => set({ selectedVoice }),
      voices: [],
      setVoices: (voices) => set({ voices }),
      
      // Language
      language: 'en',
      setLanguage: (language) => set({ language }),
      
      // Backend
      backend: 'glm-tts',
      setBackend: (backend) => set({ backend }),
      
      // Parameters
      params: DEFAULT_PARAMS,
      setParam: (key, value) =>
        set((state) => ({
          params: { ...state.params, [key]: value },
        })),
      resetParams: () => set({ params: DEFAULT_PARAMS }),
      
      // History
      history: [],
      addToHistory: (item) =>
        set((state) => ({
          history: [item, ...state.history].slice(0, 50), // Keep last 50
        })),
      removeFromHistory: (id) =>
        set((state) => ({
          history: state.history.filter((h) => h.id !== id),
        })),
      clearHistory: () => set({ history: [] }),
      
      // UI State
      isGenerating: false,
      setIsGenerating: (isGenerating) => set({ isGenerating }),
      showAdvanced: false,
      setShowAdvanced: (showAdvanced) => set({ showAdvanced }),
      sidebarCollapsed: false,
      setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
    }),
    {
      name: 'tts-storage',
      partialize: (state) => ({
        history: state.history,
        params: state.params,
        backend: state.backend,
        language: state.language,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    }
  )
);

