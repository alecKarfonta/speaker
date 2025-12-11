# Frontend Refactor Plan: TTS Interface Redesign

## Vision
Transform the current basic TTS interface into a sleek, modern, professional application that rivals commercial TTS products. The design should feel like a premium developer tool - powerful yet intuitive.

---

## Current State Analysis

### Files to Modify
| File | Lines | Purpose | Changes Needed |
|------|-------|---------|----------------|
| `src/App.tsx` | 35 | Main layout, routing | Complete redesign |
| `src/index.css` | 12 | Global styles | Expand significantly |
| `src/components/TTSGenerator.tsx` | 920 | Main TTS page | Split into components, redesign |
| `src/components/Navigation.tsx` | 38 | Nav bar | Redesign as sidebar |
| `src/components/LiveStreamTTS.tsx` | ~200 | Streaming page | Redesign to match |
| `tailwind.config.js` | 11 | Tailwind config | Add custom theme |
| `package.json` | - | Dependencies | Add new packages |

### Current Pain Points
1. **Generic Bootstrap-like appearance** - Blue header, white cards, gray backgrounds
2. **Cramped layout** - max-w-4xl constrains usability
3. **Poor information hierarchy** - All controls have equal visual weight
4. **No visual feedback** - Loading states are minimal
5. **Cluttered parameters** - 8+ sliders in a grid is overwhelming
6. **No keyboard shortcuts** - Power users need efficiency
7. **Basic audio player** - Native HTML5 audio is ugly
8. **No dark mode** - Essential for developer tools
9. **Mobile unfriendly** - Doesn't adapt well to small screens

---

## Design Direction

### Aesthetic: "Professional Audio Workstation"
Think: Figma + Linear + Descript

- **Color Palette**: Dark mode primary with optional light mode
- **Typography**: Modern sans-serif (Geist, Inter, or SF Pro)
- **Spacing**: Generous whitespace, clear sections
- **Interactions**: Smooth animations, micro-feedback
- **Layout**: Three-column workspace layout

### Color System (Dark Mode Primary)
```css
:root {
  /* Backgrounds */
  --bg-primary: #0a0a0b;      /* Main background */
  --bg-secondary: #141416;    /* Cards, panels */
  --bg-tertiary: #1c1c1f;     /* Elevated elements */
  --bg-hover: #252528;        /* Hover states */
  
  /* Borders */
  --border-subtle: #2a2a2d;
  --border-default: #3a3a3d;
  --border-strong: #4a4a4d;
  
  /* Text */
  --text-primary: #fafafa;
  --text-secondary: #a1a1aa;
  --text-tertiary: #71717a;
  
  /* Accent - Electric Blue */
  --accent-primary: #3b82f6;
  --accent-hover: #2563eb;
  --accent-glow: rgba(59, 130, 246, 0.15);
  
  /* Status */
  --success: #22c55e;
  --warning: #f59e0b;
  --error: #ef4444;
}
```

---

## Architecture: Component Breakdown

### New File Structure
```
src/
├── App.tsx                    # Redesigned layout shell
├── index.css                  # Global styles, CSS variables
├── main.tsx
│
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx        # NEW: Navigation sidebar
│   │   ├── Header.tsx         # NEW: Top bar with status
│   │   └── Layout.tsx         # NEW: Main layout wrapper
│   │
│   ├── tts/
│   │   ├── TTSWorkspace.tsx   # NEW: Main TTS workspace
│   │   ├── TextInput.tsx      # NEW: Rich text input area
│   │   ├── VoiceSelector.tsx  # NEW: Voice selection panel
│   │   ├── ParameterPanel.tsx # NEW: Collapsible params
│   │   ├── GenerateButton.tsx # NEW: Primary action button
│   │   └── HistoryPanel.tsx   # NEW: Generated audio list
│   │
│   ├── audio/
│   │   ├── AudioPlayer.tsx    # NEW: Custom audio player
│   │   ├── Waveform.tsx       # NEW: Audio waveform display
│   │   └── AudioControls.tsx  # NEW: Play/pause/download
│   │
│   ├── voice/
│   │   ├── VoiceCard.tsx      # NEW: Individual voice display
│   │   ├── VoiceUpload.tsx    # NEW: Voice upload modal
│   │   └── VoiceLibrary.tsx   # NEW: Voice management page
│   │
│   └── ui/
│       ├── Button.tsx         # NEW: Styled button component
│       ├── Slider.tsx         # NEW: Custom slider
│       ├── Select.tsx         # NEW: Custom select dropdown
│       ├── Modal.tsx          # NEW: Modal dialog
│       ├── Tooltip.tsx        # NEW: Tooltip component
│       └── Badge.tsx          # NEW: Status badges
│
├── hooks/
│   ├── useTTS.ts              # NEW: TTS API hook
│   ├── useVoices.ts           # NEW: Voice management hook
│   ├── useAudio.ts            # NEW: Audio playback hook
│   └── useKeyboard.ts         # NEW: Keyboard shortcuts
│
├── stores/
│   └── ttsStore.ts            # NEW: Zustand state store
│
├── lib/
│   ├── api.ts                 # NEW: API client
│   ├── audio.ts               # NEW: Audio utilities
│   └── constants.ts           # NEW: App constants
│
└── types/
    └── index.ts               # NEW: TypeScript types
```

---

## Phase 1: Foundation (Days 1-2)

### Checkpoint 1.1: Setup & Dependencies
**Goal**: Install required packages and configure build

**Changes to `package.json`**:
```json
{
  "dependencies": {
    // ADD these
    "zustand": "^4.5.0",          // State management
    "framer-motion": "^11.0.0",   // Animations
    "wavesurfer.js": "^7.0.0",    // Waveform visualization
    "@radix-ui/react-slider": "^1.1.0",    // Accessible slider
    "@radix-ui/react-select": "^2.0.0",    // Accessible select
    "@radix-ui/react-dialog": "^1.0.0",    // Accessible modal
    "@radix-ui/react-tooltip": "^1.0.0",   // Accessible tooltip
    "class-variance-authority": "^0.7.0",  // Component variants
    "clsx": "^2.1.0",             // Class merging
    "tailwind-merge": "^2.2.0"   // Tailwind class merging
  }
}
```

**Changes to `tailwind.config.js`**:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: 'var(--bg-primary)',
        foreground: 'var(--text-primary)',
        card: 'var(--bg-secondary)',
        border: 'var(--border-default)',
        accent: {
          DEFAULT: 'var(--accent-primary)',
          hover: 'var(--accent-hover)',
        }
      },
      fontFamily: {
        sans: ['Geist', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['Geist Mono', 'JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px var(--accent-glow)' },
          '100%': { boxShadow: '0 0 20px var(--accent-glow)' },
        }
      }
    },
  },
  plugins: [],
}
```

### Checkpoint 1.2: Global Styles & Theme
**Goal**: Establish design system foundation

**Changes to `src/index.css`**:
```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    /* Dark theme (default) */
    --bg-primary: #0a0a0b;
    --bg-secondary: #141416;
    --bg-tertiary: #1c1c1f;
    --bg-hover: #252528;
    
    --border-subtle: #2a2a2d;
    --border-default: #3a3a3d;
    
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-tertiary: #71717a;
    
    --accent-primary: #3b82f6;
    --accent-hover: #2563eb;
    --accent-glow: rgba(59, 130, 246, 0.15);
    
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
  }

  .light {
    --bg-primary: #ffffff;
    --bg-secondary: #f9fafb;
    --bg-tertiary: #f3f4f6;
    --bg-hover: #e5e7eb;
    
    --border-subtle: #e5e7eb;
    --border-default: #d1d5db;
    
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --text-tertiary: #9ca3af;
  }

  body {
    @apply bg-background text-foreground antialiased;
    font-feature-settings: 'rlig' 1, 'calt' 1;
  }

  /* Custom scrollbar */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  ::-webkit-scrollbar-track {
    background: var(--bg-secondary);
  }
  ::-webkit-scrollbar-thumb {
    background: var(--border-default);
    border-radius: 4px;
  }
  ::-webkit-scrollbar-thumb:hover {
    background: var(--border-strong);
  }
}

@layer components {
  .card {
    @apply bg-card border border-border rounded-xl;
  }
  
  .input-base {
    @apply bg-background border border-border rounded-lg px-4 py-2
           focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent
           transition-all duration-200;
  }
  
  .btn-primary {
    @apply bg-accent hover:bg-accent-hover text-white font-medium
           px-4 py-2 rounded-lg transition-all duration-200
           focus:outline-none focus:ring-2 focus:ring-accent/50;
  }
  
  .btn-secondary {
    @apply bg-bg-tertiary hover:bg-bg-hover text-text-secondary
           border border-border px-4 py-2 rounded-lg transition-all duration-200;
  }
}
```

---

## Phase 2: Layout Structure (Days 3-4)

### Checkpoint 2.1: Main Layout Shell
**Goal**: Create the overall application structure

**New file: `src/components/layout/Layout.tsx`**
```tsx
// Three-panel layout:
// [Sidebar 64px] [Main Content flex-1] [History Panel 320px]
```

**Redesigned `src/App.tsx`**:
- Remove header/footer
- Implement full-height dark layout
- Add sidebar navigation
- Add keyboard shortcut provider

### Checkpoint 2.2: Sidebar Navigation
**Goal**: Replace top nav with collapsible sidebar

**New file: `src/components/layout/Sidebar.tsx`**
- Logo/brand at top
- Navigation icons with tooltips
- Settings at bottom
- Theme toggle
- Collapse/expand button

**Visual Reference**:
```
┌──────┬────────────────────────────────────────────────┬───────────┐
│ Logo │                                                │ History   │
├──────┤                                                │           │
│ [TTS]│           Main Workspace                       │ ┌───────┐ │
│      │                                                │ │Audio 1│ │
│[Voice│                                                │ └───────┘ │
│      │                                                │ ┌───────┐ │
│[Strm]│                                                │ │Audio 2│ │
│      │                                                │ └───────┘ │
├──────┤                                                │           │
│ [Set]│                                                │           │
│ [◐◑] │                                                │           │
└──────┴────────────────────────────────────────────────┴───────────┘
```

---

## Phase 3: TTS Workspace (Days 5-8)

### Checkpoint 3.1: Text Input Component
**Goal**: Create a premium text input experience

**New file: `src/components/tts/TextInput.tsx`**

**Features**:
- Large, comfortable textarea
- Character count with visual progress bar
- Estimated duration display
- Clear button
- Paste from clipboard button
- Recent texts dropdown

**Code reference** (current): `TTSGenerator.tsx:426-441`

### Checkpoint 3.2: Voice Selector
**Goal**: Visual voice selection with preview

**New file: `src/components/tts/VoiceSelector.tsx`**

**Features**:
- Voice cards with avatar/icon
- Preview audio sample on hover
- Language badge
- Search/filter voices
- "Add Voice" card at end
- Currently selected highlight

**Code reference** (current): `TTSGenerator.tsx:443-461`

### Checkpoint 3.3: Parameter Panel
**Goal**: Organized, scannable parameters

**New file: `src/components/tts/ParameterPanel.tsx`**

**Design**:
- Collapsible sections
- Preset buttons (Fast, Balanced, Quality)
- Visual grouping by category
- Tooltips explaining each parameter
- Reset to defaults button

**Parameter Groups**:
```
┌─ Quality ──────────────────────────────┐
│ [Fast] [Balanced] [Quality]  presets   │
├────────────────────────────────────────┤
│ Sampling       ════●═══════  25        │
│ Temperature    ═══════●════  1.0       │
│ Top-P          ═════════●══  0.8       │
└────────────────────────────────────────┘

┌─ Length Control ───────────────────────┐
│ Min Ratio      ═════●══════  8         │
│ Max Ratio      ════════●═══  30        │
└────────────────────────────────────────┘

┌─ Advanced ─────────────────────────────┐
│ Beam Size      ●═══════════  1         │
│ Rep. Penalty   ══●═════════  0.1       │
│ Method         [RAS ▾]                 │
└────────────────────────────────────────┘
```

**Code reference** (current): `TTSGenerator.tsx:507-766`

### Checkpoint 3.4: Generate Button & Status
**Goal**: Clear call-to-action with feedback

**New file: `src/components/tts/GenerateButton.tsx`**

**States**:
1. **Ready**: Large blue button with glow effect
2. **Generating**: Progress animation, estimated time
3. **Success**: Brief green flash, auto-dismiss
4. **Error**: Red with retry button

**Keyboard shortcut**: `Cmd/Ctrl + Enter`

---

## Phase 4: Audio Experience (Days 9-11)

### Checkpoint 4.1: Custom Audio Player
**Goal**: Beautiful, functional audio player

**New file: `src/components/audio/AudioPlayer.tsx`**

**Features**:
- Waveform visualization (wavesurfer.js)
- Playback speed control (0.5x - 2x)
- Skip forward/back 5 seconds
- Download button
- Copy audio URL
- Fullscreen waveform option

**Visual**:
```
┌──────────────────────────────────────────────────────┐
│ ▶  ═══════════●══════════════════════  1:23 / 3:45  │
│    [waveform visualization here]                     │
│                                                      │
│ [0.5x] [1x] [1.5x] [2x]    [↓ Download] [📋 Copy]   │
└──────────────────────────────────────────────────────┘
```

### Checkpoint 4.2: History Panel
**Goal**: Persistent history with comparison features

**New file: `src/components/tts/HistoryPanel.tsx`**

**Features**:
- Scrollable list of generations
- Expandable to see full text
- Quick actions (play, download, delete, regenerate)
- Compare mode (play two side-by-side)
- Search history
- Export all as ZIP
- Persist to localStorage

**Code reference** (current): `TTSGenerator.tsx:828-914`

---

## Phase 5: Voice Management (Days 12-14)

### Checkpoint 5.1: Voice Library Page
**Goal**: Dedicated page for voice management

**New file: `src/components/voice/VoiceLibrary.tsx`**

**Features**:
- Grid of voice cards
- Upload new voice (drag & drop)
- Edit voice metadata
- Delete voice
- Voice usage statistics
- Voice sample playback

### Checkpoint 5.2: Voice Upload Modal
**Goal**: Guided voice upload experience

**New file: `src/components/voice/VoiceUpload.tsx`**

**Steps**:
1. Upload audio file (drag & drop)
2. Preview & trim audio
3. Enter voice name & metadata
4. Processing status
5. Test generation

**Code reference** (current): `TTSGenerator.tsx:768-811`

---

## Phase 6: Polish & Accessibility (Days 15-17)

### Checkpoint 6.1: Animations & Transitions
**Goal**: Smooth, professional feel

**Using Framer Motion for**:
- Page transitions
- Panel slide-in/out
- Button hover effects
- Loading states
- Success/error feedback

### Checkpoint 6.2: Keyboard Shortcuts
**Goal**: Power user efficiency

| Shortcut | Action |
|----------|--------|
| `Cmd/Ctrl + Enter` | Generate speech |
| `Space` | Play/pause audio |
| `Cmd/Ctrl + K` | Quick voice search |
| `Cmd/Ctrl + ,` | Open settings |
| `Esc` | Close modal/panel |
| `↑/↓` in history | Navigate history |

**New file: `src/hooks/useKeyboard.ts`**

### Checkpoint 6.3: Accessibility
**Goal**: WCAG 2.1 AA compliance

- Focus management
- Screen reader announcements
- Keyboard navigation
- Color contrast ratios
- Reduced motion support

---

## Phase 7: State Management (Days 18-19)

### Checkpoint 7.1: Zustand Store
**Goal**: Centralized, persistent state

**New file: `src/stores/ttsStore.ts`**
```typescript
interface TTSStore {
  // Text input
  text: string;
  setText: (text: string) => void;
  
  // Voice selection
  selectedVoice: string;
  setSelectedVoice: (voice: string) => void;
  voices: Voice[];
  
  // Parameters
  params: TTSParams;
  setParam: (key: keyof TTSParams, value: any) => void;
  resetParams: () => void;
  
  // History
  history: TTSResponse[];
  addToHistory: (response: TTSResponse) => void;
  clearHistory: () => void;
  
  // UI state
  isGenerating: boolean;
  showAdvanced: boolean;
  theme: 'dark' | 'light';
}
```

### Checkpoint 7.2: API Client
**Goal**: Type-safe API interactions

**New file: `src/lib/api.ts`**
```typescript
export const api = {
  generateSpeech: async (request: TTSRequest): Promise<Blob> => {},
  getVoices: async (): Promise<Voice[]> => {},
  uploadVoice: async (name: string, file: File): Promise<void> => {},
  getHealth: async (): Promise<HealthStatus> => {},
};
```

---

## Phase 8: Testing & Optimization (Days 20-21)

### Checkpoint 8.1: Component Testing
- Unit tests for utility functions
- Component tests for key interactions
- Integration tests for API calls

### Checkpoint 8.2: Performance
- Code splitting for routes
- Lazy loading for heavy components (waveform)
- Memoization where needed
- Bundle size analysis

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Lighthouse Performance | ~70 | >90 |
| Lighthouse Accessibility | ~80 | >95 |
| First Contentful Paint | ~2s | <1s |
| Time to Interactive | ~3s | <2s |
| Bundle Size | ~300KB | <200KB |
| Component Count | 3 | 20+ |

---

## Visual Mockup Reference

```
┌────────────────────────────────────────────────────────────────────────┐
│ ┌────┐                                                      ┌────────┐ │
│ │Logo│  Speaker TTS                              [◐] [⚙️]   │ History│ │
│ └────┘                                                      └────────┘ │
├────────┬───────────────────────────────────────────────────┬───────────┤
│        │                                                   │           │
│  [🎤]  │  ┌─────────────────────────────────────────────┐  │  ┌─────┐  │
│  TTS   │  │                                             │  │  │ ▶   │  │
│        │  │     Enter your text here...                 │  │  │Hello│  │
│  [👤]  │  │                                             │  │  │world│  │
│  Voice │  │                                             │  │  └─────┘  │
│        │  │                                   234/2000  │  │           │
│  [📡]  │  └─────────────────────────────────────────────┘  │  ┌─────┐  │
│  Stream│                                                   │  │ ▶   │  │
│        │  ┌─── Voice ────────┐  ┌─── Language ──────────┐  │  │Test │  │
│        │  │ [trump ▾]        │  │ [English ▾]           │  │  │text │  │
│        │  └──────────────────┘  └───────────────────────┘  │  └─────┘  │
│        │                                                   │           │
│        │  ┌─── Parameters ─────────────────────────────────┤  ┌─────┐  │
│        │  │ [Fast] [Balanced] [Quality]                   ││  │ ▶   │  │
│        │  │                                               ││  │More │  │
│        │  │ Sampling     ═══════●════════  25             ││  │text │  │
│        │  │ Temperature  ═══════════●════  1.0            ││  └─────┘  │
│        │  └───────────────────────────────────────────────┘│           │
│        │                                                   │           │
│  ────  │         ┌──────────────────────────┐              │           │
│  [⚙️]  │         │   ▶  Generate Speech     │              │           │
│  [◐◑]  │         └──────────────────────────┘              │           │
│        │                    ⌘ + Enter                      │           │
└────────┴───────────────────────────────────────────────────┴───────────┘
```

---

## Implementation Priority

### Must Have (MVP)
1. Dark theme foundation
2. New layout structure
3. Redesigned text input
4. Visual voice selector
5. Organized parameters
6. Custom audio player
7. History panel

### Should Have
1. Keyboard shortcuts
2. Animations
3. Parameter presets
4. Voice upload modal
5. Theme toggle

### Nice to Have
1. Waveform visualization
2. Voice library page
3. Compare mode
4. Export features
5. Audio recording in browser

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Foundation + Layout | Dark theme, new structure |
| 2 | TTS Workspace | Input, voice, params, generate |
| 3 | Audio + Voice | Player, history, upload |
| 4 | Polish + Testing | Animations, a11y, tests |

**Total Estimated Effort**: 21 days

---

## Next Steps

1. Review and approve this plan
2. Create GitHub issues for each checkpoint
3. Set up feature branch
4. Begin Phase 1 implementation

