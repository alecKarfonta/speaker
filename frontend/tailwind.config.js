/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Background colors
        background: 'var(--bg-primary)',
        'bg-primary': 'var(--bg-primary)',
        'bg-secondary': 'var(--bg-secondary)',
        'bg-tertiary': 'var(--bg-tertiary)',
        'bg-hover': 'var(--bg-hover)',
        
        // Foreground/text colors
        foreground: 'var(--text-primary)',
        'text-primary': 'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        'text-tertiary': 'var(--text-tertiary)',
        
        // Border colors
        border: 'var(--border-default)',
        'border-subtle': 'var(--border-subtle)',
        'border-strong': 'var(--border-strong)',
        
        // Card
        card: 'var(--bg-secondary)',
        'card-foreground': 'var(--text-primary)',
        
        // Accent
        accent: {
          DEFAULT: 'var(--accent-primary)',
          hover: 'var(--accent-hover)',
          glow: 'var(--accent-glow)',
        },
        
        // Status colors
        success: 'var(--success)',
        warning: 'var(--warning)',
        error: 'var(--error)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        'xxs': '0.625rem',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '100': '25rem',
        '120': '30rem',
      },
      borderRadius: {
        'xl': '0.875rem',
        '2xl': '1rem',
      },
      boxShadow: {
        'glow': '0 0 20px var(--accent-glow)',
        'glow-sm': '0 0 10px var(--accent-glow)',
        'inner-glow': 'inset 0 0 20px var(--accent-glow)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'slide-in-right': 'slideInRight 0.3s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px var(--accent-glow)' },
          '100%': { boxShadow: '0 0 20px var(--accent-glow), 0 0 40px var(--accent-glow)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideInRight: {
          '0%': { opacity: '0', transform: 'translateX(20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
      transitionDuration: {
        '250': '250ms',
      },
    },
  },
  plugins: [],
}
