/**
 * Design tokens shared across audiobook components.
 * Glass morphism styles, buttons, inputs for the dark-mode UI.
 */

export const glass = 'backdrop-blur-xl bg-white/[0.03] border border-white/[0.06]';
export const glassHover = 'hover:bg-white/[0.06] hover:border-white/[0.1]';
export const cardStyle = `${glass} rounded-2xl`;
export const inputStyle = 'w-full bg-white/[0.04] border border-white/[0.08] rounded-xl px-4 py-2.5 text-white/90 placeholder:text-white/20 focus:border-amber-500/40 focus:ring-1 focus:ring-amber-500/20 focus:outline-none transition-all';
export const selectStyle = 'bg-white/[0.04] border border-white/[0.08] rounded-xl px-3 py-2 text-sm text-white/80 focus:border-amber-500/40 focus:outline-none transition-all';
export const btnPrimary = 'px-5 py-2.5 rounded-xl text-sm font-semibold bg-gradient-to-r from-amber-500 to-orange-600 text-white shadow-lg shadow-amber-500/20 hover:shadow-amber-500/30 hover:from-amber-400 hover:to-orange-500 transition-all active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none';
export const btnGhost = `px-4 py-2 rounded-xl text-sm font-medium ${glass} ${glassHover} text-white/70 hover:text-white transition-all`;
