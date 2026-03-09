/**
 * Small reusable UI components for the audiobook page.
 * SidebarSection, StatusBadge, ProgressBar
 */
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    ChevronRight, Clock, Loader2, Check, AlertCircle,
} from 'lucide-react';

// ============================================================
// Collapsible sidebar section
// ============================================================
export const SidebarSection: React.FC<{
    icon: React.ReactNode;
    label: string;
    status?: string;
    statusColor?: string;
    defaultOpen?: boolean;
    children: React.ReactNode;
}> = ({ icon, label, status, statusColor = 'text-white/20', defaultOpen = true, children }) => {
    const [open, setOpen] = useState(defaultOpen);
    return (
        <div className="space-y-2">
            <button
                onClick={() => setOpen(!open)}
                className="w-full flex items-center gap-2 group"
            >
                <motion.div
                    animate={{ rotate: open ? 90 : 0 }}
                    transition={{ duration: 0.12 }}
                    className="flex-shrink-0"
                >
                    <ChevronRight size={10} className="text-white/20" />
                </motion.div>
                <span className="flex items-center gap-1.5 text-[10px] font-semibold text-white/40 uppercase tracking-[0.12em]">
                    {icon} {label}
                </span>
                {status && (
                    <span className={`ml-auto text-[9px] font-medium ${statusColor} tabular-nums`}>{status}</span>
                )}
            </button>
            <AnimatePresence initial={false}>
                {open && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="overflow-hidden"
                    >
                        <div className="space-y-2 pl-4">
                            {children}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

// ============================================================
// Status badge component
// ============================================================
export const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
    const config: Record<string, { bg: string; text: string; icon: React.ReactNode; label: string }> = {
        pending: { bg: 'bg-slate-500/10 border-slate-500/20', text: 'text-slate-400', icon: <Clock size={10} />, label: 'Pending' },
        generating: { bg: 'bg-sky-500/10 border-sky-500/20', text: 'text-sky-400', icon: <Loader2 size={10} className="animate-spin" />, label: 'Generating' },
        done: { bg: 'bg-emerald-500/10 border-emerald-500/20', text: 'text-emerald-400', icon: <Check size={10} />, label: 'Done' },
        error: { bg: 'bg-red-500/10 border-red-500/20', text: 'text-red-400', icon: <AlertCircle size={10} />, label: 'Error' },
    };
    const c = config[status] || config.pending;
    return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border ${c.bg} ${c.text}`}>
            {c.icon} {c.label}
        </span>
    );
};

// ============================================================
// Progress bar
// ============================================================
export const ProgressBar: React.FC<{ progress: number; size?: 'sm' | 'md' | 'lg'; glow?: boolean }> = ({ progress, size = 'sm', glow = false }) => (
    <div className={`w-full bg-white/[0.04] rounded-full overflow-hidden ${size === 'lg' ? 'h-3' : size === 'md' ? 'h-2' : 'h-1.5'}`}>
        <motion.div
            className={`h-full rounded-full bg-gradient-to-r from-amber-500 via-orange-500 to-rose-500 ${glow ? 'shadow-[0_0_12px_rgba(251,191,36,0.4)]' : ''}`}
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(Math.round(progress * 100), 100)}%` }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
        />
    </div>
);
