import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, FileText, Image, Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react';
import { useAudiobookStore } from '../../stores/audiobookStore';
import type { QueueStatus } from '../../services/audiobookApi';

// ── Queue type config ──────────────────────────────────────────────────────
const QUEUES = [
    {
        key: 'tts' as const,
        label: 'Audio',
        icon: Mic,
        color: 'amber',
        activeBg: 'bg-amber-500/20',
        activeText: 'text-amber-300',
        activeBorder: 'border-amber-500/30',
        barColor: 'bg-amber-500',
    },
    {
        key: 'prompt' as const,
        label: 'Prompt',
        icon: FileText,
        color: 'cyan',
        activeBg: 'bg-cyan-500/20',
        activeText: 'text-cyan-300',
        activeBorder: 'border-cyan-500/30',
        barColor: 'bg-cyan-400',
    },
    {
        key: 'visual' as const,
        label: 'Visual',
        icon: Image,
        color: 'emerald',
        activeBg: 'bg-emerald-500/20',
        activeText: 'text-emerald-300',
        activeBorder: 'border-emerald-500/30',
        barColor: 'bg-emerald-400',
    },
] as const;

// ── Helpers ────────────────────────────────────────────────────────────────
function shortId(id: string): string {
    return id?.slice(0, 8) ?? '—';
}

// ── Queue Row ──────────────────────────────────────────────────────────────
const QueueRow: React.FC<{
    cfg: typeof QUEUES[number];
    data: QueueStatus['tts'];
}> = ({ cfg, data }) => {
    const { icon: Icon } = cfg;
    const isActive = !!data.active;
    const totalPending = data.queued + (isActive ? 1 : 0);

    return (
        <div className={`rounded-xl p-3 border transition-all duration-300
            ${isActive
                ? `${cfg.activeBg} ${cfg.activeBorder}`
                : 'bg-white/[0.02] border-white/[0.05]'
            }`}>
            {/* Header row */}
            <div className="flex items-center gap-2 mb-2">
                <div className={`w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0
                    ${isActive ? cfg.activeBg : 'bg-white/[0.04]'}`}>
                    {isActive
                        ? <Loader2 size={11} className={`${cfg.activeText} animate-spin`} />
                        : <Icon size={11} className="text-white/25" />
                    }
                </div>
                <span className={`text-[11px] font-semibold tracking-wide flex-1
                    ${isActive ? cfg.activeText : 'text-white/35'}`}>
                    {cfg.label}
                </span>
                {totalPending > 0 && (
                    <span className={`text-[10px] font-bold tabular-nums px-1.5 py-0.5 rounded-full
                        ${isActive ? `${cfg.activeBg} ${cfg.activeText}` : 'bg-white/[0.06] text-white/30'}`}>
                        {totalPending}
                    </span>
                )}
            </div>

            {/* Active job */}
            <AnimatePresence>
                {isActive && data.active && (
                    <motion.div
                        key="active"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="overflow-hidden"
                    >
                        <div className="flex items-center gap-1.5 mb-1.5">
                            <div className="relative w-2 h-2 flex-shrink-0">
                                <span className={`absolute inset-0 rounded-full ${cfg.barColor} animate-ping opacity-60`} />
                                <span className={`absolute inset-0 rounded-full ${cfg.barColor}`} />
                            </div>
                            <span className="text-[10px] text-white/50 font-mono truncate">
                                seg {shortId(data.active.segment_id)}
                            </span>
                        </div>
                        {/* Animated progress bar */}
                        <div className="h-0.5 rounded-full bg-white/[0.06] overflow-hidden">
                            <motion.div
                                className={`h-full ${cfg.barColor} rounded-full`}
                                animate={{ x: ['-100%', '100%'] }}
                                transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
                            />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Queued count */}
            {data.queued > 0 && (
                <div className="flex items-center gap-1 mt-1.5">
                    <Clock size={9} className="text-white/20" />
                    <span className="text-[10px] text-white/25">{data.queued} waiting</span>
                </div>
            )}

            {/* Idle */}
            {!isActive && data.queued === 0 && (
                <div className="flex items-center gap-1">
                    <CheckCircle2 size={9} className="text-white/15" />
                    <span className="text-[10px] text-white/20">Idle</span>
                </div>
            )}
        </div>
    );
};

// ── Recent history ─────────────────────────────────────────────────────────
const RecentLog: React.FC<{ recent: QueueStatus['recent'] }> = ({ recent }) => {
    if (!recent || recent.length === 0) return null;
    const last = [...recent].reverse().slice(0, 6);
    return (
        <div className="mt-3">
            <p className="text-[9px] uppercase tracking-widest text-white/15 mb-1.5 px-1">Recent</p>
            <div className="space-y-0.5">
                {last.map((item, i) => (
                    <div key={i} className="flex items-center gap-1.5 px-1 py-0.5">
                        {item.status === 'done'
                            ? <CheckCircle2 size={9} className="text-emerald-400/60 flex-shrink-0" />
                            : <XCircle size={9} className="text-red-400/60 flex-shrink-0" />
                        }
                        <span className="text-[9px] font-mono text-white/20 truncate">
                            {item.type} · {shortId(item.segment_id)}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// ── Main Panel ─────────────────────────────────────────────────────────────
const QueuePanel: React.FC = () => {
    const { queueStatus, fetchQueueStatus } = useAudiobookStore();
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // Poll every 2s
    useEffect(() => {
        fetchQueueStatus(); // immediate fetch
        intervalRef.current = setInterval(fetchQueueStatus, 2000);
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [fetchQueueStatus]);

    const anyBusy = queueStatus && (
        queueStatus.tts.queued > 0 || queueStatus.tts.active ||
        queueStatus.prompt.queued > 0 || queueStatus.prompt.active ||
        queueStatus.visual.queued > 0 || queueStatus.visual.active
    );

    const totalJobs = queueStatus
        ? (queueStatus.tts.queued + (queueStatus.tts.active ? 1 : 0))
        + (queueStatus.prompt.queued + (queueStatus.prompt.active ? 1 : 0))
        + (queueStatus.visual.queued + (queueStatus.visual.active ? 1 : 0))
        : 0;

    return (
        <aside className="flex flex-col gap-2 pt-2">
            {/* Panel header */}
            <div className="flex items-center gap-2 px-1 mb-1">
                <span className="text-[10px] uppercase tracking-widest text-white/20 font-semibold">Queue</span>
                {anyBusy && (
                    <motion.span
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="text-[10px] font-bold text-amber-400/80 tabular-nums"
                    >
                        {totalJobs} active
                    </motion.span>
                )}
            </div>

            {/* Queue rows */}
            <div className="space-y-2">
                {QUEUES.map((cfg) => (
                    <QueueRow
                        key={cfg.key}
                        cfg={cfg}
                        data={queueStatus?.[cfg.key] ?? { queued: 0, active: null }}
                    />
                ))}
            </div>

            {/* Recent history */}
            {queueStatus && <RecentLog recent={queueStatus.recent} />}
        </aside>
    );
};

export default QueuePanel;
