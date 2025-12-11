import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Mic, 
  Radio, 
  Users, 
  Settings, 
  ChevronLeft, 
  ChevronRight,
  Waves,
  Sparkles,
  Zap
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { useTTSStore } from '../../stores/ttsStore';

interface NavItem {
  path: string;
  label: string;
  icon: React.ElementType;
  gradient: string;
}

const navItems: NavItem[] = [
  { path: '/', label: 'Text to Speech', icon: Mic, gradient: 'from-violet-500 to-purple-500' },
  { path: '/stream', label: 'Live Stream', icon: Radio, gradient: 'from-emerald-500 to-teal-500' },
  { path: '/voices', label: 'Voice Library', icon: Users, gradient: 'from-orange-500 to-pink-500' },
];

const Sidebar: React.FC = () => {
  const location = useLocation();
  const { sidebarCollapsed, setSidebarCollapsed } = useTTSStore();

  return (
    <motion.aside
      initial={false}
      animate={{ width: sidebarCollapsed ? 80 : 260 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className="sidebar relative"
    >
      {/* Gradient line on the right */}
      <div className="absolute right-0 top-0 bottom-0 w-px bg-gradient-to-b from-accent/50 via-purple-500/30 to-transparent" />
      
      {/* Logo */}
      <div className="p-5 border-b border-white/5">
        <Link to="/" className="flex items-center gap-4">
          <motion.div 
            whileHover={{ scale: 1.05, rotate: 5 }}
            whileTap={{ scale: 0.95 }}
            className="relative"
          >
            {/* Logo glow */}
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-accent to-purple-500 blur-xl opacity-50" />
            <div className="relative w-12 h-12 rounded-2xl bg-gradient-to-br from-accent to-purple-500 flex items-center justify-center shadow-2xl">
              <Waves className="w-6 h-6 text-white" />
            </div>
          </motion.div>
          
          <AnimatePresence mode="wait">
            {!sidebarCollapsed && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
              >
                <span className="font-bold text-xl text-white">Speaker</span>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <Sparkles className="w-3 h-3 text-accent" />
                  <span className="text-xs text-text-tertiary">AI Voice Studio</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {navItems.map((item, index) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            
            return (
              <motion.div
                key={item.path}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Link
                  to={item.path}
                  className={cn(
                    'sidebar-nav-item',
                    isActive && 'sidebar-nav-item-active'
                  )}
                >
                  <motion.div 
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    className={cn(
                      'w-11 h-11 rounded-xl flex items-center justify-center shrink-0 transition-all duration-300',
                      isActive 
                        ? `bg-gradient-to-br ${item.gradient} shadow-lg` 
                        : 'bg-white/5 group-hover:bg-white/10'
                    )}
                    style={isActive ? {
                      boxShadow: '0 8px 20px rgba(99, 102, 241, 0.3)',
                    } : undefined}
                  >
                    <Icon className={cn(
                      'w-5 h-5 transition-colors',
                      isActive ? 'text-white' : 'text-text-tertiary'
                    )} />
                  </motion.div>
                  
                  <AnimatePresence mode="wait">
                    {!sidebarCollapsed && (
                      <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        className="flex-1"
                      >
                        <span className={cn(
                          'text-sm font-medium',
                          isActive ? 'text-white' : 'text-text-secondary'
                        )}>
                          {item.label}
                        </span>
                      </motion.div>
                    )}
                  </AnimatePresence>
                  
                  {isActive && !sidebarCollapsed && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="w-2 h-2 rounded-full bg-white/80"
                    />
                  )}
                </Link>
              </motion.div>
            );
          })}
        </div>
      </nav>

      {/* Status indicator */}
      <AnimatePresence>
        {!sidebarCollapsed && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="mx-4 mb-4"
          >
            <div className="p-4 rounded-2xl bg-gradient-to-br from-accent/10 to-purple-500/5 border border-white/5">
              <div className="flex items-center gap-3 mb-3">
                <div className="relative">
                  <div className="w-2 h-2 rounded-full bg-green-400" />
                  <div className="absolute inset-0 w-2 h-2 rounded-full bg-green-400 animate-ping" />
                </div>
                <span className="text-sm text-text-secondary">System Ready</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span className="text-xs text-text-tertiary">GLM-TTS Active</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Bottom section */}
      <div className="p-4 border-t border-white/5 space-y-2">
        <Link
          to="/settings"
          className="sidebar-nav-item group"
        >
          <div className="w-11 h-11 rounded-xl bg-white/5 group-hover:bg-white/10 flex items-center justify-center transition-colors">
            <Settings className="w-5 h-5 text-text-tertiary group-hover:text-text-secondary transition-colors" />
          </div>
          <AnimatePresence mode="wait">
            {!sidebarCollapsed && (
              <motion.span
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="text-sm font-medium text-text-secondary group-hover:text-text-primary"
              >
                Settings
              </motion.span>
            )}
          </AnimatePresence>
        </Link>

        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="sidebar-nav-item w-full group"
        >
          <div className="w-11 h-11 rounded-xl bg-white/5 group-hover:bg-white/10 flex items-center justify-center transition-colors">
            <motion.div
              animate={{ rotate: sidebarCollapsed ? 180 : 0 }}
              transition={{ duration: 0.3 }}
            >
              <ChevronLeft className="w-5 h-5 text-text-tertiary group-hover:text-text-secondary transition-colors" />
            </motion.div>
          </div>
          <AnimatePresence mode="wait">
            {!sidebarCollapsed && (
              <motion.span
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="text-sm font-medium text-text-secondary group-hover:text-text-primary"
              >
                Collapse
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>
    </motion.aside>
  );
};

export default Sidebar;
