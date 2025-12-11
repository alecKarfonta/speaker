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
  Sparkles
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { useTTSStore } from '../../stores/ttsStore';

interface NavItem {
  path: string;
  label: string;
  icon: React.ElementType;
  color: string;
}

const navItems: NavItem[] = [
  { path: '/', label: 'TTS Generator', icon: Mic, color: 'text-accent' },
  { path: '/stream', label: 'Live Stream', icon: Radio, color: 'text-green-400' },
  { path: '/voices', label: 'Voice Library', icon: Users, color: 'text-purple-400' },
];

const Sidebar: React.FC = () => {
  const location = useLocation();
  const { sidebarCollapsed, setSidebarCollapsed } = useTTSStore();

  return (
    <motion.aside
      initial={false}
      animate={{ width: sidebarCollapsed ? 72 : 220 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      className="h-screen bg-bg-secondary border-r border-border flex flex-col shrink-0 relative"
    >
      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-accent/5 via-transparent to-purple-500/5 pointer-events-none" />
      
      {/* Logo */}
      <div className="relative p-4 border-b border-border">
        <Link to="/" className="flex items-center gap-3">
          <motion.div 
            whileHover={{ scale: 1.05 }}
            className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-blue-600 flex items-center justify-center shadow-lg shadow-accent/20"
          >
            <Waves className="w-5 h-5 text-white" />
          </motion.div>
          <AnimatePresence mode="wait">
            {!sidebarCollapsed && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.15 }}
              >
                <span className="font-bold text-lg text-text-primary">Speaker</span>
                <div className="flex items-center gap-1">
                  <Sparkles className="w-3 h-3 text-accent" />
                  <span className="text-xxs text-text-tertiary">AI Voice Studio</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="relative flex-1 p-3 space-y-1">
        {navItems.map((item, index) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          
          return (
            <motion.div
              key={item.path}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <Link
                to={item.path}
                className={cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-xl',
                  'transition-all duration-200 group relative',
                  isActive 
                    ? 'bg-gradient-to-r from-accent/20 to-accent/5 text-accent' 
                    : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
                )}
                title={sidebarCollapsed ? item.label : undefined}
              >
                {/* Active indicator */}
                {isActive && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-accent rounded-r-full"
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  />
                )}
                
                <div className={cn(
                  'w-9 h-9 rounded-lg flex items-center justify-center shrink-0 transition-colors',
                  isActive 
                    ? 'bg-accent/20' 
                    : 'bg-bg-tertiary group-hover:bg-bg-hover'
                )}>
                  <Icon className={cn(
                    'w-5 h-5',
                    isActive ? item.color : 'text-text-secondary group-hover:text-text-primary'
                  )} />
                </div>
                
                <AnimatePresence mode="wait">
                  {!sidebarCollapsed && (
                    <motion.span
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                      transition={{ duration: 0.15 }}
                      className="text-sm font-medium truncate"
                    >
                      {item.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </Link>
            </motion.div>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="relative p-3 border-t border-border space-y-1">
        {/* Settings */}
        <Link
          to="/settings"
          className={cn(
            'flex items-center gap-3 px-3 py-2.5 rounded-xl',
            'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary',
            'transition-all duration-200 group'
          )}
          title={sidebarCollapsed ? 'Settings' : undefined}
        >
          <div className="w-9 h-9 rounded-lg bg-bg-tertiary group-hover:bg-bg-hover flex items-center justify-center shrink-0 transition-colors">
            <Settings className="w-5 h-5" />
          </div>
          <AnimatePresence mode="wait">
            {!sidebarCollapsed && (
              <motion.span
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="text-sm font-medium"
              >
                Settings
              </motion.span>
            )}
          </AnimatePresence>
        </Link>

        {/* Collapse toggle */}
        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className={cn(
            'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl',
            'text-text-tertiary hover:text-text-secondary hover:bg-bg-tertiary',
            'transition-all duration-200 group'
          )}
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <div className="w-9 h-9 rounded-lg bg-bg-tertiary group-hover:bg-bg-hover flex items-center justify-center shrink-0 transition-colors">
            {sidebarCollapsed ? (
              <ChevronRight className="w-5 h-5" />
            ) : (
              <ChevronLeft className="w-5 h-5" />
            )}
          </div>
          <AnimatePresence mode="wait">
            {!sidebarCollapsed && (
              <motion.span
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="text-sm font-medium"
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
