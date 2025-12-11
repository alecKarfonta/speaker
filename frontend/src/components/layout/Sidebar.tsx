import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Mic, 
  Radio, 
  Users, 
  Settings, 
  ChevronLeft, 
  ChevronRight,
  Waves 
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { useTTSStore } from '../../stores/ttsStore';

interface NavItem {
  path: string;
  label: string;
  icon: React.ElementType;
}

const navItems: NavItem[] = [
  { path: '/', label: 'TTS Generator', icon: Mic },
  { path: '/stream', label: 'Live Stream', icon: Radio },
  { path: '/voices', label: 'Voice Library', icon: Users },
];

const Sidebar: React.FC = () => {
  const location = useLocation();
  const { sidebarCollapsed, setSidebarCollapsed } = useTTSStore();

  return (
    <motion.aside
      initial={false}
      animate={{ width: sidebarCollapsed ? 64 : 200 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      className="sidebar shrink-0"
    >
      {/* Logo */}
      <div className="p-4 border-b border-border">
        <Link to="/" className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
            <Waves className="w-5 h-5 text-white" />
          </div>
          {!sidebarCollapsed && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="font-semibold text-text-primary"
            >
              Speaker
            </motion.span>
          )}
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          
          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                'sidebar-nav-item',
                isActive && 'sidebar-nav-item-active'
              )}
              title={sidebarCollapsed ? item.label : undefined}
            >
              <Icon className="w-5 h-5 shrink-0" />
              {!sidebarCollapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-sm font-medium truncate"
                >
                  {item.label}
                </motion.span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="p-3 border-t border-border space-y-1">
        <Link
          to="/settings"
          className="sidebar-nav-item"
          title={sidebarCollapsed ? 'Settings' : undefined}
        >
          <Settings className="w-5 h-5 shrink-0" />
          {!sidebarCollapsed && (
            <span className="text-sm font-medium">Settings</span>
          )}
        </Link>

        {/* Collapse toggle */}
        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="sidebar-nav-item w-full"
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {sidebarCollapsed ? (
            <ChevronRight className="w-5 h-5 shrink-0" />
          ) : (
            <>
              <ChevronLeft className="w-5 h-5 shrink-0" />
              <span className="text-sm font-medium">Collapse</span>
            </>
          )}
        </button>
      </div>
    </motion.aside>
  );
};

export default Sidebar;

