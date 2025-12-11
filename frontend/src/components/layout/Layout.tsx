import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Sidebar from './Sidebar';
import { cn } from '../../lib/utils';

interface LayoutProps {
  children: React.ReactNode;
  rightPanel?: React.ReactNode;
  showRightPanel?: boolean;
}

const Layout: React.FC<LayoutProps> = ({ 
  children, 
  rightPanel,
  showRightPanel = true 
}) => {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Main workspace */}
        <main className="flex-1 overflow-y-auto">
          <div className="h-full">
            {children}
          </div>
        </main>

        {/* Right panel (History, etc.) */}
        <AnimatePresence mode="wait">
          {showRightPanel && rightPanel && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 320, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2, ease: 'easeInOut' }}
              className={cn(
                'border-l border-border bg-bg-secondary',
                'overflow-hidden shrink-0'
              )}
            >
              <div className="w-80 h-full overflow-y-auto">
                {rightPanel}
              </div>
            </motion.aside>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Layout;

