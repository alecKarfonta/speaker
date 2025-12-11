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
    <div className="flex h-screen overflow-hidden bg-bg-primary">
      {/* Global background effects */}
      <div className="fixed inset-0 pointer-events-none">
        {/* Noise texture */}
        <div 
          className="absolute inset-0 opacity-[0.015]"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
          }}
        />
        
        {/* Gradient orbs */}
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] rounded-full bg-accent/5 blur-[120px]" />
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] rounded-full bg-purple-500/5 blur-[100px]" />
      </div>

      {/* Sidebar */}
      <Sidebar />

      {/* Main content area */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Main workspace */}
        <main className="flex-1 overflow-y-auto relative">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
            className="h-full"
          >
            {children}
          </motion.div>
        </main>

        {/* Right panel divider with gradient */}
        <AnimatePresence mode="wait">
          {showRightPanel && rightPanel && (
            <>
              <div className="w-px bg-gradient-to-b from-transparent via-white/5 to-transparent" />
              <motion.aside
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 340, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                className="overflow-hidden shrink-0 relative"
              >
                {/* Panel gradient border effect */}
                <div className="absolute left-0 top-0 bottom-0 w-px bg-gradient-to-b from-accent/30 via-purple-500/20 to-transparent" />
                
                <div className="w-[340px] h-full overflow-y-auto bg-bg-secondary/50 backdrop-blur-xl">
                  {rightPanel}
                </div>
              </motion.aside>
            </>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Layout;
