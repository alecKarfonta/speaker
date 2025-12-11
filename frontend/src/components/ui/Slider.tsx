import React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';
import { motion } from 'framer-motion';
import { cn } from '../../lib/utils';

interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  description?: string;
  formatValue?: (value: number) => string;
  className?: string;
}

const Slider: React.FC<SliderProps> = ({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  label,
  description,
  formatValue = (v) => v.toString(),
  className,
}) => {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn('space-y-3', className)}>
      {label && (
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-text-secondary">
            {label}
          </label>
          <motion.span 
            key={value}
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-sm font-mono text-accent tabular-nums bg-accent/10 px-2 py-0.5 rounded-md"
          >
            {formatValue(value)}
          </motion.span>
        </div>
      )}
      
      <SliderPrimitive.Root
        className="relative flex items-center select-none touch-none w-full h-7 cursor-pointer group"
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
      >
        <SliderPrimitive.Track className="relative grow rounded-full h-2 overflow-hidden bg-white/5">
          {/* Background glow */}
          <div 
            className="absolute h-full transition-all duration-300"
            style={{ 
              width: `${percentage}%`,
              background: 'linear-gradient(90deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2))',
              filter: 'blur(8px)',
            }}
          />
          
          {/* Actual fill */}
          <SliderPrimitive.Range 
            className="absolute h-full rounded-full transition-all"
            style={{
              background: `linear-gradient(90deg, 
                hsl(239, 84%, 67%) 0%, 
                hsl(262, 83%, 58%) 50%,
                hsl(280, 87%, 65%) 100%
              )`,
            }}
          />
        </SliderPrimitive.Track>
        
        <SliderPrimitive.Thumb asChild>
          <motion.div
            whileHover={{ scale: 1.2 }}
            whileTap={{ scale: 0.9 }}
            className={cn(
              'block w-5 h-5 rounded-full cursor-grab active:cursor-grabbing',
              'bg-white shadow-xl',
              'focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-background',
              'transition-shadow duration-200'
            )}
            style={{
              boxShadow: '0 0 0 4px rgba(99, 102, 241, 0.2), 0 4px 12px rgba(0, 0, 0, 0.4)',
            }}
          />
        </SliderPrimitive.Thumb>
      </SliderPrimitive.Root>
      
      {description && (
        <p className="text-xs text-text-tertiary">{description}</p>
      )}
    </div>
  );
};

export default Slider;
