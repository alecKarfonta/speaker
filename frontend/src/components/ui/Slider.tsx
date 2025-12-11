import React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';
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
          <span className="text-sm font-mono text-accent tabular-nums">
            {formatValue(value)}
          </span>
        </div>
      )}
      
      <SliderPrimitive.Root
        className="relative flex items-center select-none touch-none w-full h-6 cursor-pointer group"
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
      >
        <SliderPrimitive.Track className="bg-bg-tertiary relative grow rounded-full h-2 overflow-hidden">
          {/* Gradient fill */}
          <SliderPrimitive.Range 
            className="absolute h-full rounded-full"
            style={{
              background: `linear-gradient(90deg, var(--accent-primary), #60a5fa)`
            }}
          />
          
          {/* Glow effect on hover */}
          <div 
            className="absolute h-full bg-accent/20 blur-sm opacity-0 group-hover:opacity-100 transition-opacity"
            style={{ width: `${percentage}%` }}
          />
        </SliderPrimitive.Track>
        
        <SliderPrimitive.Thumb
          className={cn(
            'block w-5 h-5 bg-white rounded-full',
            'shadow-lg shadow-black/20',
            'border-2 border-accent',
            'focus:outline-none focus:ring-2 focus:ring-accent/50 focus:ring-offset-2 focus:ring-offset-background',
            'hover:scale-110 active:scale-105',
            'transition-transform cursor-grab active:cursor-grabbing'
          )}
        />
      </SliderPrimitive.Root>
      
      {description && (
        <p className="text-xs text-text-tertiary">{description}</p>
      )}
    </div>
  );
};

export default Slider;
