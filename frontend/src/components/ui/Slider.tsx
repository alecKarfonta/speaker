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
  return (
    <div className={cn('space-y-2', className)}>
      {label && (
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-text-secondary">
            {label}
          </label>
          <span className="text-sm font-mono text-accent">
            {formatValue(value)}
          </span>
        </div>
      )}
      
      <SliderPrimitive.Root
        className="relative flex items-center select-none touch-none w-full h-5"
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
      >
        <SliderPrimitive.Track className="bg-bg-tertiary relative grow rounded-full h-1.5">
          <SliderPrimitive.Range className="absolute bg-accent rounded-full h-full" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          className={cn(
            'block w-4 h-4 bg-white rounded-full shadow-md',
            'border-2 border-accent',
            'focus:outline-none focus:ring-2 focus:ring-accent/50',
            'hover:scale-110 transition-transform cursor-pointer'
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

