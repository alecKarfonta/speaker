import React from 'react';
import * as SelectPrimitive from '@radix-ui/react-select';
import { ChevronDown, Check } from 'lucide-react';
import { cn } from '../../lib/utils';

interface SelectOption {
  value: string;
  label: string;
}

interface SelectProps {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  label?: string;
  className?: string;
}

const Select: React.FC<SelectProps> = ({
  value,
  onChange,
  options,
  placeholder = 'Select...',
  label,
  className,
}) => {
  return (
    <div className={cn('space-y-1.5', className)}>
      {label && (
        <label className="text-sm font-medium text-text-secondary">
          {label}
        </label>
      )}
      
      <SelectPrimitive.Root value={value} onValueChange={onChange}>
        <SelectPrimitive.Trigger
          className={cn(
            'flex items-center justify-between w-full',
            'bg-bg-secondary border border-border rounded-lg',
            'px-3 py-2.5 text-sm text-text-primary',
            'focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent',
            'transition-all duration-200 cursor-pointer',
            'data-[placeholder]:text-text-tertiary'
          )}
        >
          <SelectPrimitive.Value placeholder={placeholder} />
          <SelectPrimitive.Icon>
            <ChevronDown className="w-4 h-4 text-text-tertiary" />
          </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>

        <SelectPrimitive.Portal>
          <SelectPrimitive.Content
            className={cn(
              'bg-bg-secondary border border-border rounded-lg shadow-xl',
              'overflow-hidden z-50',
              'animate-fade-in'
            )}
            position="popper"
            sideOffset={4}
          >
            <SelectPrimitive.Viewport className="p-1">
              {options.map((option) => (
                <SelectPrimitive.Item
                  key={option.value}
                  value={option.value}
                  className={cn(
                    'relative flex items-center px-3 py-2 pr-8 text-sm rounded-md',
                    'text-text-secondary cursor-pointer select-none',
                    'outline-none',
                    'data-[highlighted]:bg-bg-hover data-[highlighted]:text-text-primary',
                    'data-[state=checked]:text-accent'
                  )}
                >
                  <SelectPrimitive.ItemText>
                    {option.label}
                  </SelectPrimitive.ItemText>
                  <SelectPrimitive.ItemIndicator className="absolute right-2">
                    <Check className="w-4 h-4" />
                  </SelectPrimitive.ItemIndicator>
                </SelectPrimitive.Item>
              ))}
            </SelectPrimitive.Viewport>
          </SelectPrimitive.Content>
        </SelectPrimitive.Portal>
      </SelectPrimitive.Root>
    </div>
  );
};

export default Select;

