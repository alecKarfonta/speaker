import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import { cn } from '../../lib/utils';

export interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  loading?: boolean;
  children?: React.ReactNode;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    className, 
    variant = 'primary', 
    size = 'md', 
    loading = false,
    disabled,
    children,
    ...props 
  }, ref) => {
    const variants = {
      primary: 'bg-accent hover:bg-accent-hover text-white focus:ring-accent',
      secondary: 'bg-bg-tertiary hover:bg-bg-hover text-text-primary border border-border focus:ring-border',
      ghost: 'bg-transparent hover:bg-bg-tertiary text-text-secondary hover:text-text-primary focus:ring-border',
      danger: 'bg-error hover:bg-error/90 text-white focus:ring-error',
    };

    const sizes = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2.5 text-sm',
      lg: 'px-6 py-3 text-base',
      icon: 'p-2.5',
    };

    return (
      <motion.button
        ref={ref}
        whileTap={{ scale: 0.98 }}
        className={cn(
          'inline-flex items-center justify-center gap-2 font-medium',
          'rounded-lg transition-colors duration-200',
          'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background',
          'disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none',
          variants[variant],
          sizes[size],
          className
        )}
        disabled={disabled || loading}
        {...props}
      >
        {loading && (
          <Loader2 className="w-4 h-4 animate-spin" />
        )}
        {children}
      </motion.button>
    );
  }
);

Button.displayName = 'Button';

export default Button;

