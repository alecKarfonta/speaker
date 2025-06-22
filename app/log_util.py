class Colors:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    LIGHTCYAN = '\033[96m'
    LIGHTMAGENTA = '\033[95m'
    LIGHTBLUE = '\033[94m'
    LIGHTGREEN = '\033[92m'
    LIGHTYELLOW = '\033[93m'
    LIGHTRED = '\033[91m'
    DARKCYAN = '\033[36m'
    DARKMAGENTA = '\033[35m'
    DARKBLUE = '\033[34m'
    DARKGREEN = '\033[32m'
    DARKYELLOW = '\033[33m'
    DARKRED = '\033[31m'
    LIGHTGRAY = '\033[37m'
    DARKGRAY = '\033[90m'

    BOLD_CYAN = '\033[1;96m'
    BOLD_MAGENTA = '\033[1;95m'
    BOLD_BLUE = '\033[1;94m'
    BOLD_GREEN = '\033[1;92m'
    BOLD_YELLOW = '\033[1;93m'
    BOLD_RED = '\033[1;91m'
    BOLD_LIGHTGRAY = '\033[1;37m'
    BOLD_DARKGRAY = '\033[1;90m'

    UNDERLINE_CYAN = '\033[4;96m'
    UNDERLINE_MAGENTA = '\033[4;95m'
    UNDERLINE_BLUE = '\033[4;94m'
    UNDERLINE_GREEN = '\033[4;92m'
    UNDERLINE_YELLOW = '\033[4;93m'
    UNDERLINE_RED = '\033[4;91m'
    UNDERLINE_LIGHTGRAY = '\033[4;37m'
    UNDERLINE_DARKGRAY = '\033[4;90m'


import logging

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on their level."""
    
    COLORS = {
        logging.DEBUG: Colors.BOLD_GREEN,
        logging.INFO: Colors.BLUE,
        logging.WARNING: Colors.UNDERLINE_YELLOW,
        logging.ERROR: Colors.UNDERLINE_RED,
        logging.CRITICAL: Colors.BOLD + Colors.UNDERLINE_RED,
    }
    
    def __init__(self, fmt='%(levelname)s | %(module)s.%(funcName)s%(message)s', datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)

    def format(self, record):
        # Store the original values
        original_levelname = record.levelname
        original_module = record.module
        original_funcName = record.funcName
        
        # Add color to the levelname if we have a color for this level
        if record.levelno in self.COLORS:
            record.levelname = self.COLORS[record.levelno] + record.levelname + Colors.ENDC
            
        # Add default colors for module and function name
        record.module = Colors.CYAN + record.module + Colors.ENDC
        record.funcName = Colors.MAGENTA + record.funcName + Colors.ENDC
            
        # Format the record
        formatted_record = super().format(record)
        
        # Restore the original values for future formatting
        record.levelname = original_levelname
        record.module = original_module
        record.funcName = original_funcName
        
        return formatted_record
    
    def format_time(self, record, datefmt=None):
        return super().formatTime(record, datefmt)
