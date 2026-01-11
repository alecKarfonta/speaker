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
    BOLD_WHITE = '\033[1;97m'

    UNDERLINE_CYAN = '\033[4;96m'
    UNDERLINE_MAGENTA = '\033[4;95m'
    UNDERLINE_BLUE = '\033[4;94m'
    UNDERLINE_GREEN = '\033[4;92m'
    UNDERLINE_YELLOW = '\033[4;93m'
    UNDERLINE_RED = '\033[4;91m'
    UNDERLINE_LIGHTGRAY = '\033[4;37m'
    UNDERLINE_DARKGRAY = '\033[4;90m'


import logging
import re

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on their level.
    
    Also highlights key values in the message:
    - Numbers (int/float with units like ms, s, GB, tokens, etc.)
    - Quoted strings
    - key=value pairs
    """
    
    COLORS = {
        logging.DEBUG: Colors.BOLD_GREEN,
        logging.INFO: Colors.BLUE,
        logging.WARNING: Colors.UNDERLINE_YELLOW,
        logging.ERROR: Colors.UNDERLINE_RED,
        logging.CRITICAL: Colors.BOLD + Colors.UNDERLINE_RED,
    }
    
    # Patterns for highlighting values
    # Matches: numbers with optional decimals and units (e.g., "123.45ms", "10GB", "2.5x")
    NUMBER_PATTERN = re.compile(
        r'\b(\d+\.?\d*)\s*(ms|s|sec|min|hr|MB|GB|TB|KB|B|tok/s|toks/s|tokens?|x|%|WPM)?\b',
        re.IGNORECASE
    )
    # Matches: key=value pairs (e.g., "dtype=fp16", "gpu_util=0.3")
    KEYVAL_PATTERN = re.compile(r'(\w+)=([^\s,\)]+)')
    # Matches: quoted strings
    QUOTED_PATTERN = re.compile(r"(['\"])(.+?)\1")
    
    def __init__(self, fmt='%(levelname)s | %(module)s.%(funcName)s%(message)s', datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)

    def _highlight_values(self, message: str) -> str:
        """Highlight key values in the message with bold/colors."""
        # Highlight key=value pairs first (bold magenta for key, bold cyan for value)
        def keyval_repl(m):
            key, val = m.group(1), m.group(2)
            return f"{Colors.BOLD_MAGENTA}{key}{Colors.ENDC}={Colors.BOLD_CYAN}{val}{Colors.ENDC}"
        message = self.KEYVAL_PATTERN.sub(keyval_repl, message)
        
        # Highlight standalone numbers with units (bold yellow)
        def number_repl(m):
            num, unit = m.group(1), m.group(2) or ''
            return f"{Colors.BOLD_YELLOW}{num}{unit}{Colors.ENDC}"
        message = self.NUMBER_PATTERN.sub(number_repl, message)
        
        # Highlight quoted strings (bold green)
        def quoted_repl(m):
            quote, content = m.group(1), m.group(2)
            return f"{quote}{Colors.BOLD_GREEN}{content}{Colors.ENDC}{quote}"
        message = self.QUOTED_PATTERN.sub(quoted_repl, message)
        
        return message

    def format(self, record):
        # Store the original values
        original_levelname = record.levelname
        original_module = record.module
        original_funcName = record.funcName
        original_msg = record.msg
        
        # Add color to the levelname if we have a color for this level
        if record.levelno in self.COLORS:
            record.levelname = self.COLORS[record.levelno] + record.levelname + Colors.ENDC
            
        # Add default colors for module and function name
        record.module = Colors.CYAN + record.module + Colors.ENDC
        record.funcName = Colors.MAGENTA + record.funcName + Colors.ENDC
        
        # Highlight key values in the message
        if isinstance(record.msg, str):
            record.msg = self._highlight_values(record.msg)
            
        # Format the record
        formatted_record = super().format(record)
        
        # Restore the original values for future formatting
        record.levelname = original_levelname
        record.module = original_module
        record.funcName = original_funcName
        record.msg = original_msg
        
        return formatted_record
    
    def format_time(self, record, datefmt=None):
        return super().formatTime(record, datefmt)

