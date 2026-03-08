"""
HuggingFace Spaces Debug Helper
================================
Adds real-time error logging and debug endpoints for easier debugging
without rebuilding Docker every time.
"""

import sys
import io
import traceback
import logging
from datetime import datetime
from pathlib import Path
from collections import deque
from threading import Lock

class DebugLogger:
    """
    Captures all errors and prints them both to console and stores them
    for viewing via web dashboard /debug endpoint.
    """
    
    def __init__(self, max_errors=100, log_file="/home/user/app/logs/debug.log"):
        self.errors = deque(maxlen=max_errors)
        self.lock = Lock()
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up file logging
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(self.file_handler)
        
    def log_error(self, error_type, error_msg, tb_str=None):
        """Log an error with timestamp and traceback"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        error_entry = {
            'timestamp': timestamp,
            'type': error_type,
            'message': str(error_msg),
            'traceback': tb_str or ''
        }
        
        with self.lock:
            self.errors.append(error_entry)
        
        # Print to console (HuggingFace captures this)
        print(f"\n{'='*80}")
        print(f"🔴 ERROR CAPTURED: {timestamp}")
        print(f"{'='*80}")
        print(f"Type: {error_type}")
        print(f"Message: {error_msg}")
        if tb_str:
            print(f"\nTraceback:")
            print(tb_str)
        print(f"{'='*80}\n")
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR: {timestamp}\n")
            f.write(f"Type: {error_type}\n")
            f.write(f"Message: {error_msg}\n")
            if tb_str:
                f.write(f"Traceback:\n{tb_str}\n")
            f.write(f"{'='*80}\n")
    
    def get_recent_errors(self, count=20):
        """Get the most recent N errors"""
        with self.lock:
            return list(self.errors)[-count:]
    
    def get_all_errors(self):
        """Get all stored errors"""
        with self.lock:
            return list(self.errors)
    
    def clear_errors(self):
        """Clear all stored errors"""
        with self.lock:
            self.errors.clear()
        print("✅ Debug error log cleared")
    
    def get_log_file_contents(self):
        """Read the debug log file"""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    return f.read()
            return "No debug log file found"
        except Exception as e:
            return f"Error reading log file: {e}"

# Global debug logger instance
debug_logger = DebugLogger()


def capture_exception(func):
    """
    Decorator to capture and log exceptions from any function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            tb_str = traceback.format_exc()
            
            debug_logger.log_error(error_type, error_msg, tb_str)
            raise  # Re-raise the exception
    
    return wrapper


def add_debug_routes(app):
    """
    Add debug routes to Flask app
    
    Usage:
        from debug_helper import add_debug_routes
        add_debug_routes(app)
    """
    
    @app.route('/debug')
    def debug_page():
        """HTML page showing recent errors"""
        errors = debug_logger.get_all_errors()
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>K1RL QUASAR Debug Dashboard</title>
    <style>
        body {
            background: #0a0e27;
            color: #00ff9d;
            font-family: 'Courier New', monospace;
            padding: 20px;
            margin: 0;
        }
        .header {
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #00ff9d;
            margin-bottom: 20px;
        }
        .error-card {
            background: #1a1f3a;
            border: 1px solid #00ff9d;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 0 10px rgba(0, 255, 157, 0.2);
        }
        .error-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #00ff9d;
        }
        .error-type {
            color: #ff3366;
            font-weight: bold;
        }
        .error-timestamp {
            color: #888;
            font-size: 0.9em;
        }
        .error-message {
            color: #ffaa00;
            margin: 10px 0;
        }
        .error-traceback {
            background: #000;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
            font-size: 0.85em;
            color: #00ff9d;
            white-space: pre;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background: #00ff9d;
            color: #0a0e27;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin: 0 5px;
        }
        button:hover {
            background: #00cc7d;
        }
        .no-errors {
            text-align: center;
            padding: 40px;
            color: #00ff9d;
            font-size: 1.2em;
        }
        .stats {
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            background: #1a1f3a;
            border-radius: 5px;
        }
    </style>
    <script>
        function refresh() {
            location.reload();
        }
        function clearErrors() {
            fetch('/debug/clear', {method: 'POST'})
                .then(() => location.reload());
        }
        // Auto-refresh every 10 seconds
        setTimeout(refresh, 10000);
    </script>
</head>
<body>
    <div class="header">
        <h1>🔴 K1RL QUASAR Debug Dashboard</h1>
        <p>Real-time error monitoring for HuggingFace Spaces</p>
    </div>
    
    <div class="stats">
        <strong>Total Errors Captured:</strong> """ + str(len(errors)) + """<br>
        <small>Auto-refreshes every 10 seconds</small>
    </div>
    
    <div class="controls">
        <button onclick="refresh()">🔄 Refresh Now</button>
        <button onclick="clearErrors()">🗑️ Clear All Errors</button>
        <button onclick="window.location.href='/debug/logs'">📄 View Full Log File</button>
        <button onclick="window.location.href='/'">🏠 Back to Dashboard</button>
    </div>
"""
        
        if not errors:
            html += """
    <div class="no-errors">
        ✅ No errors captured yet<br>
        <small>Errors will appear here when they occur</small>
    </div>
"""
        else:
            # Show errors in reverse chronological order (newest first)
            for error in reversed(errors):
                html += f"""
    <div class="error-card">
        <div class="error-header">
            <span class="error-type">{error['type']}</span>
            <span class="error-timestamp">{error['timestamp']}</span>
        </div>
        <div class="error-message">{error['message']}</div>
"""
                if error['traceback']:
                    html += f"""
        <div class="error-traceback">{error['traceback']}</div>
"""
                html += """
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    @app.route('/debug/clear', methods=['POST'])
    def clear_errors():
        """Clear all stored errors"""
        debug_logger.clear_errors()
        return {'status': 'success', 'message': 'Errors cleared'}
    
    @app.route('/debug/logs')
    def view_log_file():
        """View the raw debug log file"""
        log_contents = debug_logger.get_log_file_contents()
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Debug Log File</title>
    <style>
        body {{
            background: #0a0e27;
            color: #00ff9d;
            font-family: 'Courier New', monospace;
            padding: 20px;
        }}
        pre {{
            background: #000;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        button {{
            background: #00ff9d;
            color: #0a0e27;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin: 10px 5px;
        }}
    </style>
</head>
<body>
    <h1>📄 Debug Log File</h1>
    <button onclick="window.location.href='/debug'">🔙 Back to Debug Dashboard</button>
    <button onclick="location.reload()">🔄 Refresh</button>
    <pre>{log_contents}</pre>
</body>
</html>
"""
    
    print("✅ Debug routes added: /debug, /debug/clear, /debug/logs")


def patch_quantum_meta_controller():
    """
    Monkey-patch the QuantumMetaController to use debug logging
    Call this before creating the controller instance
    """
    print("🔧 Patching QuantumMetaController for better error logging...")
    
    # This will be imported dynamically
    import types
    
    def patched_run_with_cleanup(original_func):
        def wrapper(*args, **kwargs):
            try:
                return original_func(*args, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                tb_str = traceback.format_exc()
                
                debug_logger.log_error(error_type, error_msg, tb_str)
                
                # Store in container
                if len(args) > 0 and isinstance(args[0], dict):
                    args[0]['error'] = e
                
                raise
        return wrapper
    
    print("✅ QuantumMetaController patched for debug logging")


if __name__ == "__main__":
    # Test the debug logger
    print("Testing debug logger...")
    
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        debug_logger.log_error(
            type(e).__name__,
            str(e),
            traceback.format_exc()
        )
    
    print(f"\nCaptured {len(debug_logger.get_all_errors())} errors")
    print("\nRecent errors:")
    for error in debug_logger.get_recent_errors():
        print(f"  - {error['timestamp']}: {error['type']} - {error['message']}")