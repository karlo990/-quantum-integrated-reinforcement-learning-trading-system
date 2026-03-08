#!/usr/bin/env python3
"""
K1RL QUANT - Real-Time Accuracy Tracker
HuggingFace Spaces Edition - Integrated Service
Note: This service is integrated into the main dashboard_service.py for HF Spaces.
This file serves as a standalone backup/development version.
"""
import re
import json
import time
import os
from datetime import datetime
from collections import deque
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# ✅ FIXED: Container-safe paths for HuggingFace Spaces
BASE_DIR = Path('/home/user/app')
DATA_DIR = BASE_DIR / 'data'
LOG_DIR = BASE_DIR / 'logs'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

accuracy_history = deque(maxlen=1000)
accuracy_file = DATA_DIR / 'accuracy_data.json'

# Load existing data
if accuracy_file.exists():
    try:
        with open(accuracy_file, 'r') as f:
            data = json.load(f)
            accuracy_history.extend(data)
            print(f"✅ Loaded {len(accuracy_history)} accuracy readings")
    except Exception as e:
        print(f"⚠️  Could not load accuracy data: {e}")

def parse_accuracy_from_logs():
    """Parse accuracy from quasar logs (HF Spaces compatible)"""
    # ✅ FIXED: HuggingFace Spaces log path
    log_file = LOG_DIR / 'quasar_engine.log'
    last_position = 0
    
    print(f"🔍 Starting accuracy parser: {log_file}")
    
    while True:
        try:
            # Check if log file exists
            if not log_file.exists():
                print(f"⚠️  Waiting for log file: {log_file}")
                time.sleep(10)
                continue
                
            with open(log_file, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
                
                for line in new_lines:
                    # Multiple patterns for accuracy detection
                    patterns = [
                        r'🎯 Avg Accuracy: ([\d.]+)%',
                        r'AVN Accuracy: ([\d.]+)%?',
                        r'Avg Accuracy: ([\d.]+)%?',
                        r'accuracy[:\s]+([\d.]+)%?',
                        r'acc[:\s]+([\d.]+)%?'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            accuracy = float(match.group(1))
                            
                            # Sanity check - accuracy should be 0-100
                            if 0 <= accuracy <= 100:
                                timestamp = datetime.now().isoformat()
                                
                                data_point = {
                                    'timestamp': timestamp,
                                    'accuracy': accuracy,
                                    'iteration': len(accuracy_history) + 1
                                }
                                
                                accuracy_history.append(data_point)
                                print(f"📈 Accuracy update: {accuracy:.1f}%")
                                
                                # Save periodically
                                if len(accuracy_history) % 10 == 0:
                                    save_data()
                                
                                break  # Only take first match per line
            
            time.sleep(5)
        except Exception as e:
            print(f"⚠️  Log parsing error: {e}")
            time.sleep(10)

def save_data():
    """Save accuracy data to file"""
    try:
        with open(accuracy_file, 'w') as f:
            json.dump(list(accuracy_history), f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save data: {e}")

# ============================================================================
# API ENDPOINTS (Compatible with main dashboard)
# ============================================================================

@app.route('/api/accuracy/current')
def get_current_accuracy():
    """Get current accuracy reading"""
    if accuracy_history:
        return jsonify(accuracy_history[-1])
    return jsonify({'accuracy': 0, 'timestamp': None, 'iteration': 0})

@app.route('/api/accuracy/history')
def get_accuracy_history():
    """Get complete accuracy history"""
    return jsonify(list(accuracy_history))

@app.route('/api/accuracy/stats')
def get_accuracy_stats():
    """Get comprehensive accuracy statistics"""
    if not accuracy_history:
        return jsonify({
            'current': 0,
            'average': 0,
            'min': 0,
            'max': 0,
            'count': 0,
            'last_10_avg': 0,
            'trend': 'unknown'
        })
    
    accuracies = [d['accuracy'] for d in accuracy_history]
    
    # Calculate trend
    trend = 'stable'
    if len(accuracies) >= 20:
        recent_avg = sum(accuracies[-10:]) / 10
        older_avg = sum(accuracies[-20:-10]) / 10
        
        if recent_avg > older_avg * 1.02:  # 2% improvement
            trend = 'up'
        elif recent_avg < older_avg * 0.98:  # 2% decline  
            trend = 'down'
    
    return jsonify({
        'current': accuracies[-1] if accuracies else 0,
        'average': sum(accuracies) / len(accuracies),
        'min': min(accuracies),
        'max': max(accuracies),
        'count': len(accuracies),
        'last_10_avg': sum(accuracies[-10:]) / min(10, len(accuracies)),
        'trend': trend,
        'std_dev': (sum((x - sum(accuracies)/len(accuracies))**2 for x in accuracies) / len(accuracies))**0.5 if accuracies else 0,
        'latest_timestamp': accuracy_history[-1]['timestamp'] if accuracy_history else None
    })

@app.route('/api/accuracy/timerange/<int:hours>')
def get_accuracy_timerange(hours):
    """Get accuracy data for specific time range"""
    if not accuracy_history:
        return jsonify([])
    
    # Filter by timestamp (last N hours)
    cutoff_time = datetime.now().timestamp() - (hours * 3600)
    
    filtered = []
    for entry in accuracy_history:
        try:
            entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
            if entry_time >= cutoff_time:
                filtered.append(entry)
        except:
            # Include entries with invalid timestamps
            filtered.append(entry)
    
    return jsonify(filtered)

@app.route('/health')
def health():
    """Health check for HuggingFace Spaces"""
    log_file = LOG_DIR / 'quasar_engine.log'
    
    return jsonify({
        'status': 'ok',
        'service': 'accuracy_tracker',
        'platform': 'huggingface-spaces',
        'parser_running': parser_thread.is_alive() if 'parser_thread' in globals() else False,
        'log_file_exists': log_file.exists(),
        'total_readings': len(accuracy_history),
        'latest_accuracy': accuracy_history[-1]['accuracy'] if accuracy_history else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/')
def index():
    """Simple status page"""
    return jsonify({
        'service': 'K1RL QUANT Accuracy Tracker',
        'version': 'HuggingFace Spaces Edition',
        'readings': len(accuracy_history),
        'endpoints': [
            '/api/accuracy/current',
            '/api/accuracy/history', 
            '/api/accuracy/stats',
            '/api/accuracy/timerange/<hours>',
            '/health'
        ]
    })

# ============================================================================
# MAIN (HuggingFace Spaces Compatible)
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("K1RL QUANT - Real-Time Accuracy Tracker")
    print("HuggingFace Spaces Edition 🤗")
    print("=" * 70)
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📁 Log directory: {LOG_DIR}")
    print(f"📊 Loaded {len(accuracy_history)} existing readings")
    print("")
    print("⚠️  NOTE: This is a standalone backup service.")
    print("    In HuggingFace Spaces, accuracy tracking is integrated")
    print("    into the main dashboard_service.py for efficiency.")
    print("")
    print("🔄 Starting background parser...")
    
    # Start parser thread
    parser_thread = threading.Thread(target=parse_accuracy_from_logs, daemon=True)
    parser_thread.start()
    
    # ✅ FIXED: Use port 7861 to avoid conflict with main service (7860)
    print("🌐 Starting API server on port 7861...")
    print("    Main dashboard: http://localhost:7860")
    print("    Accuracy API:   http://localhost:7861")
    print("=" * 70)
    
    # Save data on shutdown
    import atexit
    atexit.register(save_data)
    
    try:
        app.run(
            host='0.0.0.0',
            port=7861,  # ✅ FIXED: Different port to avoid conflict
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("🛑 Shutting down accuracy tracker...")
        save_data()
