#!/usr/bin/env python3
"""
K1RL QUANT - Accuracy Tracking API
HuggingFace Spaces Edition
Standalone module to track training accuracy
"""

import re
import json
import time
import os
from datetime import datetime
from collections import deque
from pathlib import Path
import threading

# ✅ FIXED: Container-safe storage paths
BASE_DIR = Path('/home/user/app')
DATA_DIR = BASE_DIR / 'data'
LOG_DIR = BASE_DIR / 'logs'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ✅ FIXED: HuggingFace Spaces compatible paths
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
    
    print(f"🔍 Starting accuracy parser on {log_file}")
    
    while True:
        try:
            # Check if log file exists
            if not log_file.exists():
                print(f"⚠️  Log file not found: {log_file}")
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
                                print(f"📈 Accuracy: {accuracy:.1f}% (iteration {data_point['iteration']})")
                                
                                # Save periodically
                                if len(accuracy_history) % 10 == 0:
                                    save_accuracy_data()
                                
                                break  # Only take first match per line
            
            time.sleep(5)
        except Exception as e:
            print(f"⚠️  Error parsing logs: {e}")
            time.sleep(10)

def save_accuracy_data():
    """Save accuracy data to file"""
    try:
        with open(accuracy_file, 'w') as f:
            json.dump(list(accuracy_history), f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save accuracy data: {e}")

def get_current_accuracy():
    """Get current accuracy"""
    if accuracy_history:
        return accuracy_history[-1]
    return {'accuracy': 0, 'timestamp': None, 'iteration': 0}

def get_accuracy_history():
    """Get all history"""
    return list(accuracy_history)

def get_accuracy_stats():
    """Get comprehensive statistics"""
    if not accuracy_history:
        return {
            'current': 0,
            'average': 0,
            'min': 0,
            'max': 0,
            'count': 0,
            'last_10_avg': 0,
            'trend': 'unknown'
        }
    
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
    
    return {
        'current': accuracies[-1],
        'average': sum(accuracies) / len(accuracies),
        'min': min(accuracies),
        'max': max(accuracies),
        'count': len(accuracies),
        'last_10_avg': sum(accuracies[-10:]) / min(10, len(accuracies)),
        'trend': trend,
        'std_dev': (sum((x - sum(accuracies)/len(accuracies))**2 for x in accuracies) / len(accuracies))**0.5,
        'latest_timestamp': accuracy_history[-1]['timestamp'] if accuracy_history else None
    }

def get_accuracy_by_timerange(hours=24):
    """Get accuracy data for specific time range"""
    if not accuracy_history:
        return []
    
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
    
    return filtered

# ✅ ADDED: HuggingFace Spaces health check
def get_parser_status():
    """Get parser health status"""
    log_file = LOG_DIR / 'quasar_engine.log'
    
    status = {
        'running': parser_thread.is_alive() if 'parser_thread' in globals() else False,
        'log_file_exists': log_file.exists(),
        'log_file_path': str(log_file),
        'total_readings': len(accuracy_history),
        'last_reading': accuracy_history[-1] if accuracy_history else None,
        'data_file_exists': accuracy_file.exists(),
        'data_file_path': str(accuracy_file)
    }
    
    return status

# Start parser thread
print("🚀 Starting accuracy parser thread...")
parser_thread = threading.Thread(target=parse_accuracy_from_logs, daemon=True)
parser_thread.start()

# ✅ ADDED: Graceful shutdown handler
import atexit

def cleanup():
    """Save data on shutdown"""
    print("💾 Saving accuracy data on shutdown...")
    save_accuracy_data()

atexit.register(cleanup)

if __name__ == "__main__":
    print("=" * 60)
    print("K1RL QUANT - Accuracy Tracker")
    print("HuggingFace Spaces Edition 🤗")
    print("=" * 60)
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📁 Log directory: {LOG_DIR}")
    print(f"📊 Loaded {len(accuracy_history)} existing readings")
    print("🔄 Parser running in background...")
    print("=" * 60)
    
    # Keep alive
    try:
        while True:
            time.sleep(60)
            if len(accuracy_history) > 0:
                latest = accuracy_history[-1]
                print(f"📈 Latest accuracy: {latest['accuracy']:.1f}% ({latest['timestamp']})")
    except KeyboardInterrupt:
        print("🛑 Shutting down accuracy tracker...")
        cleanup()
