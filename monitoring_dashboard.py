#!/usr/bin/env python3
"""
K1RL QUANT - Real-Time Monitoring Dashboard
HuggingFace Spaces Edition
Displays live metrics, logs, and system status
"""

from flask import Flask, render_template, jsonify, Response, session, request

# Accuracy tracking module (optional - won't crash if missing)
try:
    import accuracy_api
    ACCURACY_AVAILABLE = True
except ImportError:
    ACCURACY_AVAILABLE = False
    print("⚠️  accuracy_api not found — accuracy endpoints disabled")
from flask_cors import CORS
import subprocess
import json
import time
import os
import re
from datetime import datetime, timedelta
from collections import deque
import psutil
import uuid

app = Flask(__name__)
CORS(app)

# ⚠️ IMPORTANT: Secret key for sessions (required for visitor tracking)
app.secret_key = 'k1rl-quant-2026-' + str(uuid.uuid4())[:8]

# ✅ FIXED: HuggingFace Spaces compatible paths
LOG_FILES = {
    'quasar': '/home/user/app/logs/quasar_engine.log',
    'features': '/home/user/app/logs/features.log', 
    'rewards': '/home/user/app/logs/rewards.log'
}

CHECKPOINT_DIR = '/home/user/app/checkpoints'

# In-memory storage
metrics_history = deque(maxlen=100)
log_buffer = deque(maxlen=500)

# ============================================================================
# VISITOR TRACKING SYSTEM (HF Spaces Optimized)
# ============================================================================

# In-memory storage for active visitors
active_visitors = {}  # {session_id: {'last_seen': datetime, 'ip': str}}
visitor_stats = {
    'total_unique': 0,
    'total_pageviews': 0,
    'first_visit': datetime.now().isoformat()
}

# ✅ FIXED: Container-safe visitor data file
VISITOR_DATA_FILE = '/home/user/app/logs/visitor_data.json'

def load_visitor_data():
    """Load visitor statistics from file on startup"""
    global visitor_stats
    if os.path.exists(VISITOR_DATA_FILE):
        try:
            with open(VISITOR_DATA_FILE, 'r') as f:
                visitor_stats = json.load(f)
            print(f"📊 Loaded visitor stats: {visitor_stats['total_unique']} total unique visitors")
        except Exception as e:
            print(f"⚠️  Could not load visitor data: {e}")

def save_visitor_data():
    """Save visitor statistics to file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(VISITOR_DATA_FILE), exist_ok=True)
        with open(VISITOR_DATA_FILE, 'w') as f:
            json.dump(visitor_stats, f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save visitor data: {e}")

def cleanup_inactive_visitors():
    """Remove visitors inactive for more than 5 minutes"""
    global active_visitors
    now = datetime.now()
    inactive_threshold = timedelta(minutes=5)
    
    to_remove = [vid for vid, data in active_visitors.items() 
                 if now - data['last_seen'] > inactive_threshold]
    
    for vid in to_remove:
        del active_visitors[vid]
    
    if to_remove:
        print(f"🧹 Cleaned up {len(to_remove)} inactive visitors")
    
    return len(to_remove)

def track_visitor():
    """Track visitor activity and return visitor ID"""
    global active_visitors, visitor_stats
    
    # Clean up inactive visitors
    cleanup_inactive_visitors()
    
    # Get or create visitor session ID
    if 'visitor_id' not in session:
        session['visitor_id'] = str(uuid.uuid4())
        visitor_stats['total_unique'] += 1
        save_visitor_data()
        
        visitor_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        print(f"👤 New visitor #{visitor_stats['total_unique']}: {visitor_ip}")
    
    # Update visitor activity
    visitor_id = session['visitor_id']
    active_visitors[visitor_id] = {
        'last_seen': datetime.now(),
        'ip': request.headers.get('X-Forwarded-For', request.remote_addr)
    }
    
    # Increment pageview counter
    visitor_stats['total_pageviews'] += 1
    
    # Save periodically (every 50 pageviews)
    if visitor_stats['total_pageviews'] % 50 == 0:
        save_visitor_data()
    
    return visitor_id

# Track visitors before each request
@app.before_request
def before_request():
    """Track visitors on every request"""
    # Skip API and static files to avoid inflating counts
    if not request.path.startswith('/api/') and not request.path.startswith('/static/'):
        try:
            track_visitor()
        except Exception as e:
            print(f"⚠️  Error tracking visitor: {e}")

# ============================================================================
# METRIC EXTRACTION (HF Spaces Compatible)
# ============================================================================

def extract_latest_metrics():
    """Extract latest metrics from quasar log"""
    try:
        # Check if log file exists
        if not os.path.exists(LOG_FILES['quasar']):
            print(f"⚠️  Log file not found: {LOG_FILES['quasar']}")
            return None
            
        # Read last 200 lines
        result = subprocess.run(
            ['tail', '-200', LOG_FILES['quasar']],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.split('\n')
        
        metrics = {
            'training_steps': None,
            'actor_loss': None,
            'critic_loss': None,
            'avn_loss': None,
            'avn_accuracy': None,  # ✅ ADDED: AVN accuracy
            'buffer_size': None,
            'matched_rewards': None,
            'unmatched_rewards': None,
            'duplicates': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Parse metrics from logs
        for line in reversed(lines):
            # Training steps
            if 'avn_training_steps:' in line and not metrics['training_steps']:
                match = re.search(r'avn_training_steps:\s*(\d+)', line)
                if match:
                    metrics['training_steps'] = int(match.group(1))
            
            # Actor loss
            if 'Actor Loss:' in line and not metrics['actor_loss']:
                match = re.search(r'Actor Loss:\s*([-\d.]+)', line)
                if match:
                    metrics['actor_loss'] = float(match.group(1))
            
            # Critic loss
            if 'Critic Loss:' in line and not metrics['critic_loss']:
                match = re.search(r'Critic Loss:\s*([\d.]+)', line)
                if match:
                    metrics['critic_loss'] = float(match.group(1))
            
            # AVN loss
            if 'Avg Loss:' in line and not metrics['avn_loss']:
                match = re.search(r'Avg Loss:\s*([\d.]+)', line)
                if match:
                    metrics['avn_loss'] = float(match.group(1))
            
            # ✅ ADDED: AVN Accuracy extraction
            if 'AVN Accuracy:' in line or 'Avg Accuracy:' in line and not metrics['avn_accuracy']:
                match = re.search(r'Avg? Accuracy:\s*([\d.]+)%?', line)
                if match:
                    metrics['avn_accuracy'] = float(match.group(1))
            
            # Buffer size
            if 'buffer_size' in line and not metrics['buffer_size']:
                match = re.search(r'buffer_size[\'"]?\s*:\s*(\d+)', line)
                if match:
                    metrics['buffer_size'] = int(match.group(1))
            
            # Rewards
            if 'avn_rewards:' in line:
                matched = re.search(r'matched=(\d+)', line)
                unmatched = re.search(r'unmatched=(\d+)', line)
                duplicate = re.search(r'duplicate=(\d+)', line)
                
                if matched and not metrics['matched_rewards']:
                    metrics['matched_rewards'] = int(matched.group(1))
                if unmatched and not metrics['unmatched_rewards']:
                    metrics['unmatched_rewards'] = int(unmatched.group(1))
                if duplicate and not metrics['duplicates']:
                    metrics['duplicates'] = int(duplicate.group(1))
        
        return metrics
        
    except Exception as e:
        print(f"Metric extraction error: {e}")
        return None


def get_service_status():
    """Get supervisor service status (HF Spaces compatible)"""
    services = {}

    service_map = {
        'quasar':   'quasar_engine',
        'features': 'health_monitor', 
        'rewards':  'redis'
    }

    try:
        result = subprocess.run(
            ['supervisorctl', 'status'],
            capture_output=True, text=True, timeout=3
        )
        output = result.stdout

        for friendly, supervisor_name in service_map.items():
            if supervisor_name in output:
                if 'RUNNING' in output:
                    services[friendly] = 'active'
                elif 'STOPPED' in output:
                    services[friendly] = 'inactive'
                else:
                    services[friendly] = 'unknown'
            else:
                services[friendly] = 'unknown'
    except Exception:
        # Fallback — check process list
        for friendly in service_map:
            services[friendly] = 'unknown'

    return services

def get_system_resources():
    """Get system resource usage (HF Spaces optimized)"""
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        # Find quasar process
        quasar_memory = 0
        process_count = 0
        for proc in psutil.process_iter(['name', 'memory_info', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline', []) or [])
                if 'quasar' in cmdline.lower() or 'quasar' in (proc.info.get('name', '') or '').lower():
                    quasar_memory += proc.info['memory_info'].rss / (1024**3)
                    process_count += 1
            except:
                pass
        
        return {
            'cpu_percent': round(cpu_percent, 1),
            'memory_percent': round(memory_percent, 1),
            'memory_used_gb': round(memory_used_gb, 2),
            'memory_total_gb': round(memory_total_gb, 2),
            'disk_percent': round(disk_percent, 1),
            'disk_used_gb': round(disk_used_gb, 2),
            'disk_total_gb': round(disk_total_gb, 2),
            'quasar_memory_gb': round(quasar_memory, 2),
            'process_count': process_count
        }
    except Exception as e:
        print(f"Resource error: {e}")
        return None


def get_checkpoint_info():
    """Get latest checkpoint information"""
    try:
        if not os.path.exists(CHECKPOINT_DIR):
            return None
        
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
        latest = checkpoints[-1]
        
        filepath = os.path.join(CHECKPOINT_DIR, latest)
        size_mb = os.path.getsize(filepath) / (1024**2)
        modified = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        # Extract step from filename
        match = re.search(r'step_(\d+)', latest)
        step = int(match.group(1)) if match else None
        
        return {
            'filename': latest,
            'size_mb': round(size_mb, 2),
            'modified': modified.strftime('%Y-%m-%d %H:%M:%S'),
            'step': step,
            'count': len(checkpoints)
        }
    except Exception as e:
        print(f"Checkpoint error: {e}")
        return None


def get_process_count():
    """Count running quasar processes"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        count = 0
        for line in result.stdout.split('\n'):
            if 'quasar' in line.lower() and 'python' in line.lower() and 'grep' not in line:
                count += 1
        
        return count
    except:
        return None


# ============================================================================
# API ENDPOINTS (HF Spaces Compatible)
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/metrics')
def api_metrics():
    """Get current metrics (matches dashboard_service.py format)"""
    metrics = extract_latest_metrics()
    services = get_service_status()
    resources = get_system_resources()
    checkpoint = get_checkpoint_info()
    process_count = get_process_count()
    
    # ✅ FIXED: Match dashboard_service.py format
    response = {
        'metrics': metrics or {},
        'services': services,
        'resources': resources or {},
        'health': resources or {},  # Alias for compatibility
        'checkpoint': checkpoint,
        'process_count': process_count,
        'timestamp': datetime.now().isoformat(),
        'visitors': {
            'active_now': len(active_visitors),
            'all_time_unique': visitor_stats['total_unique']
        }
    }
    
    return jsonify(response)


@app.route('/api/visitors')
def api_visitors():
    """Get current visitor statistics"""
    cleanup_inactive_visitors()
    
    stats = {
        'active_now': len(active_visitors),
        'all_time_unique': visitor_stats['total_unique'],
        'total_pageviews': visitor_stats['total_pageviews'],
        'timestamp': datetime.now().isoformat()
    }
    
    # Log occasionally for debugging
    if visitor_stats.get('total_pageviews', 0) % 100 == 0:
        print(f"📊 Visitor stats: {stats['active_now']} active, {stats['all_time_unique']} total unique")
    
    return jsonify(stats)


@app.route('/api/logs/<service>')
@app.route('/logs/<service>')  # ✅ ADDED: Alternative endpoint for compatibility
def api_logs(service):
    """Get recent logs for a service (compatible with both formats)"""
    if service not in LOG_FILES:
        return jsonify({'error': 'Invalid service'}), 400
    
    try:
        log_file = LOG_FILES[service]
        if not os.path.exists(log_file):
            return f"Log file not found: {log_file}", 404
            
        result = subprocess.run(
            ['tail', '-100', log_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.split('\n')
        
        # ✅ ADDED: Support both JSON and text responses
        if request.headers.get('Accept', '').startswith('application/json'):
            return jsonify({
                'service': service,
                'lines': lines[-100:],
                'count': len(lines)
            })
        else:
            # Return as plain text for dashboard compatibility
            return '\n'.join(lines[-100:]), 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream/logs')
def stream_logs():
    """Stream logs in real-time using Server-Sent Events"""
    def generate():
        try:
            # Start tailing the log
            proc = subprocess.Popen(
                ['tail', '-f', LOG_FILES['quasar']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                for line in iter(proc.stdout.readline, ''):
                    if line:
                        yield f"data: {json.dumps({'line': line.strip(), 'timestamp': datetime.now().isoformat()})}\n\n"
            except:
                pass
            finally:
                proc.kill()
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/health')
@app.route('/health')  # ✅ ADDED: Alternative endpoint
def health():
    """Health check endpoint (HF Spaces compatible)"""
    return jsonify({
        'status': 'ok',
        'platform': 'huggingface-spaces',
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# ACCURACY API ENDPOINTS (Enhanced)
# ============================================================================

@app.route('/api/accuracy/current')
def api_accuracy_current():
    if not ACCURACY_AVAILABLE:
        # ✅ FALLBACK: Extract from metrics if accuracy_api not available
        metrics = extract_latest_metrics()
        if metrics and metrics.get('avn_accuracy'):
            return jsonify({
                'accuracy': metrics['avn_accuracy'],
                'timestamp': metrics['timestamp'],
                'iteration': 0
            })
        return jsonify({'error': 'accuracy not available'}), 503
    return jsonify(accuracy_api.get_current_accuracy())

@app.route('/api/accuracy/history')
def api_accuracy_history():
    if not ACCURACY_AVAILABLE:
        return jsonify({'error': 'accuracy history not available'}), 503
    return jsonify(accuracy_api.get_accuracy_history())

@app.route('/api/accuracy/stats')
def api_accuracy_stats():
    if not ACCURACY_AVAILABLE:
        # ✅ FALLBACK: Basic stats from current metrics
        metrics = extract_latest_metrics()
        if metrics and metrics.get('avn_accuracy'):
            acc = metrics['avn_accuracy']
            return jsonify({
                'current': acc,
                'average': acc,
                'min': acc,
                'max': acc,
                'count': 1,
                'last_10_avg': acc,
                'trend': 'stable'
            })
        return jsonify({'error': 'accuracy stats not available'}), 503
    return jsonify(accuracy_api.get_accuracy_stats())


# ============================================================================
# MAIN (HuggingFace Spaces)
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("K1RL QUANT - Quantitative Intelligence Observatory")
    print("HuggingFace Spaces Edition 🤗")
    print("=" * 70)
    
    # Load existing visitor data on startup
    load_visitor_data()
    print(f"📊 Starting with {visitor_stats['total_unique']} total unique visitors")
    
    # ✅ FIXED: Port 7860 for HuggingFace Spaces
    print(f"Starting server on http://0.0.0.0:7860")
    print("HuggingFace Spaces URL: https://your-space-name.hf.space")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=7860,  # ✅ FIXED: HuggingFace Spaces port
        debug=False,
        threaded=True
    )
