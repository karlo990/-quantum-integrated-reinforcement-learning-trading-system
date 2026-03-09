#!/usr/bin/env python3
"""
K1RL QUASAR — Real-Time Monitoring Dashboard (HuggingFace Spaces)
Enhanced with visitor tracking, metrics extraction, and container optimization

FIXES APPLIED:
  BUG-1: avn_accuracy regex 'Avg? Accuracy' never matched 'AVN Accuracy:'
          + operator precedence broke the not-yet-set guard
  BUG-2: get_service_status() checked 'RUNNING' in the whole output blob
          instead of per-service line → every service showed ACTIVE even
          when crashed
  BUG-3: /logs/<service> only validated 'quasar_engine' but dashboard fetched
          '/logs/quasar_engine' which returned 400 — added quasar_engine key
          to LOG_FILES and allowed_services
"""

from flask import Flask, render_template, jsonify, Response, send_from_directory, request
from flask_cors import CORS
import subprocess
import json
import time
import os
import re
from datetime import datetime
from collections import deque, defaultdict
import psutil
import threading
import sys
from pathlib import Path

# ── Rich visitor tracking (bot-aware, geo-located) ─────────────────────────
try:
    import visitor_tracker as _vt
    _VT_AVAILABLE = True
except Exception as _vt_err:
    _VT_AVAILABLE = False
    print(f"⚠️  visitor_tracker unavailable — using fallback: {_vt_err}")

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION (HuggingFace Spaces)
# ============================================================================

BASE_DIR       = Path('/home/user/app')
LOG_DIR        = BASE_DIR / 'logs'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
STATUS_FILE    = Path('/tmp/quasar_status.json')

for dir_path in [LOG_DIR, CHECKPOINT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

LOG_FILES = {
    'quasar':        LOG_DIR / 'quasar_engine.log',
    'quasar_engine': LOG_DIR / 'quasar_engine.log',   # FIX BUG-3: alias
    'features':      LOG_DIR / 'features.log',
    'rewards':       LOG_DIR / 'rewards.log'
}

# ============================================================================
# VISITOR TRACKING  (delegates to visitor_tracker.py when available)
# ============================================================================

# Fallback simple tracker used only when visitor_tracker import fails
_fb_active:   dict = {}
_fb_all_time: set  = set()
_fb_lock              = threading.Lock()
_FB_TIMEOUT           = 120


def track_visitor(ip, user_agent=''):
    if _VT_AVAILABLE:
        _vt.track_visitor(ip, user_agent)
    else:
        with _fb_lock:
            _fb_active[ip] = time.time()
            _fb_all_time.add(ip)


def get_visitor_stats():
    if _VT_AVAILABLE:
        return _vt.get_stats()
    # Fallback
    with _fb_lock:
        now   = time.time()
        stale = [ip for ip, ts in _fb_active.items() if now - ts > _FB_TIMEOUT]
        for ip in stale:
            del _fb_active[ip]
        return {
            'active_now':      len(_fb_active),
            'active_ips':      list(_fb_active.keys()),
            'all_time_unique': len(_fb_all_time),
            'bots_active':     0,
            'bots_total':      0,
        }

# ============================================================================
# METRICS STORAGE
# ============================================================================

loss_history     = deque(maxlen=50)
metrics_history  = deque(maxlen=100)
log_buffer       = deque(maxlen=500)

# ============================================================================
# METRIC EXTRACTION
# ============================================================================

def extract_latest_metrics():
    """Extract latest metrics from quasar log"""
    try:
        log_file = LOG_FILES['quasar']
        if not log_file.exists():
            return None

        with open(log_file, 'r') as f:
            lines = f.readlines()[-500:]

        metrics = {
            'training_steps':    None,
            'actor_loss':        None,
            'critic_loss':       None,
            'total_loss':        None,
            'avn_loss':          None,
            'avn_accuracy':      None,
            'buffer_size':       None,
            'matched_rewards':   None,
            'unmatched_rewards': None,
            'duplicates':        None,
            'entropy_bonus':     None,
            'diversity_loss':    None,
            'entanglement_loss': None,
            'timestamp':         datetime.now().isoformat()
        }

        for line in reversed(lines):

            if "'actor_loss':" in line or '"actor_loss":' in line:
                if not metrics['actor_loss']:
                    m = re.search(r"'actor_loss':\s*([-\d.]+)", line)
                    if m:
                        metrics['actor_loss'] = float(m.group(1))

                if not metrics['critic_loss']:
                    m = re.search(r"'critic_loss':\s*([-\d.]+)", line)
                    if m:
                        metrics['critic_loss'] = float(m.group(1))

                if not metrics['total_loss']:
                    m = re.search(r"'total_loss':\s*([-\d.]+)", line)
                    if m:
                        metrics['total_loss'] = float(m.group(1))

                if not metrics['buffer_size']:
                    m = re.search(r"'buffer_size':\s*(\d+)", line)
                    if m:
                        metrics['buffer_size'] = int(m.group(1))

            if 'avn_training_steps:' in line and not metrics['training_steps']:
                m = re.search(r'avn_training_steps:\s*(\d+)', line)
                if m:
                    metrics['training_steps'] = int(m.group(1))

            if '🎯 Avg Loss:' in line and not metrics['avn_loss']:
                m = re.search(r'Avg Loss:\s*([\d.]+)', line)
                if m:
                    metrics['avn_loss'] = float(m.group(1))

            # ── FIX BUG-1a: parenthesise OR before applying `and not` guard
            # ── FIX BUG-1b: regex uses (?:AVN|Avg) to match both forms
            if ('AVN Accuracy:' in line or 'Avg Accuracy:' in line) and metrics['avn_accuracy'] is None:
                m = re.search(r'(?:AVN|Avg) Accuracy:\s*([\d.]+)%?', line)
                if m:
                    metrics['avn_accuracy'] = float(m.group(1))

            if 'avn_rewards:' in line:
                if not metrics['matched_rewards']:
                    m = re.search(r'matched=(\d+)', line)
                    if m:
                        metrics['matched_rewards'] = int(m.group(1))
                if not metrics['unmatched_rewards']:
                    m = re.search(r'unmatched=(\d+)', line)
                    if m:
                        metrics['unmatched_rewards'] = int(m.group(1))
                if not metrics['duplicates']:
                    m = re.search(r'duplicate=(\d+)', line)
                    if m:
                        metrics['duplicates'] = int(m.group(1))

        if metrics['actor_loss'] is not None:
            loss_history.append({
                'actor_loss':   metrics['actor_loss'],
                'critic_loss':  metrics['critic_loss'],
                'total_loss':   metrics['total_loss'],
                'avn_accuracy': metrics['avn_accuracy'],
                'timestamp':    datetime.now().isoformat()
            })

        return metrics

    except Exception as e:
        print(f"Metric extraction error: {e}")
        return None


# ── FIX BUG-2: service status now parsed per line ──────────────────────────
def get_service_status():
    """Get supervisor service status — per-line so one RUNNING doesn't
       mask other services that are STOPPED/FATAL"""
    services = {}

    # Maps supervisord program name → friendly API key used by the dashboard
    service_map = {
        'quasar_engine':  'quasar_engine',
        'dashboard':      'dashboard',
        'health_monitor': 'health_monitor',
        'redis':          'redis',
        'features':       'features',
        'rewards':        'rewards',
    }

    try:
        result = subprocess.run(
            ['supervisorctl', 'status'],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"supervisorctl error: {result.stderr.strip()}")

        for line in result.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue

            program_name = parts[0]
            status_token = parts[1] if len(parts) > 1 else 'UNKNOWN'
            friendly     = service_map.get(program_name)

            if friendly is None:
                continue

            if status_token == 'RUNNING':
                services[friendly] = 'active'
            elif status_token in ('STOPPED', 'FATAL', 'EXITED', 'BACKOFF'):
                services[friendly] = 'inactive'
            else:
                services[friendly] = 'unknown'

    except Exception:
        # Safe fallback: mark unknown rather than falsely green
        services = {
            'dashboard':      'active',   # we're responding so this is up
            'health_monitor': 'unknown',
            'redis':          'unknown',
            'quasar_engine':  'unknown'
        }

    return services


def get_system_resources():
    """Get system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory      = psutil.virtual_memory()
        disk        = psutil.disk_usage('/')

        quasar_memory = 0
        process_count = 0
        for proc in psutil.process_iter(['name', 'memory_info', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline', []) or [])
                if 'quasar' in cmdline.lower() or 'python' in cmdline.lower():
                    quasar_memory += proc.info['memory_info'].rss / (1024 ** 3)
                    process_count += 1
            except Exception:
                pass

        return {
            'cpu_percent':       round(cpu_percent, 1),
            'memory_percent':    round(memory.percent, 1),
            'memory_used_gb':    round(memory.used       / (1024 ** 3), 2),
            'memory_total_gb':   round(memory.total      / (1024 ** 3), 2),
            'memory_available_gb': round(memory.available / (1024 ** 3), 2),
            'disk_percent':      round(disk.percent, 1),
            'disk_used_gb':      round(disk.used         / (1024 ** 3), 2),
            'disk_total_gb':     round(disk.total         / (1024 ** 3), 2),
            'quasar_memory_gb':  round(quasar_memory, 2),
            'process_count':     process_count
        }
    except Exception as e:
        print(f"Resource error: {e}")
        return None


def get_checkpoint_info():
    """Get latest checkpoint information"""
    try:
        if not CHECKPOINT_DIR.exists():
            return None

        checkpoints = list(CHECKPOINT_DIR.glob('checkpoint_*.pt'))
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = checkpoints[0]

        size_mb  = latest.stat().st_size / (1024 ** 2)
        modified = datetime.fromtimestamp(latest.stat().st_mtime)

        match = re.search(r'step_(\d+)', latest.name)
        step  = int(match.group(1)) if match else None

        return {
            'filename': latest.name,
            'size_mb':  round(size_mb, 2),
            'modified': modified.strftime('%Y-%m-%d %H:%M:%S'),
            'step':     step,
            'count':    len(checkpoints)
        }
    except Exception as e:
        print(f"Checkpoint error: {e}")
        return None

# ============================================================================
# REQUEST TRACKING
# ============================================================================

@app.before_request
def before_request():
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if ip:
        ip         = ip.split(',')[0].strip()
        user_agent = request.headers.get('User-Agent', '')
        track_visitor(ip, user_agent)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    dashboard_path = BASE_DIR / 'dashboard.html'
    if dashboard_path.exists():
        return send_from_directory(str(BASE_DIR), 'dashboard.html')

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>K1RL QUASAR — Quantitative Intelligence Observatory</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background: #0a0a0a; color: #00ffff; font-family: monospace; text-align: center; padding: 50px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { background: #1a1a1a; padding: 20px; margin: 20px 0; border-radius: 10px; }
            .loading { animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌌 K1RL QUASAR</h1>
            <h2>Quantitative Intelligence Observatory</h2>
            <div class="status loading">
                <p>🚀 System Initializing...</p>
                <p>Dashboard components loading...</p>
                <div id="status">Connecting to services...</div>
            </div>
        </div>
        <script>
            setTimeout(() => window.location.reload(), 5000);
        </script>
    </body>
    </html>
    """, 200


@app.route('/api/metrics')
def api_metrics():
    try:
        metrics    = extract_latest_metrics()
        services   = get_service_status()
        resources  = get_system_resources()
        checkpoint = get_checkpoint_info()

        return jsonify({
            'metrics':    metrics,
            'services':   services,
            'resources':  resources,
            'checkpoint': checkpoint,
            'timestamp':  datetime.now().isoformat(),
            'visitors':   get_visitor_stats(),
            'platform':   'huggingface-spaces'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'platform': 'huggingface-spaces'}), 500


@app.route('/api/visitors')
def api_visitors():
    return jsonify(get_visitor_stats())


@app.route('/api/visitors/detailed')
def api_visitors_detailed():
    """Full visitor analytics — geographic, device, browser breakdown + session list"""
    try:
        if _VT_AVAILABLE:
            stats    = _vt.get_stats()
            detailed = _vt.get_detailed_stats()
            all_v    = _vt.get_all_visitors()
            return jsonify({
                **stats,
                **detailed,
                'visitors':  all_v.get('visitors', []),
                'timestamp': datetime.now().isoformat(),
            })
        # Fallback when visitor_tracker not available
        basic = get_visitor_stats()
        return jsonify({**basic, 'visitors': [], 'countries': {},
                        'devices': {}, 'browsers': {}, 'cities': {},
                        'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE) as f:
                data = json.load(f)
        else:
            data = {'status': 'starting'}

        data['visitors'] = get_visitor_stats()
        data['platform'] = 'huggingface-spaces'
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/health')
@app.route('/api/health')
def health():
    return jsonify({
        'status':    'ok',
        'service':   'k1rl-quasar',
        'platform':  'huggingface-spaces',
        'timestamp': datetime.now().isoformat(),
        'visitors':  get_visitor_stats()
    })


# ── FIX BUG-3: allowed_services now includes 'quasar_engine' ───────────────
@app.route('/logs/<service>')
@app.route('/api/logs/<service>')
def stream_logs(service):
    """Serve service logs — plain text or JSON depending on Accept header"""
    allowed_services = list(LOG_FILES.keys())   # includes 'quasar_engine' now
    if service not in allowed_services:
        return f"Invalid service '{service}'. Valid: {allowed_services}", 400

    log_path = LOG_FILES[service]

    def generate():
        try:
            if log_path.exists():
                with open(log_path, 'r') as f:
                    lines = f.readlines()[-200:]
                    yield ''.join(lines)
            else:
                yield f'Log file not found: {log_path}'
        except Exception as e:
            yield f'Error reading log: {e}'

    if request.headers.get('Accept', '').startswith('application/json'):
        try:
            if log_path.exists():
                with open(log_path, 'r') as f:
                    lines = f.readlines()[-200:]
            else:
                lines = []
            return jsonify({'service': service, 'lines': [l.rstrip() for l in lines], 'count': len(lines)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return Response(generate(), mimetype='text/plain')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("K1RL QUASAR — Quantitative Intelligence Observatory")
    print("HuggingFace Spaces Deployment")
    print("=" * 70)
    print(f"Port: 7860 (HF Spaces Standard)")
    print(f"Base Directory: {BASE_DIR}")
    print("=" * 70)

    app.run(
        host='0.0.0.0',
        port=7860,
        debug=False,
        threaded=True
    )