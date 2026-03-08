"""
============================================================================
K1RL QUASAR — Health Monitor Service (HuggingFace Spaces)
============================================================================
Monitors system health with proper container paths and 12GB RAM management
============================================================================
"""

import os
import gc
import sys
import json
import time
import logging
import shutil
import subprocess
import threading
import psutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [HEALTH] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/user/app/logs/health_monitor.log', mode='a')
    ]
)
log = logging.getLogger('health_monitor')

# ── Configuration (HuggingFace Spaces paths) ───────────────────────────────
RAM_WARN      = 70   # % — log warning (lower for 12GB limit)
RAM_GC        = 78   # % — force garbage collection
RAM_CLEAR     = 85   # % — clear tmp caches
RAM_CRITICAL  = 90   # % — alert + aggressive cleanup

DISK_WARN     = 80   # % disk used — start pruning
DISK_CRITICAL = 90   # % disk — emergency prune

CHECK_INTERVAL = 30  # seconds between health checks

# Paths (HuggingFace Spaces standard)
STATUS_FILE    = Path('/tmp/quasar_status.json')
CACHE_DIR      = Path('/tmp/quasar_cache')
LOG_DIR        = Path('/home/user/app/logs')
CHECKPOINT_DIR = Path('/home/user/app/checkpoints')
BASE_DIR       = Path('/home/user/app')

# Global health state
health_state = {
    'ram_pct': 0,
    'disk_pct': 0,
    'gc_runs': 0,
    'cache_clears': 0,
    'uptime_seconds': 0,
    'last_check': 0,
    'platform': 'huggingface-spaces'
}
start_time = time.time()

# Ensure directories exist
for dir_path in [LOG_DIR, CHECKPOINT_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def _read_status():
    """Read current status from file"""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE) as f:
                return json.load(f)
    except Exception as e:
        log.debug(f"Status read error: {e}")
    return {}

def _write_status(update: dict):
    """Write status update to file"""
    try:
        current = _read_status()
        current.update(update)
        with open(STATUS_FILE, 'w') as f:
            json.dump(current, f)
    except Exception as e:
        log.error(f"Status write failed: {e}")

def _clear_tmp_caches():
    """Clear temporary cache directories"""
    cleared_mb = 0
    try:
        # Clear quasar cache
        if CACHE_DIR.exists():
            size = sum(f.stat().st_size for f in CACHE_DIR.rglob('*') if f.is_file())
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(exist_ok=True)
            cleared_mb = size / 1e6
            log.info(f"🧹 Cleared quasar cache: {cleared_mb:.1f} MB freed")

        # Clear Python bytecode cache
        for pycache in BASE_DIR.rglob('__pycache__'):
            shutil.rmtree(pycache, ignore_errors=True)

        # Clear HuggingFace cache if too large
        hf_cache = Path('/home/user/app/.cache/huggingface')
        if hf_cache.exists():
            try:
                cache_size = sum(f.stat().st_size for f in hf_cache.rglob('*') if f.is_file()) / 1e6
                if cache_size > 500:  # If cache > 500MB
                    shutil.rmtree(hf_cache)
                    hf_cache.mkdir(parents=True, exist_ok=True)
                    log.info(f"🧹 Cleared HF cache: {cache_size:.1f} MB freed")
            except Exception:
                pass

        gc.collect()
        health_state['cache_clears'] += 1
        log.info(f"✅ Cache clear complete (total: {health_state['cache_clears']})")

    except Exception as e:
        log.error(f"Cache clear error: {e}")

def _prune_old_checkpoints(keep_last=3):
    """Keep only last N checkpoints to save disk space"""
    try:
        ckpts = sorted(CHECKPOINT_DIR.glob('checkpoint_*.pt'), key=lambda f: f.stat().st_mtime)
        to_delete = ckpts[:-keep_last] if len(ckpts) > keep_last else []
        
        for f in to_delete:
            size_mb = f.stat().st_size / 1e6
            f.unlink()
            log.info(f"🗑️ Pruned checkpoint: {f.name} ({size_mb:.1f}MB)")
            
    except Exception as e:
        log.error(f"Checkpoint prune error: {e}")

def _prune_old_logs(keep_mb=50):
    """Keep logs under specified total size"""
    try:
        log_files = sorted(LOG_DIR.glob('*.log'), key=lambda f: f.stat().st_mtime)
        total_mb = sum(f.stat().st_size for f in log_files) / 1e6
        
        while total_mb > keep_mb and log_files:
            oldest = log_files.pop(0)
            size_mb = oldest.stat().st_size / 1e6
            
            # Don't delete the current health monitor log
            if oldest.name == 'health_monitor.log':
                continue
                
            oldest.unlink()
            total_mb -= size_mb
            log.info(f"🗑️ Pruned log: {oldest.name} ({size_mb:.1f}MB)")
            
    except Exception as e:
        log.error(f"Log prune error: {e}")

def _check_services():
    """Check if services are running via supervisorctl"""
    services_status = {}
    try:
        result = subprocess.run(
            ['supervisorctl', 'status'],
            capture_output=True, text=True, timeout=5
        )
        
        for line in result.stdout.splitlines():
            if 'RUNNING' in line:
                service_name = line.split()[0]
                services_status[service_name] = 'running'
            elif 'FATAL' in line or 'STOPPED' in line:
                service_name = line.split()[0] 
                services_status[service_name] = 'stopped'
                log.warning(f"⚠️ Service not running: {line.strip()}")
                
        _write_status({'services': services_status})
        
    except Exception as e:
        log.debug(f"Service check error: {e}")

def run_health_loop():
    """Main health monitoring loop"""
    log.info("🏥 K1RL QUASAR Health Monitor started")
    log.info(f"   Platform: HuggingFace Spaces (12GB RAM)")
    log.info(f"   RAM thresholds: warn={RAM_WARN}% gc={RAM_GC}% clear={RAM_CLEAR}% critical={RAM_CRITICAL}%")
    log.info(f"   Check interval: {CHECK_INTERVAL}s")
    log.info("=" * 60)

    while True:
        try:
            # ── RAM monitoring ────────────────────────────────────────────
            mem = psutil.virtual_memory()
            ram_pct = mem.percent
            ram_used_gb = mem.used / 1e9
            ram_avail_gb = mem.available / 1e9
            ram_total_gb = mem.total / 1e9
            
            health_state['ram_pct'] = ram_pct

            if ram_pct >= RAM_CRITICAL:
                log.critical(f"🚨 RAM CRITICAL: {ram_pct:.1f}% ({ram_used_gb:.1f}GB used) — Aggressive cleanup!")
                _clear_tmp_caches()
                _prune_old_checkpoints(keep_last=1)
                _prune_old_logs(keep_mb=20)

            elif ram_pct >= RAM_CLEAR:
                log.warning(f"⚠️ RAM HIGH: {ram_pct:.1f}% — Clearing caches")
                _clear_tmp_caches()

            elif ram_pct >= RAM_GC:
                log.info(f"♻️ RAM {ram_pct:.1f}% — Running garbage collection")
                gc.collect()
                health_state['gc_runs'] += 1

            elif ram_pct >= RAM_WARN:
                log.info(f"📊 RAM: {ram_pct:.1f}% ({ram_used_gb:.1f}GB/{ram_total_gb:.1f}GB)")

            # ── Disk monitoring ───────────────────────────────────────────
            disk = psutil.disk_usage('/')
            disk_pct = disk.percent
            disk_used_gb = disk.used / 1e9
            disk_total_gb = disk.total / 1e9
            
            health_state['disk_pct'] = disk_pct

            if disk_pct >= DISK_CRITICAL:
                log.warning(f"💾 DISK CRITICAL: {disk_pct:.1f}% — Emergency prune")
                _prune_old_checkpoints(keep_last=1)
                _prune_old_logs(keep_mb=10)
            elif disk_pct >= DISK_WARN:
                log.info(f"💾 DISK HIGH: {disk_pct:.1f}% — Pruning old files")
                _prune_old_checkpoints(keep_last=3)
                _prune_old_logs(keep_mb=50)

            # ── Status update ─────────────────────────────────────────────
            uptime = time.time() - start_time
            health_state['uptime_seconds'] = uptime
            health_state['last_check'] = time.time()

            _write_status({
                'health': {
                    'ram_pct': round(ram_pct, 1),
                    'ram_used_gb': round(ram_used_gb, 2),
                    'ram_avail_gb': round(ram_avail_gb, 2),
                    'ram_total_gb': round(ram_total_gb, 2),
                    'disk_pct': round(disk_pct, 1),
                    'disk_used_gb': round(disk_used_gb, 2),
                    'disk_total_gb': round(disk_total_gb, 2),
                    'gc_runs': health_state['gc_runs'],
                    'cache_clears': health_state['cache_clears'],
                    'uptime_hours': round(uptime / 3600, 2),
                    'last_check': health_state['last_check'],
                    'platform': health_state['platform']
                }
            })

            # ── Service check every 5 minutes ────────────────────────────
            if int(uptime) % 300 < CHECK_INTERVAL:
                _check_services()

            # ── Log health summary every hour ─────────────────────────────
            if int(uptime) % 3600 < CHECK_INTERVAL:
                log.info(f"🏥 Health Summary: RAM {ram_pct:.1f}%, Disk {disk_pct:.1f}%, "
                        f"GC runs: {health_state['gc_runs']}, Cache clears: {health_state['cache_clears']}")

        except Exception as e:
            log.error(f"Health check error: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    run_health_loop()
