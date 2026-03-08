"""
============================================================================
K1RL QUASAR — Training Engine Service (HuggingFace Spaces)
============================================================================
Production-ready training service with:
- Graceful shutdown handling
- HuggingFace Hub checkpoint management  
- Container-optimized resource management
- Crash recovery with exponential backoff
============================================================================
"""

import os
import sys
import gc
import signal
import time
import logging
import traceback
import threading
import torch
import psutil
from pathlib import Path
from huggingface_hub import HfApi
import json

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path('/home/user/app/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [QUASAR-ENGINE] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / 'quasar_engine.log', mode='a')
    ]
)
log = logging.getLogger('quasar_engine')

# ── Configuration ────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get('HF_TOKEN', '')
HF_USERNAME = os.environ.get('HF_USERNAME', 'KarlQuant')
CHECKPOINT_REPO = f"{HF_USERNAME}/k1rl-checkpoints"

# Paths (HuggingFace Spaces standard)
BASE_DIR = Path('/home/user/app')
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
STATUS_FILE = Path('/tmp/quasar_status.json')

# Create directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
CHECKPOINT_EVERY = 500       # Save every N training steps
RAM_GC_THRESHOLD = 75        # Trigger GC when RAM > 75% (HF Spaces has 12GB)
RAM_CRITICAL = 85            # Emergency cache clear when RAM > 85%

# Initialize HuggingFace API
api = HfApi(token=HF_TOKEN) if HF_TOKEN else None
shutdown_requested = False
controller = None

# ── Signal handlers (graceful shutdown) ──────────────────────────────────────

def handle_sigterm(signum, frame):
    """Graceful shutdown: checkpoint → cleanup → exit"""
    global shutdown_requested
    log.info("⚡ SIGTERM received — initiating graceful shutdown...")
    shutdown_requested = True

    if controller is not None:
        try:
            log.info("💾 Emergency checkpoint before shutdown...")
            _save_checkpoint(controller, emergency=True)
        except Exception as e:
            log.error(f"Emergency checkpoint failed: {e}")

    _clear_all_caches()
    log.info("✅ Graceful shutdown complete")
    sys.exit(0)

def handle_sigint(signum, frame):
    handle_sigterm(signum, frame)

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigint)

# ── Cache management ─────────────────────────────────────────────────────────

def _clear_all_caches():
    """Clear all caches — called on crash, OOM, or shutdown"""
    log.info("🧹 Clearing all caches...")
    try:
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear temporary cache directories
        import shutil
        cache_dirs = [
            Path('/tmp/quasar_cache'),
            BASE_DIR / '.cache' / 'huggingface',
            BASE_DIR / '.cache' / 'torch'
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    size_mb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1e6
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    log.info(f"🧹 Cleared {cache_dir.name}: {size_mb:.1f}MB freed")
                except Exception as e:
                    log.debug(f"Cache clear error for {cache_dir}: {e}")
        
        log.info("✅ Cache cleared")
    except Exception as e:
        log.error(f"Cache clear error: {e}")

# ── RAM monitoring ────────────────────────────────────────────────────────────

def _ram_monitor_thread():
    """Background thread: monitor RAM, auto-GC, emergency clear"""
    while not shutdown_requested:
        try:
            mem = psutil.virtual_memory()
            used_pct = mem.percent
            used_gb = mem.used / 1e9
            total_gb = mem.total / 1e9

            if used_pct > RAM_CRITICAL:
                log.warning(f"🚨 RAM CRITICAL: {used_pct:.1f}% ({used_gb:.1f}GB/{total_gb:.1f}GB) — Emergency cache clear!")
                _clear_all_caches()
            elif used_pct > RAM_GC_THRESHOLD:
                log.info(f"⚠️ RAM HIGH: {used_pct:.1f}% ({used_gb:.1f}GB/{total_gb:.1f}GB) — Running GC...")
                gc.collect()

            # Update status for dashboard
            _write_status({
                'ram_pct': used_pct,
                'ram_used_gb': used_gb,
                'ram_total_gb': total_gb,
                'last_ram_check': time.time()
            })

        except Exception as e:
            log.error(f"RAM monitor error: {e}")

        time.sleep(30)

# ── Status management ────────────────────────────────────────────────────────

_status = {}

def _write_status(update: dict):
    """Update status file for dashboard"""
    global _status
    _status.update(update)
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(_status, f)
    except Exception as e:
        log.debug(f"Status write error: {e}")

# ── Checkpoint management ─────────────────────────────────────────────────────

def _save_checkpoint(ctrl, emergency=False):
    """Save checkpoint locally then upload to HF Dataset repo"""
    if not ctrl:
        return
        
    try:
        step = getattr(ctrl, 'training_step', 0)
        tag = 'emergency' if emergency else f'step_{step}'
        local_path = CHECKPOINT_DIR / f'checkpoint_{tag}.pt'

        # Gather state
        state = {
            'step': step,
            'timestamp': time.time(),
            'platform': 'huggingface-spaces'
        }

        # Add trainer state if available
        if hasattr(ctrl, 'trainer') and ctrl.trainer is not None:
            t = ctrl.trainer
            if hasattr(t, 'optimizer') and t.optimizer is not None:
                state['optimizer'] = t.optimizer.state_dict()
            if hasattr(t, 'system') and hasattr(t.system, 'state_dict'):
                state['system'] = t.system.state_dict()

        # Save locally
        torch.save(state, local_path)
        log.info(f"💾 Checkpoint saved locally: {local_path.name}")

        # Upload to HuggingFace Hub if token available
        if api and HF_TOKEN:
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=f"checkpoints/checkpoint_{tag}.pt",
                    repo_id=CHECKPOINT_REPO,
                    repo_type='dataset',
                    token=HF_TOKEN
                )
                log.info(f"☁️ Checkpoint uploaded: checkpoints/checkpoint_{tag}.pt")
            except Exception as e:
                log.warning(f"HF upload failed (continuing locally): {e}")

        _write_status({
            'last_checkpoint': tag,
            'last_checkpoint_step': step,
            'last_checkpoint_time': time.time()
        })

    except Exception as e:
        log.error(f"Checkpoint save failed: {e}")
        traceback.print_exc()

def _load_latest_checkpoint(ctrl):
    """Resume from latest checkpoint"""
    if not ctrl:
        return
        
    try:
        log.info("🔍 Searching for latest checkpoint...")
        
        # First check local checkpoints
        local_ckpts = list(CHECKPOINT_DIR.glob('checkpoint_*.pt'))
        if local_ckpts:
            # Sort by modification time
            latest_local = max(local_ckpts, key=lambda x: x.stat().st_mtime)
            log.info(f"📁 Found local checkpoint: {latest_local.name}")
            _load_checkpoint_file(ctrl, latest_local)
            return

        # Then try HuggingFace Hub if available
        if api and HF_TOKEN:
            try:
                files = api.list_repo_files(CHECKPOINT_REPO, repo_type='dataset', token=HF_TOKEN)
                ckpts = sorted([f for f in files if f.startswith('checkpoints/') and f.endswith('.pt')])

                if ckpts:
                    latest = ckpts[-1]
                    log.info(f"📥 Downloading from HF Hub: {latest}")

                    local_path = api.hf_hub_download(
                        CHECKPOINT_REPO, latest,
                        repo_type='dataset', token=HF_TOKEN,
                        local_dir=str(CHECKPOINT_DIR)
                    )
                    _load_checkpoint_file(ctrl, Path(local_path))
                    return
            except Exception as e:
                log.warning(f"HF checkpoint load failed: {e}")

        log.info("📭 No checkpoints found — starting fresh training")

    except Exception as e:
        log.warning(f"Could not load checkpoint (starting fresh): {e}")

def _load_checkpoint_file(ctrl, checkpoint_path):
    """Load checkpoint from file"""
    try:
        state = torch.load(checkpoint_path, map_location='cpu')
        step = state.get('step', 0)

        # Restore trainer state
        if hasattr(ctrl, 'trainer') and ctrl.trainer is not None:
            t = ctrl.trainer
            if 'optimizer' in state and hasattr(t, 'optimizer') and t.optimizer:
                t.optimizer.load_state_dict(state['optimizer'])
            if 'system' in state and hasattr(t, 'system') and hasattr(t.system, 'load_state_dict'):
                t.system.load_state_dict(state['system'])
            if hasattr(t, 'training_step'):
                t.training_step = step

        log.info(f"✅ Resumed from step {step} ({checkpoint_path.name})")
        _write_status({
            'resumed_from': checkpoint_path.name,
            'resumed_step': step,
            'resumed_time': time.time()
        })

    except Exception as e:
        log.error(f"Checkpoint load error: {e}")

# ── Main service ──────────────────────────────────────────────────────────────

def main():
    """Main service entry point"""
    global controller

    log.info("=" * 60)
    log.info("  K1RL QUASAR ENGINE — HuggingFace Spaces")
    log.info("=" * 60)
    log.info(f"  Base Directory: {BASE_DIR}")
    log.info(f"  Checkpoint Directory: {CHECKPOINT_DIR}")
    log.info(f"  HuggingFace Token: {'✓' if HF_TOKEN else '✗'}")
    log.info("=" * 60)

    # Start RAM monitor
    ram_thread = threading.Thread(target=_ram_monitor_thread, daemon=True)
    ram_thread.start()
    log.info("✅ RAM monitor started")

    # Import QUASAR system
    log.info("⏳ Importing QUASAR system...")
    _write_status({'status': 'importing'})
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, str(BASE_DIR))
        
        # Import the main QUASAR controller
        from quasar_main4 import QuantumMetaController
        log.info("✅ QUASAR system imported successfully")
    except Exception as e:
        log.critical(f"❌ Failed to import QUASAR: {e}")
        traceback.print_exc()
        _write_status({'status': 'import_failed', 'error': str(e)})
        sys.exit(1)

    # Initialize controller
    log.info("⏳ Initializing QuantumMetaController...")
    _write_status({'status': 'initializing'})

    try:
        controller = QuantumMetaController(reset_interval_minutes=60000000)
        log.info("✅ QuantumMetaController initialized")
        _write_status({'status': 'initialized'})
    except Exception as e:
        log.critical(f"❌ Initialization failed: {e}")
        traceback.print_exc()
        _clear_all_caches()
        _write_status({'status': 'init_failed', 'error': str(e)})
        sys.exit(1)

    # Load checkpoint
    _load_latest_checkpoint(controller)
    _write_status({'status': 'running', 'started_time': time.time()})

    log.info("🚀 Starting training loop...")

    # ── Main training loop with crash recovery ───────────────────────────────
    consecutive_errors = 0
    start_time = time.time()

    while not shutdown_requested:
        try:
            # Update uptime
            uptime = time.time() - start_time
            _write_status({'uptime_seconds': uptime})

            # Run training
            if hasattr(controller, 'run_forever'):
                controller.run_forever()
            else:
                # Fallback: sleep and check status
                time.sleep(60)

            # If run_forever returns normally, restart
            if not shutdown_requested:
                log.warning("run_forever() returned — restarting loop in 5s")
                time.sleep(5)

        except MemoryError as e:
            log.error(f"🚨 OOM Error: {e} — clearing caches and restarting")
            _clear_all_caches()
            _save_checkpoint(controller, emergency=True)
            consecutive_errors += 1
            time.sleep(10)

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down gracefully")
            break

        except Exception as e:
            consecutive_errors += 1
            log.error(f"❌ Training error #{consecutive_errors}: {e}")
            traceback.print_exc()

            # Save emergency checkpoint
            try:
                _save_checkpoint(controller, emergency=True)
            except Exception as ckpt_e:
                log.error(f"Emergency checkpoint failed: {ckpt_e}")

            # Clear caches
            _clear_all_caches()

            # Check if too many consecutive errors
            if consecutive_errors >= 5:
                log.critical("💥 5 consecutive errors — exiting for supervisor restart")
                _write_status({'status': 'crashed', 'consecutive_errors': consecutive_errors})
                sys.exit(1)

            # Exponential backoff
            backoff = min(60, consecutive_errors * 10)
            log.info(f"⏳ Restarting in {backoff}s... (error #{consecutive_errors})")
            _write_status({'status': 'error_recovery', 'restart_in': backoff})
            time.sleep(backoff)

        else:
            consecutive_errors = 0

    log.info("🛑 QUASAR Engine stopped")
    _write_status({'status': 'stopped', 'stopped_time': time.time()})

if __name__ == '__main__':
    main()
    
