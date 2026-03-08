# ============================================================================
# K1RL QUASAR — Redis Configuration (HuggingFace Spaces)
# Container-only environment: Redis runs locally within the Space
# ============================================================================

import os

# ── Environment detection ──────────────────────────────────────────────────
IS_HUGGINGFACE_SPACE = os.environ.get('SPACE_ID') is not None
IS_LOCAL_DEV = not IS_HUGGINGFACE_SPACE

# ── Credentials ─────────────────────────────────────────────────────────────
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "k1rl_099a0c008e32300dc3c14189")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

# ── Host selection ─────────────────────────────────────────────────────────
if IS_HUGGINGFACE_SPACE:
    # HuggingFace Spaces: Redis runs inside container
    REDIS_HOST = "127.0.0.1"
elif IS_LOCAL_DEV:
    # Local development
    REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
else:
    # Production server fallback
    REDIS_HOST = os.environ.get("REDIS_HOST", "145.241.100.50")

# ── Connection URLs ────────────────────────────────────────────────────────
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

# Database selection
REDIS_DB_METRICS = 0    # Real-time metrics
REDIS_DB_VISITORS = 1   # Visitor tracking  
REDIS_DB_CACHE = 2      # General cache
REDIS_DB_FEATURES = 3   # Feature data
REDIS_DB_REWARDS = 4    # Reward signals

# Connection pool settings
REDIS_POOL_SETTINGS = {
    'max_connections': 20,
    'retry_on_timeout': True,
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'health_check_interval': 30
}

# ── Debug info ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("K1RL QUASAR — Redis Configuration")
    print("=" * 50)
    print(f"Environment: {'HuggingFace Spaces' if IS_HUGGINGFACE_SPACE else 'Local/Production'}")
    print(f"Redis Host: {REDIS_HOST}")
    print(f"Redis Port: {REDIS_PORT}")
    print(f"Redis URL: {REDIS_URL}")
    print(f"Databases: metrics={REDIS_DB_METRICS}, visitors={REDIS_DB_VISITORS}, cache={REDIS_DB_CACHE}")
    print("=" * 50)

    
