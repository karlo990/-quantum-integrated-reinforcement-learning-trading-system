# -*- coding: utf-8 -*-
"""
K1RL QUANT - INSTITUTIONAL REWARDS SYSTEM v5.2.1-CRASH500
HuggingFace Spaces Edition - Maximum Performance

CRASH500 NAMESPACE ISOLATION:
✅ All channels prefixed with "crash500:" — zero cross-talk with v25
✅ Uses DB 8/9 (features/rewards) — separate from v24's DB 0-4
✅ Imports from redis_config_crash500 for CRASH500-specific configuration

CRITICAL FIX (v5.2.1):
✅ FIXED: asyncio.get_event_loop() from listener thread returned WRONG loop
   → Reward tasks silently dropped (never scheduled)
   → Now stores loop reference via asyncio.get_running_loop() in start()
✅ All v5.2.0 fixes retained

CRITICAL FIX (v5.2.0):
✅ REMOVED duplicate RedisAblyClient - uses redis_connection_manager.RedisAblyClient
✅ Added connection health monitoring with auto-reconnection
✅ Bounded reward task pool (prevents coroutine leak)
✅ Deriv WebSocket auto-reconnection loop
✅ Pub/sub heartbeat detection (detects silent disconnects)

PREVIOUS OPTIMIZATIONS (v5.1.0):
✅ Proper async price streaming with reconnection
✅ LRU cache for price data with TTL
✅ O(1) signal tracking with hash maps
✅ Batch processing with backpressure
✅ Connection health monitoring
✅ Memory-efficient deque buffers
✅ HuggingFace Spaces compatibility
✅ Container-safe logging and paths
"""

import asyncio
import logging
import sys
import time
import json
import traceback
import ssl
import websockets
import os
from datetime import datetime, timezone
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Deque
from functools import lru_cache
import numpy as np
from pathlib import Path

# Async compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ============================================================================
# ✅ FIX #1: Import the ROBUST RedisAblyClient from redis_connection_manager
# instead of defining a broken local version with no reconnection
# ============================================================================
import redis
from redis_config_crash500 import (
    REDIS_URL, REDIS_PASSWORD,
    REDIS_DB_FEATURES, REDIS_DB_REWARDS,
    CHANNEL_PREFIX, prefixed_channel, QUASAR_VERSION
)
from redis_connection_manager import (
    RedisAblyClient,
    RedisMessage,
    DedicatedRedisConnectionManager,
    diagnose_redis_connection,
    IS_HF_SPACES
)

# ============================================================================
# HUGGINGFACE SPACES CONFIGURATION
# ============================================================================

# ✅ FIXED: Environment variable for API key
DERIV_API_KEY = os.environ.get('DERIV_API_KEY', '1KJKxIJKR8LCyKB')
DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

SYMBOL_MAP = {
    "Volatility 25 Index": "R_25",
    "Crash 500 Index": "CRASH500",
    "Volatility 100 Index": "R_100",
    "Volatility 50 Index": "R_50",
}

SYMBOL = "Crash 500 Index"
DERIV_SYMBOL = "CRASH500"

# Ably Configuration (now Redis channels — CRASH500 NAMESPACED)
ABLY_SIGNAL_CHANNEL = prefixed_channel("final_signals")      # → "crash500:final_signals"
ABLY_REWARD_CHANNEL = prefixed_channel("rewards")             # → "crash500:rewards"
ABLY_BATCH_CHANNEL = prefixed_channel("reward-batches")       # → "crash500:reward-batches"

ACTION_MAP = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
ACTION_REVERSE = {'BUY': 0, 'SELL': 1, 'HOLD': 2}

# Performance tuning
EVALUATION_DELAY = 60  # seconds
BATCH_SIZE = 10
PRICE_CACHE_TTL = 5.0  # seconds - price considered stale after this
MAX_TRACKED_SIGNALS = 10000
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_ATTEMPTS = 10

# ✅ FIX #2: Bounded concurrency for reward calculation tasks
MAX_CONCURRENT_REWARD_TASKS = 200  # Prevents unbounded coroutine growth

# ✅ FIXED: Container-safe logging
BASE_DIR = Path('/home/user/app')
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REWARDS] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / 'rewards.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

if IS_HF_SPACES:
    logger.info("🤗 HuggingFace Spaces environment detected")

# ============================================================================
# HIGH-PERFORMANCE DATA STRUCTURES (Unchanged - Already Optimized)
# ============================================================================

class PriceData:
    """Immutable price snapshot with timestamp - optimized with __slots__"""
    __slots__ = ('bid', 'ask', 'last', 'timestamp', 'epoch')
    
    def __init__(self, bid: float, ask: float, last: float, timestamp: float, epoch: int):
        self.bid = bid
        self.ask = ask
        self.last = last
        self.timestamp = timestamp
        self.epoch = epoch
    
    @property
    def age(self) -> float:
        return time.time() - self.timestamp
    
    @property
    def is_stale(self) -> bool:
        return self.age > PRICE_CACHE_TTL
    
    def get_price(self, action: str) -> float:
        return self.ask if action == "BUY" else self.bid

class TrackedSignal:
    """Tracked signal with minimal memory footprint - optimized with __slots__"""
    __slots__ = ('signal_key', 'action', 'entry_price', 'timestamp', 'agent')
    
    def __init__(self, signal_key: str, action: str, entry_price: float, timestamp: float, agent: str = "unknown"):
        self.signal_key = signal_key
        self.action = action
        self.entry_price = entry_price
        self.timestamp = timestamp
        self.agent = agent

class TTLCache:
    """O(1) cache with time-to-live expiration"""
    
    __slots__ = ('_cache', '_ttl', '_max_size')
    
    def __init__(self, ttl: float = 5.0, max_size: int = 1000):
        self._cache: OrderedDict = OrderedDict()
        self._ttl = ttl
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None
        return value
    
    def set(self, key: str, value: Any) -> None:
        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (value, time.time())
        # Move to end (most recently used)
        self._cache.move_to_end(key)
    
    def __len__(self) -> int:
        return len(self._cache)

# ============================================================================
# OPTIMIZED DERIV BRIDGE - HuggingFace Spaces Edition
# ============================================================================

class DerivStreamingBridge:
    """
    High-performance Deriv WebSocket bridge - HuggingFace Spaces optimized.
    
    Features:
    - Auto-reconnection with exponential backoff
    - Container-safe error handling
    - HF Spaces compatibility
    """
    
    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.is_authorized = False
        
        # Price cache with TTL
        self._price_cache: Dict[str, PriceData] = {}
        self._cache_lock = asyncio.Lock()
        
        # Connection management
        self._reconnect_attempts = 0
        self._last_tick_time: Dict[str, float] = {}
        self._streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        
        # HF Spaces features
        self._hf_spaces_mode = IS_HF_SPACES
        self._max_connection_attempts = 10
        
        # Stats
        self.ticks_received = 0
        self.reconnections = 0
    
    async def connect(self) -> bool:
        """Connect and authorize to Deriv with HF Spaces resilience"""
        # V9.0 FIX: Recreate lock on the RUNNING loop
        self._cache_lock = asyncio.Lock()
        
        try:
            self._reconnect_attempts += 1
            logger.info(f"🔄 Connecting to Deriv WebSocket... (attempt {self._reconnect_attempts})")
            
            # Connection attempt limit
            if self._reconnect_attempts > self._max_connection_attempts:
                logger.error("❌ Max connection attempts exceeded")
                return False
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    DERIV_WS_URL,
                    ssl=ssl_context,
                    ping_interval=25,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**20
                ),
                timeout=15.0 if IS_HF_SPACES else 30.0
            )
            
            # Authorize
            await self.ws.send(json.dumps({"authorize": DERIV_API_KEY}))
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            if 'authorize' in data:
                self.is_connected = True
                self.is_authorized = True
                self._reconnect_attempts = 0
                balance = data['authorize'].get('balance', 0)
                logger.info(f"✅ Deriv connected | Balance: ${balance:.2f}")
                
                # Start streaming
                await self._start_streaming()
                return True
            else:
                logger.error(f"❌ Auth failed: {data.get('error', 'Unknown')}")
                return False
                
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Connection timeout (attempt {self._reconnect_attempts})")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Connection error: {e}")
            return False
    
    
    async def _start_streaming(self):
        """Start price streaming"""
        self._stream_task = asyncio.create_task(self._real_price_stream())
        self._streaming = True
        logger.info(f"📡 Streaming started")
    
    async def _real_price_stream(self):
        """Real price streaming from Deriv WebSocket"""
        try:
            # Subscribe to ticks
            await self.ws.send(json.dumps({"ticks": DERIV_SYMBOL}))
            logger.info(f"📡 Subscribed to {DERIV_SYMBOL}")
            
            while self.is_connected:
                try:
                    data = await self.ws.recv()
                    json_data = json.loads(data)
                    
                    if 'tick' in json_data:
                        await self._process_tick(json_data['tick'])
                        self.ticks_received += 1
                
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("📡 WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"❌ Stream error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"❌ Real price streaming error: {e}")
    
    
    async def _process_tick(self, tick_data):
        """Process incoming tick data"""
        try:
            price = float(tick_data['quote'])
            epoch = int(tick_data['epoch'])
            
            # Create price data
            price_data = PriceData(
                bid=price - 0.0005,
                ask=price + 0.0005,
                last=price,
                timestamp=time.time(),
                epoch=epoch
            )
            
            # Update cache
            async with self._cache_lock:
                self._price_cache[DERIV_SYMBOL] = price_data
                self._last_tick_time[DERIV_SYMBOL] = time.time()
                
        except Exception as e:
            logger.error(f"❌ Tick processing error: {e}")
    
    async def _reconnect(self) -> bool:
        """Reconnect with exponential backoff"""
        self.is_connected = False
        self._streaming = False
        
        if self._stream_task:
            self._stream_task.cancel()
        
        if self.ws:
            await self.ws.close()
        
        delay = min(60, RECONNECT_DELAY * (2 ** min(self._reconnect_attempts, 5)))
        logger.info(f"🔄 Reconnecting in {delay}s...")
        await asyncio.sleep(delay)
        
        success = await self.connect()
        if success:
            self.reconnections += 1
            logger.info(f"✅ Reconnected successfully (#{self.reconnections})")
        
        return success
    
    async def get_current_price(self, symbol: str = DERIV_SYMBOL) -> Optional[PriceData]:
        """Get current cached price"""
        async with self._cache_lock:
            price_data = self._price_cache.get(symbol)
            
            if price_data and not price_data.is_stale:
                return price_data
            
            return None
    
    def get_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            'connected': self.is_connected,
            'authorized': self.is_authorized,
            'ticks_received': self.ticks_received,
            'reconnections': self.reconnections,
            'streaming': self._streaming
        }
    
    async def shutdown(self):
        """Shutdown with cleanup"""
        logger.info("🛑 Shutting down Deriv bridge...")
        self.is_connected = False
        self._streaming = False
        
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        
        logger.info("✅ Deriv bridge shutdown complete")

# ============================================================================
# REWARD CALCULATION COMPONENTS (Updated for HF Spaces)
# ============================================================================

class RewardNormalizer:
    def __init__(self, base_multiplier=1000):
        self.base_multiplier = base_multiplier
        self.volatility_buffer = deque(maxlen=100)
        
    def normalize(self, entry_price, exit_price, action):
        """Normalize reward based on action and price movement"""
        if entry_price <= 0:
            return 0, "invalid", 0, 0
        
        # Calculate raw basis points
        raw_bps = ((exit_price - entry_price) / entry_price) * 10000
        
        # Apply action multiplier
        if action == "BUY":
            directional_bps = raw_bps
        elif action == "SELL":
            directional_bps = -raw_bps
        else:  # HOLD
            directional_bps = -abs(raw_bps) * 0.1  # Small penalty for holding
        
        # Simple regime detection
        regime = "normal"
        if abs(raw_bps) > 50:
            regime = "high_vol"
        elif abs(raw_bps) < 5:
            regime = "low_vol"
        
        # Normalize to [-1, 1] range
        normalized = np.tanh(directional_bps / 100)
        confidence = min(abs(directional_bps) / 20, 1.0)
        
        return normalized, regime, confidence, raw_bps

class AgentTracker:
    """Track agent performance and streaks"""
    
    def __init__(self):
        self.agents = {}
        self.reset()
    
    def reset(self):
        """Reset tracking data"""
        self.agents = {
            "5s": {"count": 0, "action": None, "cycles": 0},
            "15s": {"count": 0, "action": None, "cycles": 0},
            "30s": {"count": 0, "action": None, "cycles": 0},
            "1m": {"count": 0, "action": None, "cycles": 0},
            "2m": {"count": 0, "action": None, "cycles": 0},
            "5m": {"count": 0, "action": None, "cycles": 0},
            "10m": {"count": 0, "action": None, "cycles": 0},
            "15m": {"count": 0, "action": None, "cycles": 0}
        }
    
    def update(self, agent, action):
        """Update agent tracking, return True if cycle completed"""
        if agent not in self.agents:
            return False
        
        if self.agents[agent]["action"] == action:
            self.agents[agent]["count"] += 1
        else:
            if self.agents[agent]["count"] >= 3:  # Cycle completion
                self.agents[agent]["cycles"] += 1
                self.agents[agent]["count"] = 1
                self.agents[agent]["action"] = action
                return True
            else:
                self.agents[agent]["count"] = 1
                self.agents[agent]["action"] = action
        
        return False
    
    def get_info(self, agent):
        """Get agent tracking info"""
        return self.agents.get(agent, {"count": 0, "action": None, "cycles": 0})

# ============================================================================
# MAIN REWARDS ENGINE - HuggingFace Spaces Edition v5.2.0
# ============================================================================

class RewardsEngine:
    """
    Main rewards calculation engine with HF Spaces compatibility.
    
    v5.2.0 FIXES:
    - Uses robust RedisAblyClient from redis_connection_manager.py
    - Bounded reward task pool (semaphore)
    - Connection health heartbeat
    - Deriv auto-reconnection loop
    """
    
    def __init__(self):
        # Core components
        self.normalizer = RewardNormalizer()
        self.agent_tracker = AgentTracker()
        
        # Tracking
        self._tracked: Dict[str, TrackedSignal] = {}
        self._processed_keys = TTLCache(ttl=300)  # 5 minutes
        
        # Batch processing
        self._batch: List[Dict] = []
        self._batch_lock: Optional[asyncio.Lock] = None
        self._last_batch_time = time.time()
        
        # ✅ FIX #3: Bounded task pool for reward calculations
        self._reward_semaphore: Optional[asyncio.Semaphore] = None
        self._active_reward_tasks = 0
        
        # Statistics
        self.signals_received = 0
        self.rewards_sent = 0
        self.correct = 0
        self.wrong = 0
        
        # ✅ FIX #4: Track last signal time for health monitoring
        self._last_signal_time = 0.0
        self._last_heartbeat_time = 0.0
        self._connection_healthy = True
        
        # ✅ FIX v5.2.1: Store event loop reference for thread→asyncio bridge
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Connections
        self.ably_realtime: Optional[RedisAblyClient] = None
        self.signal_channel = None
        self.reward_channel_batch = None
        self.reward_channel_individual = None
        
        # Control
        self._shutdown: Optional[asyncio.Event] = None
    
    async def initialize(self) -> bool:
        """Initialize connections and channels"""
        try:
            logger.info("📡 Connecting to Redis (CRASH500 namespace)...")
            
            # ✅ FIX #5: Use the ROBUST RedisAblyClient from redis_connection_manager.py
            # This version has: blocking listener, auto-reconnection, health monitoring
            # CRASH500: Uses DB 8 (features) — isolated from v25's DB 0
            self.ably_realtime = RedisAblyClient(
                redis_url=REDIS_URL,
                password=REDIS_PASSWORD,
                use_streams=True,
                database=REDIS_DB_FEATURES  # CRASH500: DB 8 (was DB 0 in v25)
            )
            
            # Set up channels (already prefixed via constants above)
            self.signal_channel = self.ably_realtime.channels.get(ABLY_SIGNAL_CHANNEL)
            self.reward_channel_batch = self.ably_realtime.channels.get(ABLY_BATCH_CHANNEL)
            self.reward_channel_individual = self.ably_realtime.channels.get(ABLY_REWARD_CHANNEL)
            
            logger.info(f"✅ Redis channels initialized (CRASH500 — prefix='{CHANNEL_PREFIX}', DB={REDIS_DB_FEATURES})")
            logger.info(f"   Signal:  {ABLY_SIGNAL_CHANNEL}")
            logger.info(f"   Rewards: {ABLY_REWARD_CHANNEL}")
            logger.info(f"   Batches: {ABLY_BATCH_CHANNEL}")
            
            # Initialize Deriv bridge
            logger.info("🔄 Connecting to Deriv...")
            success = await deriv_bridge.connect()
            if not success:
                logger.error("❌ Deriv connection failed")
                return False
            
            logger.info("✅ All connections established")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            return False
    
    def _extract_agent(self, signal_key: str) -> str:
        """Extract agent timeframe from signal key"""
        for tf in ['15m', '10m', '5m', '2m', '1m', '30s', '15s', '5s']:
            if tf in signal_key:
                return tf
        return "unknown"
    
    async def _get_price(self, action: str) -> Optional[float]:
        """Get current price for action"""
        try:
            price_data = await deriv_bridge.get_current_price()
            if price_data:
                return price_data.get_price(action)
            return None
        except Exception as e:
            logger.error(f"❌ Price fetch error: {e}")
            return None
    
    def _on_signal(self, message: RedisMessage):
        """
        Handle incoming signal - FIXED for RedisAblyClient V10.1 format.
        
        ✅ FIX #6: This callback is now called by the BLOCKING listener thread
        in redis_connection_manager.py's RedisAblyClient, NOT the broken polling
        listener from the old Rewards.py RedisAblyClient.
        
        The RedisAblyClient V10.1 delivers RedisMessage objects, not raw dicts.
        """
        try:
            self.signals_received += 1
            self._last_signal_time = time.time()
            
            # RedisMessage from redis_connection_manager has .data attribute
            data = message.data if isinstance(message, RedisMessage) else message
            
            # Handle nested data (envelope format: {"event": "message", "data": {...}})
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            
            # Parse if string
            if isinstance(data, str):
                data = json.loads(data)
            
            # Extract fields
            action = data.get('final_action', data.get('action', '')).upper()
            signal_keys = data.get('signal_keys', [])
            entry_price = data.get('price', 0.0)
            
            if action not in ['BUY', 'SELL']:
                logger.warning(f"⚠️ Invalid action: {action}")
                return
            
            if not entry_price or entry_price == 0.0:
                logger.warning(f"⚠️ No entry price in signal: {entry_price}")
                return
            
            # Log signal received
            logger.info(f"🔔 [SIGNAL] {action} @ {entry_price:.5f} | Keys: {len(signal_keys)} | Loop: {'✅' if self._loop and self._loop.is_running() else '❌'}")
            
            # Ensure signal_keys is list
            if not isinstance(signal_keys, list):
                signal_keys = [str(signal_keys)]
            
            # Track each signal
            for key in signal_keys[:8]:  # Limit to 8 signals
                key = str(key)
                
                # Skip duplicates - O(1)
                if key in self._tracked or self._processed_keys.get(key):
                    continue
                
                # Memory bound check
                if len(self._tracked) >= MAX_TRACKED_SIGNALS:
                    oldest = min(self._tracked.items(), key=lambda x: x[1].timestamp)
                    del self._tracked[oldest[0]]
                
                agent = self._extract_agent(key)
                
                # Track signal
                self._tracked[key] = TrackedSignal(
                    signal_key=key,
                    action=action,
                    entry_price=entry_price,
                    timestamp=time.time(),
                    agent=agent
                )
                
                logger.debug(f"✅ [TRACKING] {key} | {action} @ {entry_price:.5f}")
                
                # Agent streak tracking
                if agent == "10m":
                    if self.agent_tracker.update(agent, action):
                        info = self.agent_tracker.get_info("10m")
                        logger.info(f"🔥 [10m CYCLE #{info['cycles']}] {action} x{info['count']}")
                
                # ✅ FIX #7: Schedule reward via bounded task pool
                # Uses the event loop from the main thread
                self._schedule_reward(key)
                
        except Exception as e:
            logger.error(f"❌ Signal processing error: {e}")
            traceback.print_exc()
    
    def _schedule_reward(self, signal_key: str):
        """
        Schedule reward calculation on the event loop.
        
        ✅ FIX v5.2.1: Since _on_signal is called from a THREAD (the RedisAblyClient
        listener thread), we MUST use the stored loop reference from start().
        asyncio.get_event_loop() from a non-main thread returns a NEW loop
        (not the running one), silently dropping all reward tasks.
        """
        try:
            if self._loop is not None and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._bounded_calculate_reward(signal_key), self._loop
                )
            else:
                logger.warning(f"⚠️ Event loop not available (loop={self._loop}), reward for {signal_key} dropped")
        except RuntimeError as e:
            logger.warning(f"⚠️ Cannot schedule reward: {e}")
    
    async def _bounded_calculate_reward(self, signal_key: str):
        """
        ✅ FIX #8: Bounded reward calculation with semaphore.
        Prevents unbounded coroutine growth from 60s sleep per signal.
        """
        if self._reward_semaphore is None:
            return
            
        async with self._reward_semaphore:
            self._active_reward_tasks += 1
            try:
                await self._calculate_reward(signal_key)
            finally:
                self._active_reward_tasks -= 1
    
    async def _calculate_reward(self, signal_key: str) -> None:
        """Calculate reward after delay"""
        # Wait for evaluation period
        await asyncio.sleep(EVALUATION_DELAY)
        
        # Get signal
        signal = self._tracked.pop(signal_key, None)
        if not signal:
            return
        
        # Mark as processed
        self._processed_keys.set(signal_key, True)
        
        # ✅ BUG FIX 3: Retry price fetch up to 3 times with 5s backoff
        # Previously, a single failed price fetch would silently drop the reward forever
        exit_price = None
        for attempt in range(3):
            exit_price = await self._get_price(signal.action)
            if exit_price:
                break
            logger.warning(f"⚠️ Price fetch attempt {attempt+1}/3 failed for {signal_key}")
            if not deriv_bridge.is_connected:
                logger.warning(f"⚠️ Deriv disconnected, triggering reconnect...")
                await deriv_bridge._reconnect()
            await asyncio.sleep(5)
        
        if not exit_price:
            logger.error(f"❌ All price fetch attempts failed for {signal_key} — reward dropped permanently")
            return
        
        # Calculate reward
        normalized, regime, confidence, raw_bps = self.normalizer.normalize(
            signal.entry_price, exit_price, signal.action
        )
        
        # Track accuracy
        if normalized > 0:
            self.correct += 1
            correct_action = ACTION_REVERSE[signal.action]
        else:
            self.wrong += 1
            correct_action = 1 - ACTION_REVERSE.get(signal.action, 0)
        
        # Log with price difference
        price_diff = exit_price - signal.entry_price
        logger.info(
            f"[REWARD] {signal.action} | "
            f"entry={signal.entry_price:.2f} → exit={exit_price:.2f} (Δ{price_diff:+.2f}) | "
            f"reward={normalized:+.4f} | {signal_key[:25]}"
        )
        
        # Add to batch
        await self._add_to_batch({
            "signal_key": signal_key,
            "reward": normalized,
            "entry_price": signal.entry_price,
            "exit_price": exit_price,
            "executed_action": signal.action,
            "correct_action": correct_action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price_source": "deriv_streaming_v5_live",
            "platform": "huggingface-spaces" if IS_HF_SPACES else "local"
        })
    
    async def _add_to_batch(self, reward_data: Dict) -> None:
        """Add reward to batch with backpressure"""
        async with self._batch_lock:
            self._batch.append(reward_data)
            self.rewards_sent += 1
            
            # Send batch if full or timeout
            if len(self._batch) >= BATCH_SIZE:
                await self._send_batch()
    
    async def _send_batch(self) -> None:
        """Send reward batch via Redis pub/sub"""
        if not self._batch:
            return
        
        try:
            batch_data = {
                "rewardz": self._batch.copy(),
                "batch_id": f"batch_{int(time.time() * 1000)}",
                "batch_size": len(self._batch),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price_source": "deriv_streaming_v5_live",
                "platform": "huggingface-spaces" if IS_HF_SPACES else "local"
            }
            
            # ✅ FIX #10: Use synchronous publish for RedisAblyClient V10.1
            # The RedisAblyChannel.publish() in redis_connection_manager.py is async,
            # but we can also use the underlying redis client directly for reliability.
            await self.reward_channel_batch.publish("reward-batch", batch_data)
            
            # Send individual (for compatibility)
            for r in self._batch:
                try:
                    await self.reward_channel_individual.publish("new-reward", r)
                except Exception:
                    pass
            
            logger.info(f"📤 Sent batch of {len(self._batch)} rewards | Active tasks: {self._active_reward_tasks}")
            self._batch.clear()
            self._last_batch_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Batch send error: {e}")
            self._batch.clear()
    
    async def _health_monitor_loop(self):
        """
        ✅ FIX #11: Health monitor that detects silent disconnections.
        
        If no signals received for 5 minutes AND we expect signals to be flowing,
        trigger diagnostics and alert.
        """
        SIGNAL_TIMEOUT = 300  # 5 minutes without signals = problem
        DERIV_CHECK_INTERVAL = 60  # Check Deriv every 60s
        
        while not self._shutdown.is_set():
            await asyncio.sleep(30)
            
            now = time.time()
            
            # Check Redis connection health
            if self.ably_realtime:
                try:
                    # The robust RedisAblyClient has connection state
                    redis_state = self.ably_realtime.connection.state
                    if redis_state != 'connected':
                        logger.warning(f"⚠️ [HEALTH] Redis state: {redis_state}")
                        self._connection_healthy = False
                except Exception as e:
                    logger.warning(f"⚠️ [HEALTH] Redis check failed: {e}")
            
            # Check signal flow
            if self._last_signal_time > 0:
                signal_age = now - self._last_signal_time
                if signal_age > SIGNAL_TIMEOUT:
                    logger.warning(
                        f"⚠️ [HEALTH] No signals for {signal_age:.0f}s! "
                        f"Last signal at {datetime.fromtimestamp(self._last_signal_time).strftime('%H:%M:%S')}. "
                        f"Possible pub/sub disconnection."
                    )
                    self._connection_healthy = False
            
            # Check Deriv WebSocket
            if not deriv_bridge.is_connected or not deriv_bridge._streaming:
                logger.warning("⚠️ [HEALTH] Deriv disconnected, attempting reconnect...")
                success = await deriv_bridge._reconnect()
                if success:
                    logger.info("✅ [HEALTH] Deriv reconnected")
                else:
                    logger.error("❌ [HEALTH] Deriv reconnection failed")
    
    async def _status_loop(self) -> None:
        """Periodic status reporting"""
        while not self._shutdown.is_set():
            await asyncio.sleep(60)
            
            total = self.correct + self.wrong
            win_rate = (100 * self.correct / max(1, total))
            bridge_stats = deriv_bridge.get_stats()
            
            # ✅ FIX: Include health info in status
            signal_age = time.time() - self._last_signal_time if self._last_signal_time > 0 else -1
            
            logger.info(
                f"[STATUS] Signals={self.signals_received} | "
                f"Rewards={self.rewards_sent} | "
                f"WinRate={win_rate:.1f}% | "
                f"Tracked={len(self._tracked)} | "
                f"ActiveTasks={self._active_reward_tasks} | "
                f"SignalAge={signal_age:.0f}s | "
                f"Ticks={bridge_stats['ticks_received']} | "
                f"Healthy={'✅' if self._connection_healthy else '❌'} | "
                f"Platform={'HF Spaces' if IS_HF_SPACES else 'Local'}"
            )
            
            # Flush any pending batch
            async with self._batch_lock:
                if self._batch and time.time() - self._last_batch_time > 30:
                    await self._send_batch()
    
    async def start(self) -> None:
        """Start the rewards engine"""
        # V9.0 FIX: Recreate asyncio primitives on the running loop
        self._batch_lock = asyncio.Lock()
        self._shutdown = asyncio.Event()
        self._reward_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REWARD_TASKS)
        
        # ✅ FIX v5.2.1: Store event loop reference BEFORE anything else
        # This is CRITICAL because _on_signal runs in the Redis listener THREAD
        # and needs to schedule coroutines on THIS loop
        self._loop = asyncio.get_running_loop()
        
        logger.info("=" * 70)
        logger.info("K1RL QUANT - INSTITUTIONAL REWARDS v5.2.1-CRASH500")
        logger.info("HuggingFace Spaces Edition 🤗 (CRASH500 NAMESPACE ISOLATION))")
        logger.info("=" * 70)
        
        if IS_HF_SPACES:
            logger.info("🤗 Running in HuggingFace Spaces environment")
            logger.info("   - CRASH500 channel prefix: '%s'" % CHANNEL_PREFIX)
            logger.info("   - CRASH500 databases: features=DB%d, rewards=DB%d" % (REDIS_DB_FEATURES, REDIS_DB_REWARDS))
            logger.info("   - Robust RedisAblyClient V10.1 (blocking listener)")
            logger.info("   - Bounded reward task pool (max=%d)" % MAX_CONCURRENT_REWARD_TASKS)
            logger.info("   - Connection health monitoring")
            logger.info("   - ✅ Event loop captured for thread→asyncio bridge")
        
        if not await self.initialize():
            raise Exception("Initialization failed")
        
        # ✅ FIX #12: Subscribe using RedisAblyClient V10.1 format
        # The V10.1 RedisAblyChannel.subscribe() expects (event_name, callback)
        # where callback receives a RedisMessage object (not raw dict)
        await self.signal_channel.subscribe("message", self._on_signal)
        logger.info(f"✅ Subscribed to {ABLY_SIGNAL_CHANNEL}:message (CRASH500 namespace, robust V10.1 listener)")
        logger.info(f"   Event loop captured: {self._loop is not None} | Running: {self._loop.is_running() if self._loop else False}")
        
        # Start health monitor
        health_task = asyncio.create_task(self._health_monitor_loop())
        
        # Start status loop
        status_task = asyncio.create_task(self._status_loop())
        
        try:
            # Main loop
            while not self._shutdown.is_set():
                await asyncio.sleep(1)
        finally:
            health_task.cancel()
            status_task.cancel()
    
    async def shutdown(self) -> None:
        """Clean shutdown"""
        logger.info("🛑 Shutting down rewards engine...")
        self._shutdown.set()
        
        # Flush batch
        async with self._batch_lock:
            if self._batch:
                await self._send_batch()
        
        # Close connections
        if self.ably_realtime:
            self.ably_realtime.close()
        await deriv_bridge.shutdown()
        
        logger.info("✅ Shutdown complete")

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

deriv_bridge = DerivStreamingBridge()

# ============================================================================
# MAIN
# ============================================================================

async def main():
    print("\n" + "=" * 70)
    print("K1RL QUANT - INSTITUTIONAL REWARDS v5.2.1-CRASH500")
    print("HuggingFace Spaces Edition 🤗 (CRASH500 NAMESPACE ISOLATION)")
    print("=" * 70)
    print(f"✅ CRASH500 channel prefix: '{CHANNEL_PREFIX}'")
    print(f"✅ CRASH500 databases: features=DB{REDIS_DB_FEATURES}, rewards=DB{REDIS_DB_REWARDS}")
    print("✅ FIXED: Uses robust RedisAblyClient V10.1 (blocking listener)")
    print("✅ FIXED: Bounded reward task pool (no coroutine leak)")
    print("✅ FIXED: Connection health monitoring")
    print("✅ FIXED: Deriv auto-reconnection")
    print("✅ FIXED: Event loop reference for thread→asyncio bridge")
    print("✅ Auto-reconnecting WebSocket")
    print("✅ O(1) signal tracking")
    print("✅ TTL price cache")
    print("✅ Batch processing with backpressure")
    print("✅ Memory-bounded buffers")
    print("✅ HuggingFace Spaces compatibility")
    print("✅ ZERO cross-talk with v24 Space")
    print("=" * 70 + "\n")
    
    if IS_HF_SPACES:
        print("🤗 HuggingFace Spaces environment detected")
        print("")
    
    engine = RewardsEngine()
    
    try:
        await engine.start()
    except KeyboardInterrupt:
        print("\n>>> Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        await engine.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n>>> Stopped")