#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  K1RL QUANT — INSTITUTIONAL SIGNAL FLIP NOTIFIER v3.1              ║
║                                                                      ║
║  Alerts ONLY on BUY↔SELL flips. No NEUTRAL. No noise.              ║
║  Each alert includes 5-minute signal window assessment.              ║
║                                                                      ║
║  Architecture:                                                       ║
║    Redis pub/sub → SignalCache (ring buffer) → FlipDetector → TG    ║
║                                                                      ║
║  Signal Cache:                                                       ║
║    - Fixed-capacity ring buffer (no unbounded growth)                ║
║    - TTL-based eviction on read path (lazy cleanup)                  ║
║    - Periodic compaction every 60s (background cleanup)              ║
║    - O(1) insert, O(n) window query (n bounded by capacity)         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple, NamedTuple
from collections import deque
from pathlib import Path

import requests

# ── Redis ─────────────────────────────────────────────────────────────────
from redis_config import REDIS_URL, REDIS_PASSWORD
from redis_connection_manager import RedisAblyClient, RedisMessage, IS_HF_SPACES

# ── Logging ───────────────────────────────────────────────────────────────
LOG_DIR = Path('/home/user/app/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [TELEGRAM] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / 'telegram_notifier.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "8473902481:AAE0wep-lSXJ9yamU0Sj2KyNTQj9MImA3fk"
)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "6224721972")

SIGNAL_CHANNEL = "final_signals"
TARGET_AGENT = os.environ.get("TARGET_AGENT", "10m")

# Timing
SIGNAL_WINDOW_SEC = 300       # 5-minute assessment window
NOTIFICATION_COOLDOWN = 60    # Min seconds between alerts
COMPACTION_INTERVAL = 60      # Background cleanup interval

# Cache limits
CACHE_CAPACITY = 2000         # Max entries in ring buffer (hard ceiling)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class SignalEntry(NamedTuple):
    """Immutable signal record — lightweight, hashable, sortable by timestamp."""
    timestamp: float
    direction: str        # "BUY" or "SELL"
    price: float
    agent: str            # Agent that produced this signal
    signal_key: str


class WindowStats(NamedTuple):
    """Snapshot of signal distribution within the assessment window."""
    buy_count: int
    sell_count: int
    total: int
    buy_pct: float
    sell_pct: float
    window_seconds: int
    agents_buy: List[str]
    agents_sell: List[str]


# ============================================================================
# SIGNAL CACHE — RING BUFFER WITH TTL EVICTION
# ============================================================================

class SignalCache:
    """
    Fixed-capacity signal cache with TTL-based eviction.
    
    Design:
      - Ring buffer (deque with maxlen) — O(1) insert, bounded memory
      - Lazy eviction: stale entries removed on every read query
      - Background compaction: periodic sweep removes expired entries
      - Thread-safe: all mutations under lock (called from Redis listener thread)
    
    Memory ceiling: CACHE_CAPACITY * ~120 bytes = ~240KB max
    """

    __slots__ = ('_buffer', '_lock', '_ttl', '_capacity',
                 '_total_inserted', '_total_evicted')

    def __init__(self, capacity: int = CACHE_CAPACITY, ttl: int = SIGNAL_WINDOW_SEC):
        self._buffer: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._ttl = ttl
        self._capacity = capacity
        self._total_inserted = 0
        self._total_evicted = 0

    def insert(self, entry: SignalEntry) -> None:
        """Insert a signal. O(1). Thread-safe. Oldest auto-evicted at capacity."""
        with self._lock:
            self._buffer.append(entry)
            self._total_inserted += 1

    def query_window(self, window_sec: Optional[int] = None) -> List[SignalEntry]:
        """
        Return all signals within the window. Evicts stale entries lazily.
        O(n) where n <= capacity.
        """
        cutoff = time.time() - (window_sec or self._ttl)
        with self._lock:
            evicted = 0
            while self._buffer and self._buffer[0].timestamp < cutoff:
                self._buffer.popleft()
                evicted += 1
            self._total_evicted += evicted
            return list(self._buffer)

    def compact(self) -> int:
        """
        Background compaction — remove all expired entries.
        Called periodically by the compaction thread.
        Returns number of entries evicted.
        """
        cutoff = time.time() - self._ttl
        with self._lock:
            before = len(self._buffer)
            while self._buffer and self._buffer[0].timestamp < cutoff:
                self._buffer.popleft()
            evicted = before - len(self._buffer)
            self._total_evicted += evicted
            return evicted

    def get_latest(self, direction: Optional[str] = None) -> Optional[SignalEntry]:
        """Get the most recent signal, optionally filtered by direction."""
        with self._lock:
            if direction:
                for entry in reversed(self._buffer):
                    if entry.direction == direction:
                        return entry
                return None
            return self._buffer[-1] if self._buffer else None

    def clear(self) -> None:
        """Full cache flush."""
        with self._lock:
            self._buffer.clear()

    @property
    def stats(self) -> Dict:
        with self._lock:
            return {
                'size': len(self._buffer),
                'capacity': self._capacity,
                'ttl': self._ttl,
                'total_inserted': self._total_inserted,
                'total_evicted': self._total_evicted,
                'utilization': f"{len(self._buffer) / max(1, self._capacity) * 100:.1f}%"
            }

    def compute_window_stats(self, window_sec: Optional[int] = None) -> WindowStats:
        """
        Compute signal distribution stats for the assessment window.
        Used to enrich flip alerts with market context.
        """
        signals = self.query_window(window_sec)

        buy_signals = [s for s in signals if s.direction == "BUY"]
        sell_signals = [s for s in signals if s.direction == "SELL"]

        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        total = buy_count + sell_count

        # Unique agents voting each direction
        agents_buy = sorted(set(s.agent for s in buy_signals))
        agents_sell = sorted(set(s.agent for s in sell_signals))

        return WindowStats(
            buy_count=buy_count,
            sell_count=sell_count,
            total=total,
            buy_pct=round(buy_count / total * 100, 1) if total > 0 else 0.0,
            sell_pct=round(sell_count / total * 100, 1) if total > 0 else 0.0,
            window_seconds=window_sec or self._ttl,
            agents_buy=agents_buy,
            agents_sell=agents_sell
        )


# ============================================================================
# UTILITY
# ============================================================================

def extract_agent_name(signal_key: str) -> str:
    """Extract agent name from signal key (e.g., '10m' from '10m_1736584234_54321')"""
    if not signal_key:
        return "unknown"
    key = signal_key.strip().lower()
    if "_" in key:
        return key.split("_")[0]
    match = re.search(r"(\d+[mhdw])", key)
    return match.group(1) if match else "unknown"


# ============================================================================
# TELEGRAM NOTIFIER
# ============================================================================

class TelegramNotifier:
    """Institutional-grade Telegram message delivery with rate limiting."""

    __slots__ = ('bot_token', 'chat_id', 'api_url',
                 '_last_send', '_min_interval', '_messages_sent')

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._last_send = 0.0
        self._min_interval = 2.0
        self._messages_sent = 0

    def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Rate-limited message dispatch."""
        elapsed = time.time() - self._last_send
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            resp = requests.post(
                self.api_url,
                json={'chat_id': self.chat_id, 'text': text[:4096], 'parse_mode': parse_mode},
                timeout=10
            )
            self._last_send = time.time()
            self._messages_sent += 1
            if resp.status_code == 200:
                logger.info(f"✅ TG sent (#{self._messages_sent})")
                return True
            logger.error(f"❌ TG error: {resp.status_code} — {resp.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"❌ TG request failed: {e}")
            return False

    def send_flip_alert(self, old: str, new: str, price: float,
                        agent: str, stats: WindowStats) -> bool:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if new == "BUY":
            header = "BULLISH FLIP"
            emoji = "🟢"
            trend = "Buying pressure dominates"
        else:
            header = "BEARISH FLIP"
            emoji = "🔴"
            trend = "Selling pressure dominates"

        old_emoji = "🟢" if old == "BUY" else "🔴"

        buy_agents = ", ".join(a.upper() for a in stats.agents_buy) or "—"
        sell_agents = ", ".join(a.upper() for a in stats.agents_sell) or "—"

        # Conviction score based on window spread
        if stats.total >= 5:
            spread = abs(stats.buy_pct - stats.sell_pct)
            if spread > 60:
                conviction = "🔥 Very High"
            elif spread > 30:
                conviction = "✅ High"
            elif spread > 10:
                conviction = "⚠️ Moderate"
            else:
                conviction = "❓ Low (contested)"
        else:
            conviction = "📉 Low data"

        msg = f"""🚨 <b>{header}</b> 🚨
<b>Agent: {agent.upper()}</b>

{old_emoji} {old} → {emoji} <b>{new}</b>
{trend}

💰 Price: <b>{price:.5f}</b>

━━━━━━━━━━━━━━━━━━━━━
📊 <b>5-Min Window:</b>
   🟢 BUY:  <b>{stats.buy_count}</b> ({stats.buy_pct:.1f}%)
   🔴 SELL: <b>{stats.sell_count}</b> ({stats.sell_pct:.1f}%)
   📈 Total: {stats.total} signals
   💪 Conviction: {conviction}

🗳️ <b>Agents:</b>
   BUY:  {buy_agents}
   SELL: {sell_agents}
━━━━━━━━━━━━━━━━━━━━━

⏰ {ts}
<i>K1RL QUANT v3.1</i>"""

        return self._send(msg)

    def send_startup_alert(self, agent: str, cache_stats: Dict) -> bool:
        msg = f"""🔔 <b>Signal Flip Monitor v3.1</b>

📡 Redis: Connected
🎯 Agent: <b>{agent.upper()}</b>
📊 Channel: {SIGNAL_CHANNEL}
🔄 Window: {SIGNAL_WINDOW_SEC // 60} min
🧊 Cache: {cache_stats['capacity']} slots, {cache_stats['ttl']}s TTL
⏰ {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🤗 {'HuggingFace Spaces' if IS_HF_SPACES else 'Local'}

Alerts on BUY↔SELL flips only.

<i>K1RL QUANT v3.1</i>"""
        return self._send(msg)

    def send_shutdown_alert(self, cache_stats: Dict) -> bool:
        msg = f"""🛑 <b>Monitor Stopped</b>
📦 Cache: {cache_stats['total_inserted']} inserted, {cache_stats['total_evicted']} evicted
⏰ {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""
        return self._send(msg)


# ============================================================================
# SIGNAL FLIP MONITOR
# ============================================================================

class RedisSignalMonitor:
    """
    Institutional signal flip detector.
    
    Architecture:
      1. Redis pub/sub delivers signals via RedisAblyClient listener thread
      2. _on_signal() inserts ALL agent signals into SignalCache
      3. Checks if TARGET agent's direction flipped
      4. On flip: compute WindowStats from cache -> send enriched TG alert
      5. Background compaction thread evicts stale cache entries every 60s
    
    No NEUTRAL state. No dominance thresholds. Just direction flips.
    """

    def __init__(self, target_agent: str, notifier: TelegramNotifier):
        self.target_agent = target_agent
        self.notifier = notifier

        # Signal cache (thread-safe ring buffer with TTL)
        self.cache = SignalCache(
            capacity=CACHE_CAPACITY,
            ttl=SIGNAL_WINDOW_SEC
        )

        # Flip state
        self.last_signal: Optional[str] = None   # "BUY" or "SELL"
        self.last_price: float = 0.0
        self.last_flip_time: float = 0.0

        # Counters
        self.total_signals = 0
        self.agent_signals = 0
        self.flips = 0

        # Control
        self.running = False
        self.redis_client: Optional[RedisAblyClient] = None
        self._compaction_thread: Optional[threading.Thread] = None

    # ── Signal handler (runs on Redis listener thread) ────────────────

    def _on_signal(self, message: RedisMessage):
        """Process incoming signal: cache ALL agents, detect target flip."""
        try:
            self.total_signals += 1

            # ── Parse ────────────────────────────────────────────
            data = message.data if isinstance(message, RedisMessage) else message
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            if isinstance(data, str):
                data = json.loads(data)

            action = data.get("final_action", data.get("action", "")).upper()
            if action not in ("BUY", "SELL"):
                return

            price = float(data.get("price", 0))
            signal_keys = data.get("signal_keys", [])
            if not isinstance(signal_keys, list):
                signal_keys = [str(signal_keys)]

            # ── Cache ALL agent signals for window stats ─────────
            now = time.time()
            for key in signal_keys:
                agent = extract_agent_name(str(key))
                self.cache.insert(SignalEntry(
                    timestamp=now,
                    direction=action,
                    price=price,
                    agent=agent,
                    signal_key=str(key)
                ))

            # ── Check if TARGET agent is in this batch ───────────
            target_found = any(
                extract_agent_name(str(k)) == self.target_agent
                for k in signal_keys
            )
            if not target_found:
                return

            self.agent_signals += 1

            # ── Flip detection ───────────────────────────────────
            if self.last_signal is None:
                self.last_signal = action
                self.last_price = price
                logger.info(f"📊 Initial: {action} @ {price:.5f}")
                return

            if action == self.last_signal:
                self.last_price = price
                return

            # ── FLIP DETECTED ────────────────────────────────────
            now_t = time.time()
            if now_t - self.last_flip_time < NOTIFICATION_COOLDOWN:
                remaining = NOTIFICATION_COOLDOWN - (now_t - self.last_flip_time)
                logger.info(f"⏳ Flip {self.last_signal}→{action} cooldown ({remaining:.0f}s)")
                self.last_signal = action
                self.last_price = price
                return

            self.flips += 1
            old = self.last_signal
            self.last_signal = action
            self.last_price = price

            logger.warning(f"🚨 FLIP #{self.flips}: {old} → {action} @ {price:.5f}")

            # Compute 5-min window stats from cache
            stats = self.cache.compute_window_stats()

            success = self.notifier.send_flip_alert(
                old=old,
                new=action,
                price=price,
                agent=self.target_agent,
                stats=stats
            )

            if success:
                self.last_flip_time = now_t

        except Exception as e:
            logger.error(f"❌ Signal error: {e}")

    # ── Background compaction ─────────────────────────────────────────

    def _compaction_loop(self):
        """Periodic cache cleanup — runs in background daemon thread."""
        while self.running:
            time.sleep(COMPACTION_INTERVAL)
            try:
                evicted = self.cache.compact()
                if evicted > 0:
                    logger.debug(f"🧹 Compacted {evicted} stale entries")
            except Exception as e:
                logger.error(f"❌ Compaction error: {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self):
        """Start monitoring (blocking)."""
        self.running = True

        logger.info("=" * 60)
        logger.info("🚀 K1RL QUANT — Signal Flip Notifier v3.1")
        logger.info(f"   Agent:       {self.target_agent.upper()}")
        logger.info(f"   Channel:     {SIGNAL_CHANNEL}")
        logger.info(f"   Window:      {SIGNAL_WINDOW_SEC // 60} min")
        logger.info(f"   Cache:       {CACHE_CAPACITY} slots")
        logger.info(f"   Cooldown:    {NOTIFICATION_COOLDOWN}s")
        logger.info(f"   Compaction:  every {COMPACTION_INTERVAL}s")
        logger.info(f"   Mode:        Flip only (no NEUTRAL)")
        logger.info("=" * 60)

        # Start compaction thread
        self._compaction_thread = threading.Thread(
            target=self._compaction_loop,
            daemon=True,
            name="CacheCompaction"
        )
        self._compaction_thread.start()

        # Connect to Redis
        self.redis_client = RedisAblyClient(
            redis_url=REDIS_URL,
            password=REDIS_PASSWORD,
            use_streams=False,
            database=0
        )

        # Subscribe
        channel = self.redis_client.channels.get(SIGNAL_CHANNEL)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(channel.subscribe("message", self._on_signal))
        logger.info(f"✅ Subscribed to {SIGNAL_CHANNEL}")

        # Startup alert
        self.notifier.send_startup_alert(self.target_agent, self.cache.stats)

        # Status loop
        logger.info("✅ Monitoring for flips...")
        try:
            while self.running:
                time.sleep(60)
                cs = self.cache.stats
                logger.info(
                    f"📊 STATUS | "
                    f"Signals: {self.total_signals} | "
                    f"{self.target_agent.upper()}: {self.agent_signals} | "
                    f"Direction: {self.last_signal or '—'} | "
                    f"Flips: {self.flips} | "
                    f"Cache: {cs['size']}/{cs['capacity']} ({cs['utilization']})"
                )
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Clean shutdown with cache flush."""
        self.running = False
        stats = self.cache.stats
        self.cache.clear()
        if self.redis_client:
            self.redis_client.close()
        self.notifier.send_shutdown_alert(stats)
        logger.info(f"✅ Shutdown | {stats}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("K1RL QUANT — Signal Flip Notifier v3.1")
    print("=" * 60)
    print(f"  Bot:        {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"  Chat:       {TELEGRAM_CHAT_ID}")
    print(f"  Channel:    {SIGNAL_CHANNEL}")
    print(f"  Agent:      {TARGET_AGENT}")
    print(f"  Window:     {SIGNAL_WINDOW_SEC // 60} min")
    print(f"  Cache:      {CACHE_CAPACITY} slots")
    print(f"  Cooldown:   {NOTIFICATION_COOLDOWN}s")
    print(f"  Mode:       Flip only (BUY↔SELL)")
    print("=" * 60 + "\n")

    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    monitor = RedisSignalMonitor(target_agent=TARGET_AGENT, notifier=notifier)

    try:
        monitor.start()
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        notifier.send_shutdown_alert(monitor.cache.stats)


if __name__ == "__main__":
    main()
    