# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ██╗  ██╗ ██╗██████╗ ██╗         ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗║
║  ██║ ██╔╝███║██╔══██╗██║        ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝║
║  █████╔╝ ╚██║██████╔╝██║        ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ║
║  ██╔═██╗  ██║██╔══██╗██║        ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ║
║  ██║  ██╗ ██║██║  ██║███████╗   ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ║
║  ╚═╝  ╚═╝ ╚═╝╚═╝  ╚═╝╚══════╝    ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ║
║                                                                              ║
║ ──────────────────────────────────────────────────────────────────────────── ║
║                                                                              ║
║              MT5 TERMINAL BRIDGE — DEEPHEDGE PLAYER                          ║
║                                                                              ║
║        Real-Time Execution • Risk-Governed • Institutional-Grade             ║
║                                                                              ║
║ ──────────────────────────────────────────────────────────────────────────── ║
║                                                                              ║
║   ASSET:        Volatility 75 Index        BROKER:     Deriv (MT5)           ║
║   ACTIONS:      BUY • SELL • HOLD • CLOSE  EXECUTION:  Market (IOC/FOK/RET)  ║
║   MAGIC:        234567                     COMMENT:    DeepHedge PlayMaker   ║
║                                                                              ║
║         “Deterministic execution, governed risk, real-market feedback.”      ║
║                                                                              ║
║              [ MT5 BRIDGE ONLINE ] v2.2-FINAL | 2025-12-24                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

=======================================================
CHANGELOG v2.2-FINAL:
- ✅ Added CLOSE action handler for exit orders
- ✅ Enhanced position streaming with validation
- ✅ Thread-safe operation with asyncio.Lock
- ✅ Better error handling and logging
- ✅ Rate limiting protection
- ✅ Synchronized with PlayMaker v2.2-FINAL
"""

import MetaTrader5 as mt5
import time
import asyncio
import logging
import sys
from datetime import datetime, timezone
from collections import deque
from ably import AblyRealtime
import nest_asyncio
import traceback
import json
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

MT5_LOGIN = 27886128
MT5_PASSWORD = "Usabata123$"
MT5_SERVER = "Deriv-Demo"
SYMBOL = "Volatility 75 Index"

DEFAULT_LOT_SIZE = 0.01
MIN_LOT_SIZE = 0.001
LOT_SIZE_STEP = 0.001

# ✅ RISK MANAGEMENT
MAX_DRAWDOWN = 1000.0  # Maximum drawdown in dollars
CLOSE_OPPOSITE_ON_SIGNAL = True  # Close opposite positions when new signal arrives

FULL_ABLY_KEY = "W4r0PQ.AFDs8w:y85JlyZVcIP9E5C5ECCQygR9xkR3z0CN3wjeXTNG8Mw"
ABLY_PRICE_SPREAD_POSITIONINFO_CHANNEL = "feature_data"
ABLY_MT5_ORDERS_CHANNEL = "orders"
ABLY_KEY_NAME, ABLY_KEY_SECRET = FULL_ABLY_KEY.split(":")

PRICE_UPDATE_INTERVAL = 1.0
POSITION_UPDATE_INTERVAL = 5.0

TRADE_MAGIC = 234567
TRADE_COMMENT = "DeepHedge PlayMaker"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimitedSender:
    """Rate limiter to prevent Ably channel rate limit errors"""
    
    def __init__(self, max_rate=40, time_window=1.0):
        self.max_rate = max_rate
        self.time_window = time_window
        self.send_times = deque()
        self.send_lock = asyncio.Lock()
    
    async def can_send(self):
        async with self.send_lock:
            now = time.time()
            while self.send_times and self.send_times[0] <= now - self.time_window:
                self.send_times.popleft()
            
            if len(self.send_times) < self.max_rate:
                self.send_times.append(now)
                return True
            return False
    
    async def wait_until_can_send(self):
        while not await self.can_send():
            await asyncio.sleep(0.1)


# ============================================================================
# MT5 TERMINAL BRIDGE
# ============================================================================

class MT5TerminalBridge:
    """Bridge between MT5 and DeepHedge PlayMaker via Ably"""
    
    def __init__(self):
        self.ably_key = f"{ABLY_KEY_NAME}:{ABLY_KEY_SECRET}"
        self.ably_realtime = None
        self.feature_channel = None
        self.order_channel = None
        
        # Rate limiters
        self.price_rate_limiter = RateLimitedSender(max_rate=30, time_window=1.0)
        self.order_rate_limiter = RateLimitedSender(max_rate=20, time_window=1.0)
        
        # State tracking
        self.current_positions = {}
        self.total_volume = 0.0
        self.total_pnl = 0.0
        
        # Price history
        self.price_history = deque(maxlen=200)
        self.returns_history = deque(maxlen=200)
        
        # Statistics
        self.orders_received = 0
        self.orders_executed = 0
        self.orders_failed = 0
        self.data_packets_sent = 0
        self.completed_trades = deque(maxlen=100)
        self.winning_trades = 0
        
        # ✅ Risk management
        self.peak_balance = 10000.0  # Starting balance
        self.current_balance = 10000.0
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0
        self.trading_halted = False
        
        # ✅ Thread-safe locks
        self.position_lock = asyncio.Lock()
        self.order_lock = asyncio.Lock()
        
        logger.info("=" * 70)
        logger.info("🚀 MT5 TERMINAL BRIDGE INITIALIZED v2.2-FINAL")
        logger.info(f"   Symbol: {SYMBOL}")
        logger.info(f"   CLOSE action: ACTIVE")
        logger.info(f"   Thread-safe: ACTIVE")
        logger.info(f"   ✅ Max Drawdown: ${MAX_DRAWDOWN}")
        logger.info(f"   ✅ Close Opposite: {CLOSE_OPPOSITE_ON_SIGNAL}")
        logger.info("=" * 70)
    
    def normalize_volume(self, volume):
        """Normalize volume to valid lot size"""
        try:
            volume = float(volume)
            volume = round(volume / LOT_SIZE_STEP) * LOT_SIZE_STEP
            if abs(volume) < MIN_LOT_SIZE:
                return 0.0
            return round(volume, 3)
        except (TypeError, ValueError) as e:
            logger.error(f"Volume normalization error: {e}")
            return 0.0
    
    async def initialize_ably(self):
        """Initialize Ably clients within running event loop"""
        logger.info("🔌 Initializing Ably connection...")
        self.ably_realtime = AblyRealtime(key=self.ably_key)
        
        connection_timeout = 10
        start_time = time.time()
        
        while self.ably_realtime.connection.state != 'connected':
            if time.time() - start_time > connection_timeout:
                raise TimeoutError("Ably connection timeout")
            await asyncio.sleep(0.5)
        
        self.feature_channel = self.ably_realtime.channels.get(ABLY_PRICE_SPREAD_POSITIONINFO_CHANNEL)
        self.order_channel = self.ably_realtime.channels.get(ABLY_MT5_ORDERS_CHANNEL)
        
        logger.info("✅ Ably connected successfully")
    
    # ========================================================================
    # MT5 DATA RETRIEVAL
    # ========================================================================
    
    def get_current_tick(self):
        """Get current market tick"""
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            if not tick:
                logger.error(f"Failed to get tick for {SYMBOL}")
                return None
            return tick
        except Exception as e:
            logger.error(f"Tick error: {e}")
            return None
    
    def get_symbol_info(self):
        """Get symbol information"""
        try:
            info = mt5.symbol_info(SYMBOL)
            if not info:
                logger.error(f"Failed to get symbol info for {SYMBOL}")
                return None
            return info
        except Exception as e:
            logger.error(f"Symbol info error: {e}")
            return None
    
    def get_open_positions(self):
        """Get all open positions for DeepHedge"""
        try:
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions is None:
                return []
            
            our_positions = [p for p in positions 
                           if p.magic == TRADE_MAGIC or TRADE_COMMENT in (p.comment or "")]
            return our_positions
            
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []
    
    def calculate_position_metrics(self, positions):
        """Calculate aggregate position metrics"""
        if not positions:
            return {
                'total_volume': 0.0,
                'net_position': 0.0,
                'total_pnl': 0.0,
                'num_positions': 0,
                'long_volume': 0.0,
                'short_volume': 0.0,
                'avg_entry_price': 0.0
            }
        
        long_volume = sum(p.volume for p in positions if p.type == mt5.ORDER_TYPE_BUY)
        short_volume = sum(p.volume for p in positions if p.type == mt5.ORDER_TYPE_SELL)
        net_position = long_volume - short_volume
        total_pnl = sum(p.profit for p in positions)
        
        if abs(net_position) > 1e-6:
            total_value = sum(p.volume * p.price_open * (1 if p.type == mt5.ORDER_TYPE_BUY else -1) 
                            for p in positions)
            avg_entry_price = abs(total_value / net_position)
        else:
            avg_entry_price = 0.0
        
        return {
            'total_volume': long_volume + short_volume,
            'net_position': net_position,
            'total_pnl': total_pnl,
            'num_positions': len(positions),
            'long_volume': long_volume,
            'short_volume': short_volume,
            'avg_entry_price': avg_entry_price
        }
    
    def calculate_performance_metrics(self):
        """Calculate win rate and other performance metrics"""
        if not self.completed_trades:
            return {'win_rate': 0.5, 'avg_pnl': 0.0, 'total_trades': 0, 'winning_trades': 0}
        
        total_trades = len(self.completed_trades)
        winning_trades = sum(1 for trade in self.completed_trades if trade.get('pnl', 0) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.5
        avg_pnl = sum(trade.get('pnl', 0) for trade in self.completed_trades) / total_trades
        
        return {
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades
        }
    
    def update_balance_and_drawdown(self, total_pnl):
        """✅ Update balance tracking and check drawdown limits"""
        self.current_balance = 10000.0 + total_pnl
        
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        self.current_drawdown = self.peak_balance - self.current_balance
        self.max_drawdown_reached = max(self.max_drawdown_reached, self.current_drawdown)
        
        # Check if max drawdown exceeded
        if self.current_drawdown >= MAX_DRAWDOWN and not self.trading_halted:
            self.trading_halted = True
            logger.error(f"🛑 MAX DRAWDOWN REACHED: ${self.current_drawdown:.2f} >= ${MAX_DRAWDOWN}")
            logger.error(f"🛑 TRADING HALTED - Manual intervention required")
            return True
        
        return False
    
    async def close_all_positions(self, reason="max_drawdown"):
        """✅ Emergency close all positions"""
        async with self.position_lock:
            positions = self.get_open_positions()
            
        if not positions:
            return
        
        logger.warning(f"⚠️ Closing all {len(positions)} positions | Reason: {reason}")
        
        for pos in positions:
            try:
                result = await self.close_position(pos.ticket, None, reason)
                if result and result.get('success'):
                    logger.info(f"✅ Emergency close: Ticket {pos.ticket}")
                await asyncio.sleep(0.5)  # Rate limit
            except Exception as e:
                logger.error(f"❌ Failed to close {pos.ticket}: {e}")
    
    async def close_opposite_positions(self, incoming_action):
        """✅ Close positions opposite to incoming signal"""
        if not CLOSE_OPPOSITE_ON_SIGNAL:
            return 0
        
        if incoming_action not in ['BUY', 'SELL']:
            return 0
        
        async with self.position_lock:
            positions = self.get_open_positions()
        
        if not positions:
            return 0
        
        # Determine opposite type
        if incoming_action == 'BUY':
            opposite_type = mt5.ORDER_TYPE_SELL
            opposite_name = 'SELL'
        else:
            opposite_type = mt5.ORDER_TYPE_BUY
            opposite_name = 'BUY'
        
        # Find opposite positions
        opposite_positions = [p for p in positions if p.type == opposite_type]
        
        if not opposite_positions:
            return 0
        
        logger.info(f"🔄 Closing {len(opposite_positions)} opposite {opposite_name} positions before {incoming_action}")
        
        closed_count = 0
        for pos in opposite_positions:
            try:
                result = await self.close_position(
                    pos.ticket, 
                    None,  # Close entire position
                    f"opposite_to_{incoming_action}"
                )
                if result and result.get('success'):
                    closed_count += 1
                    logger.info(f"✅ Closed opposite: Ticket {pos.ticket} | PnL: ${result.get('pnl', 0):.2f}")
                await asyncio.sleep(0.5)  # Rate limit
            except Exception as e:
                logger.error(f"❌ Failed to close opposite position {pos.ticket}: {e}")
        
        return closed_count
    
    def validate_position_for_playmaker(self, pos):
        """✅ Validate position data before sending to PlayMaker"""
        try:
            return {
                'ticket': int(pos.ticket),
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': float(pos.volume),
                'price_open': float(pos.price_open),
                'price_current': float(pos.price_current),
                'profit': float(pos.profit),
                'swap': float(pos.swap),
                'time_open': int(pos.time),
                'comment': str(pos.comment or "")
            }
        except Exception as e:
            logger.error(f"Position validation error: {e}")
            return None
    
    # ========================================================================
    # FEATURE DATA STREAMING
    # ========================================================================
    
    async def stream_price_data(self):
        """Continuously stream price data to feature_data channel"""
        logger.info(f"📡 Price streaming started (interval: {PRICE_UPDATE_INTERVAL}s)")
        
        while True:
            try:
                await self.price_rate_limiter.wait_until_can_send()
                
                tick = self.get_current_tick()
                if not tick:
                    await asyncio.sleep(PRICE_UPDATE_INTERVAL)
                    continue
                
                symbol_info = self.get_symbol_info()
                
                mid_price = float((tick.bid + tick.ask) / 2)
                spread = float(tick.ask - tick.bid)
                rel_spread = spread / mid_price if mid_price > 0 else 0.0
                
                self.price_history.append(mid_price)
                
                if len(self.price_history) >= 2:
                    ret = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
                    self.returns_history.append(ret)
                
                price_lag1 = self.price_history[-2] if len(self.price_history) >= 2 else mid_price
                price_lag5 = self.price_history[-6] if len(self.price_history) >= 6 else mid_price
                price_lag20 = self.price_history[-21] if len(self.price_history) >= 21 else mid_price
                
                if len(self.returns_history) >= 20:
                    recent_returns = list(self.returns_history)[-20:]
                    realized_vol = float(np.std(recent_returns) * np.sqrt(252 * 24 * 12))
                else:
                    realized_vol = 75.0
                
                # ✅ Thread-safe position access
                async with self.position_lock:
                    positions = self.get_open_positions()
                    position_metrics = self.calculate_position_metrics(positions)
                    performance_metrics = self.calculate_performance_metrics()
                    
                    # ✅ Update balance and check drawdown
                    drawdown_exceeded = self.update_balance_and_drawdown(position_metrics['total_pnl'])
                    
                    # ✅ Emergency close if drawdown exceeded
                    if drawdown_exceeded:
                        await self.close_all_positions("max_drawdown")
                    
                    # ✅ Validate position data for PlayMaker
                    position_details = []
                    for pos in positions:
                        validated_pos = self.validate_position_for_playmaker(pos)
                        if validated_pos:
                            position_details.append(validated_pos)
                
                feature_data = {
                    # Price data
                    'price': mid_price,
                    'bid': float(tick.bid),
                    'ask': float(tick.ask),
                    'spread': spread,
                    'rel_spread': rel_spread,
                    
                    # Lagged prices
                    'price_lag1': float(price_lag1),
                    'price_lag5': float(price_lag5),
                    'price_lag20': float(price_lag20),
                    
                    # Returns and volatility
                    'returns_history': [float(r) for r in list(self.returns_history)[-100:]],
                    'realized_vol': realized_vol,
                    'vol_ratio': realized_vol / 75.0,
                    
                    # Position information
                    'current_position': float(position_metrics['net_position']),
                    'total_volume': float(position_metrics['total_volume']),
                    'long_volume': float(position_metrics['long_volume']),
                    'short_volume': float(position_metrics['short_volume']),
                    'current_pnl': float(position_metrics['total_pnl']),
                    'num_positions': int(position_metrics['num_positions']),
                    'avg_entry_price': float(position_metrics['avg_entry_price']),
                    
                    # ✅ VALIDATED POSITIONS FOR EXIT OPTIMIZER
                    'positions': position_details,
                    
                    # Symbol info
                    'point': float(symbol_info.point if symbol_info else 0.01),
                    'digits': int(symbol_info.digits if symbol_info else 2),
                    
                    # Timestamp
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'server_time': int(tick.time),
                    
                    # Market state indicators
                    'bars_since_entry': int(len(self.price_history)),
                    'win_rate': float(performance_metrics['win_rate']),
                    'energy_level': float(1.0 - abs(position_metrics['total_pnl'] / 10000.0)),
                    'rolling_pnl': float(position_metrics['total_pnl']),
                    'signal_accuracy': float(performance_metrics['win_rate']),
                    'vol_spike': bool(realized_vol > 90.0),
                    'flash_crash': False,
                    
                    # ✅ Risk management indicators
                    'current_balance': float(self.current_balance),
                    'current_drawdown': float(self.current_drawdown),
                    'max_drawdown_limit': float(MAX_DRAWDOWN),
                    'trading_halted': bool(self.trading_halted)
                }
                
                await self.feature_channel.publish('market_data', feature_data)
                self.data_packets_sent += 1
                
                if self.data_packets_sent % 20 == 0:
                    logger.info(f"📊 Market data sent | Price: {mid_price:.2f} | "
                               f"Positions: {len(position_details)} | "
                               f"PnL: ${position_metrics['total_pnl']:.2f}")
                
                await asyncio.sleep(PRICE_UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Price streaming error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)
    
    # ========================================================================
    # ORDER EXECUTION
    # ========================================================================
    
    def get_filling_mode(self):
        """Get the correct filling mode for the symbol"""
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            return mt5.ORDER_FILLING_RETURN
        
        filling = symbol_info.filling_mode
        
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN
    
    async def close_position(self, ticket, volume=None, reason="exit_optimizer"):
        """
        ✅ Close a specific position (full or partial)
        
        Args:
            ticket: Position ticket to close
            volume: Volume to close (None = close entire position)
            reason: Reason for closing
        """
        async with self.order_lock:
            try:
                # Validate ticket type
                try:
                    ticket = int(ticket)
                except (TypeError, ValueError):
                    raise Exception(f"Invalid ticket type: {type(ticket)}")
                
                position = mt5.positions_get(ticket=ticket)
                if not position or len(position) == 0:
                    raise Exception(f"Position {ticket} not found")
                
                position = position[0]
                
                # Determine close volume
                if volume is None:
                    close_volume = position.volume
                else:
                    close_volume = min(float(volume), position.volume)
                    close_volume = self.normalize_volume(close_volume)
                
                if close_volume < MIN_LOT_SIZE:
                    raise Exception(f"Close volume {close_volume} below minimum {MIN_LOT_SIZE}")
                
                tick = self.get_current_tick()
                if not tick:
                    raise Exception("Cannot get price for close order")
                
                # Determine close order type and price
                if position.type == mt5.ORDER_TYPE_BUY:
                    close_type = mt5.ORDER_TYPE_SELL
                    close_price = tick.bid
                else:
                    close_type = mt5.ORDER_TYPE_BUY
                    close_price = tick.ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": close_volume,
                    "type": close_type,
                    "position": ticket,
                    "price": close_price,
                    "deviation": 20,
                    "magic": TRADE_MAGIC,
                    "comment": f"{TRADE_COMMENT} {reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": self.get_filling_mode(),
                }
                
                logger.info(f"🔒 Closing position {ticket} | Volume: {close_volume:.3f} | Reason: {reason}")
                
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Calculate PnL for this close
                    if position.type == mt5.ORDER_TYPE_BUY:
                        pnl = (close_price - position.price_open) * close_volume
                    else:
                        pnl = (position.price_open - close_price) * close_volume
                    
                    self.completed_trades.append({
                        'ticket': ticket,
                        'pnl': float(pnl),
                        'close_time': time.time(),
                        'reason': reason
                    })
                    
                    if pnl > 0:
                        self.winning_trades += 1
                    
                    logger.info(f"✅ Position closed | Ticket: {ticket} | "
                               f"Volume: {close_volume:.3f} | PnL: ${pnl:.2f} | Reason: {reason}")
                    
                    execution_result = {
                        'success': True,
                        'ticket': ticket,
                        'volume_closed': close_volume,
                        'pnl': float(pnl),
                        'reason': reason,
                        'close_price': float(close_price)
                    }
                    
                    self.orders_executed += 1
                    
                    return execution_result
                else:
                    error_msg = f"Close failed: retcode={result.retcode if result else 'No result'}"
                    if result:
                        error_msg += f", comment={result.comment}"
                    raise Exception(error_msg)
                    
            except Exception as e:
                logger.error(f"❌ Close position error: {e}")
                self.orders_failed += 1
                return {
                    'success': False,
                    'error': str(e),
                    'ticket': ticket
                }
    
    async def execute_order(self, order_data):
        """✅ Execute order from DeepHedge PlayMaker v2.2-FINAL"""
        async with self.order_lock:
            try:
                action = str(order_data.get('action', '')).upper()
                
                # ✅ Check if trading is halted
                if self.trading_halted and action in ['BUY', 'SELL']:
                    logger.error(f"🛑 Trading halted due to max drawdown - Order rejected: {action}")
                    return {
                        'success': False,
                        'error': 'Trading halted - Max drawdown exceeded',
                        'action': action
                    }
                
                # ✅ HANDLE CLOSE ACTION
                if action == 'CLOSE':
                    ticket = order_data.get('ticket')
                    volume = order_data.get('volume')
                    reason = order_data.get('reason', 'exit_optimizer')
                    
                    if not ticket:
                        raise Exception("CLOSE action requires 'ticket' field")
                    
                    result = await self.close_position(ticket, volume, reason)
                    
                    # Send confirmation
                    await self.send_execution_confirmation(order_data, result)
                    
                    return result
                
                # ✅ Close opposite positions if enabled
                if action in ['BUY', 'SELL'] and CLOSE_OPPOSITE_ON_SIGNAL:
                    closed_count = await self.close_opposite_positions(action)
                    if closed_count > 0:
                        logger.info(f"✅ Closed {closed_count} opposite positions before {action}")
                
                # HANDLE BUY/SELL ACTIONS
                volume = order_data.get('volume', DEFAULT_LOT_SIZE)
                symbol = order_data.get('symbol', SYMBOL)
                reason = order_data.get('reason', 'DeepHedge')
                signal_direction = order_data.get('signal_direction', 0.0)
                signal_confidence = order_data.get('signal_confidence', 0.0)
                
                volume = self.normalize_volume(volume)
                
                if volume < MIN_LOT_SIZE:
                    logger.warning(f"⚠️ Volume {volume:.3f} below minimum {MIN_LOT_SIZE}, skipping")
                    return {
                        'success': False,
                        'error': 'Volume below minimum',
                        'volume': volume
                    }
                
                logger.info(f"🎯 Executing order: {action} {volume:.3f} lots")
                logger.info(f"   Signal: {signal_direction:+.2f} @ {signal_confidence:.2f} confidence")
                
                # Get current positions
                async with self.position_lock:
                    current_positions = self.get_open_positions()
                    mt5_metrics = self.calculate_position_metrics(current_positions)
                    mt5_net_position = mt5_metrics['net_position']
                
                tick = self.get_current_tick()
                if not tick:
                    raise Exception("Cannot get current price")
                
                execution_result = None
                
                if action == 'HOLD':
                    logger.info("⏸️ HOLD action received, no execution needed")
                    return {
                        'success': True,
                        'action': 'HOLD',
                        'message': 'No action taken'
                    }
                
                elif action == 'BUY':
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": volume,
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": tick.ask,
                        "deviation": 20,
                        "magic": TRADE_MAGIC,
                        "comment": TRADE_COMMENT,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": self.get_filling_mode(),
                    }
                    
                    result = mt5.order_send(request)
                    
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        execution_result = {
                            'success': True,
                            'ticket': int(result.order),
                            'price': float(result.price),
                            'volume': float(result.volume),
                            'action': 'BUY',
                            'signal_direction': float(signal_direction),
                            'signal_confidence': float(signal_confidence),
                            'mt5_position_before': float(mt5_net_position),
                            'mt5_position_after': float(mt5_net_position + result.volume)
                        }
                        self.orders_executed += 1
                        logger.info(f"✅ BUY executed | Ticket: {result.order} | "
                                   f"Price: {result.price:.2f} | Volume: {result.volume:.3f}")
                    else:
                        error_msg = f"Order failed: retcode={result.retcode if result else 'No result'}"
                        if result:
                            error_msg += f", comment={result.comment}"
                        raise Exception(error_msg)
                
                elif action == 'SELL':
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": volume,
                        "type": mt5.ORDER_TYPE_SELL,
                        "price": tick.bid,
                        "deviation": 20,
                        "magic": TRADE_MAGIC,
                        "comment": TRADE_COMMENT,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": self.get_filling_mode(),
                    }
                    
                    result = mt5.order_send(request)
                    
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        execution_result = {
                            'success': True,
                            'ticket': int(result.order),
                            'price': float(result.price),
                            'volume': float(result.volume),
                            'action': 'SELL',
                            'signal_direction': float(signal_direction),
                            'signal_confidence': float(signal_confidence),
                            'mt5_position_before': float(mt5_net_position),
                            'mt5_position_after': float(mt5_net_position - result.volume)
                        }
                        self.orders_executed += 1
                        logger.info(f"✅ SELL executed | Ticket: {result.order} | "
                                   f"Price: {result.price:.2f} | Volume: {result.volume:.3f}")
                    else:
                        error_msg = f"Order failed: retcode={result.retcode if result else 'No result'}"
                        if result:
                            error_msg += f", comment={result.comment}"
                        raise Exception(error_msg)
                
                else:
                    raise Exception(f"Unknown action: {action}")
                
                if execution_result:
                    await self.send_execution_confirmation(order_data, execution_result)
                
                return execution_result
                
            except Exception as e:
                logger.error(f"❌ Order execution failed: {e}")
                traceback.print_exc()
                self.orders_failed += 1
                
                await self.send_execution_confirmation(order_data, {
                    'success': False,
                    'error': str(e)
                })
                
                return None
    
    async def send_execution_confirmation(self, original_order, execution_result):
        """Send execution confirmation back to DeepHedge"""
        try:
            await self.order_rate_limiter.wait_until_can_send()
            
            confirmation = {
                'original_order': original_order,
                'execution': execution_result,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'bridge_version': '2.2-FINAL'
            }
            
            await self.order_channel.publish('execution_confirmation', confirmation)
            
            status = "✅ SUCCESS" if execution_result.get('success', False) else "❌ FAILED"
            action = execution_result.get('action', original_order.get('action', 'UNKNOWN'))
            logger.info(f"📤 Execution confirmation sent | {status} | Action: {action}")
            
        except Exception as e:
            logger.error(f"Failed to send confirmation: {e}")
    
    # ========================================================================
    # ORDER LISTENING
    # ========================================================================
    
    async def on_order_received(self, message):
        """Handle incoming orders from DeepHedge PlayMaker"""
        try:
            self.orders_received += 1
            
            data = message.data if isinstance(message.data, dict) else json.loads(message.data)
            
            action = data.get('action', 'UNKNOWN')
            
            if action == 'CLOSE':
                ticket = data.get('ticket', 'N/A')
                volume = data.get('volume', 0.0)
                reason = data.get('reason', 'unknown')
                logger.info(f"🔥 CLOSE order received: Ticket {ticket} | Volume: {volume:.3f} | Reason: {reason}")
            else:
                volume = data.get('volume', 0.0)
                signal_direction = data.get('signal_direction', 0.0)
                signal_confidence = data.get('signal_confidence', 0.0)
                logger.info(f"🔥 Order received: {action} {volume:.3f} lots | "
                           f"Signal: {signal_direction:+.2f} @ {signal_confidence:.2f}")
            
            result = await self.execute_order(data)
            
            if result and result.get('success'):
                logger.info(f"✅ Order executed successfully")
            else:
                logger.error(f"❌ Order execution failed")
                
        except Exception as e:
            logger.error(f"Order processing error: {e}")
            traceback.print_exc()
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    
    async def start(self):
        """Start the MT5 terminal bridge"""
        logger.info("🚀 MT5 TERMINAL BRIDGE STARTING v2.2-FINAL")
        
        await self.initialize_ably()
        
        asyncio.create_task(self.stream_price_data())
        
        try:
            await self.order_channel.subscribe('order', self.on_order_received)
            await self.order_channel.subscribe('close_order', self.on_order_received)
            logger.info(f"✅ Subscribed to {ABLY_MT5_ORDERS_CHANNEL}")
            
            while True:
                await asyncio.sleep(60)
                
                async with self.position_lock:
                    positions = self.get_open_positions()
                    metrics = self.calculate_position_metrics(positions)
                    performance = self.calculate_performance_metrics()
                
                logger.info(
                    f"📊 STATUS | Data Sent: {self.data_packets_sent} | "
                    f"Orders: Received={self.orders_received} Executed={self.orders_executed} Failed={self.orders_failed} | "
                    f"Open: {len(positions)} | Net Position: {metrics['net_position']:.3f} | "
                    f"PnL: ${metrics['total_pnl']:.2f} | Win Rate: {performance['win_rate']:.1%} | "
                    f"Balance: ${self.current_balance:.2f} | DD: ${self.current_drawdown:.2f}/${MAX_DRAWDOWN} | "
                    f"Halted: {self.trading_halted}"
                )
                
        except Exception as e:
            logger.error(f"Service error: {e}")
            raise


# ============================================================================
# ENTRY POINTS
# ============================================================================

async def run_bridge():
    """Run the MT5 terminal bridge"""
    try:
        nest_asyncio.apply()
        
        # Initialize MT5
        if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return
        
        logger.info(f"✅ MT5 connected - Account: {MT5_LOGIN}")
        
        if not mt5.symbol_select(SYMBOL, True):
            logger.error(f"❌ Failed to select symbol: {SYMBOL}")
            mt5.shutdown()
            return
        
        logger.info(f"✅ Symbol selected: {SYMBOL}")
        
        # Create and run bridge
        bridge = MT5TerminalBridge()
        await bridge.start()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopped by user")
    except Exception as e:
        logger.error(f"Bridge error: {e}")
        traceback.print_exc()
    finally:
        mt5.shutdown()
        logger.info("👋 MT5 disconnected")


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("🚀 MT5 TERMINAL BRIDGE FOR DEEPHEDGE PLAYMAKER v2.2-FINAL")
    print("=" * 80)
    print(f"   Symbol: {SYMBOL}")
    print(f"   Min Lot Size: {MIN_LOT_SIZE}")
    print(f"   Lot Size Step: {LOT_SIZE_STEP}")
    print(f"   Price Updates: Every {PRICE_UPDATE_INTERVAL}s")
    print(f"   Position Updates: Continuous")
    print(f"   Feature Channel: {ABLY_PRICE_SPREAD_POSITIONINFO_CHANNEL}")
    print(f"   Order Channel: {ABLY_MT5_ORDERS_CHANNEL}")
    print("\n   ✅ SENDS: Market data, validated positions, Real PnL")
    print("   ✅ RECEIVES: BUY/SELL/CLOSE orders from DeepHedge PlayMaker")
    print("   ✅ Thread-safe operation with asyncio.Lock")
    print("   ✅ Rate limiting protection (30/s price, 20/s orders)")
    print("   ✅ Position data validation before streaming")
    print(f"   ✅ Max Drawdown Protection: ${MAX_DRAWDOWN}")
    print(f"   ✅ Auto-Close Opposite Positions: {CLOSE_OPPOSITE_ON_SIGNAL}")
    print("\n   🔧 v2.2-FINAL Improvements:")
    print("   • Thread-safe position access")
    print("   • Enhanced CLOSE order handling")
    print("   • Type validation for all numeric fields")
    print("   • Better error messages and logging")
    print("   • Synchronized with PlayMaker v2.2-FINAL")
    print("   • Max drawdown protection with auto-halt")
    print("   • Opposite position closure on signal reversal")
    print("=" * 80 + "\n")
    
    asyncio.run(run_bridge())


if __name__ == "__main__":
    main()
else:
    print("\n" + "=" * 70)
    print("🚀 MT5 TERMINAL BRIDGE v2.2-FINAL")
    print("=" * 70)
    print("\nRun with:")
    print("  asyncio.run(run_bridge())")
    print("\nChannels:")
    print(f"  SENDS → {ABLY_PRICE_SPREAD_POSITIONINFO_CHANNEL} (market data)")
    print(f"  RECEIVES ← {ABLY_MT5_ORDERS_CHANNEL} (orders)")
    print("\nChangelog v2.2-FINAL:")
    print("  ✅ Thread-safe position access with asyncio.Lock")
    print("  ✅ Enhanced CLOSE action handler")
    print("  ✅ Position data validation before streaming")
    print("  ✅ Type conversion for all numeric fields")
    print("  ✅ Better error handling and recovery")
    print("  ✅ Synchronized with PlayMaker v2.2-FINAL")
    print("  ✅ Rate limiting protection")
    print("  ✅ Max drawdown protection ($1000)")
    print("  ✅ Auto-close opposite positions on signal reversal")
    print("=" * 70)
    
#+263780563561  ENG Karl Muzunze Masvingo Zimbabwe