"""
Integrated Feature Enhancer for Trading System
Combines 62+ features from multiple sources:
- Core technical indicators
- Price derivatives
- Candlestick patterns  
- Support/resistance levels
- Meta voting features (published separately)

FIXED:
- Deprecated fillna warnings
- Async publish issues
- NaN handling in entropy calculation
- Added meta features extraction and publishing
- Added raw price to features (close_scaled, price, close_price)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore, skew, kurtosis
from collections import deque
from datetime import datetime
import threading
import logging
import nest_asyncio
import time
import asyncio
from ably import AblyRealtime
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================


ABLY_API_KEY = "NQGegQ.zUgpeg:li51-KyV8d1NlJZbnikF_McbYFV5FVZsXXAInLpMO34"
MT5_LOGIN = 27886128
MT5_PASSWORD = "Usabata123$"
MT5_SERVER = "Deriv-Demo"
SYMBOL = "Volatility 75 Index"


FEATURE_WINDOW = 10
TIMEFRAMES = {
    'xs': FEATURE_WINDOW,       # Extra small (10 ticks)
    's': FEATURE_WINDOW * 2,    # Small (20 ticks)
    'm': FEATURE_WINDOW * 4,
    'l': FEATURE_WINDOW * 8,
    'xl': FEATURE_WINDOW * 16,
    '5m': FEATURE_WINDOW * 30,
}

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_skew(x): 
    clean_x = x[~np.isnan(x)]
    return skew(clean_x) if len(clean_x) >= 3 else 0.0

def safe_kurtosis(x): 
    clean_x = x[~np.isnan(x)]
    return kurtosis(clean_x) if len(clean_x) >= 3 else 0.0

def min_max_scale(series):
    if len(series) == 0:
        return pd.Series([])
    min_val, max_val = series.min(), series.max()
    if max_val - min_val == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)

def safe_entropy(series):
    """Safely compute entropy with proper NaN handling"""
    try:
        clean_series = series.dropna()
        if len(clean_series) < 5:
            return 0.0
        
        if clean_series.nunique() == 1:
            return 0.0
        
        hist, _ = np.histogram(clean_series, bins=10, density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
            
        return -np.sum(hist * np.log(hist))
    except:
        return 0.0

# Candlestick pattern functions
def gravestone_doji(o, h, l, c): 
    try:
        return int(abs(c-o) < 0.02 and h-max(o,c) > 0.02)
    except:
        return 0

def four_price_doji(o, h, l, c): 
    try:
        return int(round(o,2) == round(h,2) == round(l,2) == round(c,2))
    except:
        return 0

def doji(o, h, l, c): 
    try:
        return int(round(o,2) == round(c,2) and l <= o-0.01 and h >= o+0.01)
    except:
        return 0

def spinning_top(o, h, l, c): 
    try:
        return int(abs(h-l) >= 2*abs(c-o))
    except:
        return 0

def bullish_candle(o, h, l, c): 
    try:
        return int(c > o + 0.02)
    except:
        return 0

def bearish_candle(o, h, l, c): 
    try:
        return int(c < o and abs(c-o) >= 0.01)
    except:
        return 0

def dragonfly_candle(o, h, l, c): 
    try:
        return int(abs(c-o) < 0.01 and (min(c,o)-l) > 2*(h-max(c,o)))
    except:
        return 0

def spinning_top_bearish_followup(c1, c2): 
    try:
        return spinning_top(*c1) and c2[0] > c2[3]
    except:
        return 0

def bullish_candle_followed_by_dragonfly(c1, c2): 
    try:
        return bullish_candle(*c1) and dragonfly_candle(*c2) and c2[0] > c1[3]
    except:
        return 0

def find_supports(p, df): 
    try:
        return list(df['Low'][(df['Low'].shift(1) > df['Low']) & 
                              (df['Low'].shift(-1) > df['Low']) & 
                              (df['Low'] < p)])
    except:
        return []

def find_resistances(p, df): 
    try:
        return list(df['High'][(df['High'].shift(1) < df['High']) & 
                               (df['High'].shift(-1) < df['High']) & 
                               (df['High'] > p)])
    except:
        return []

def find_stop_level(p, df):
    try:
        lows = df['Low'][-10:]
        mins = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
        below = mins[mins < p]
        return float(below.max()) if not below.empty else None
    except:
        return None

def dist_to_nearest(p, levels): 
    try:
        return float(min(abs(p - x) for x in levels)) if levels else -1.0
    except:
        return -1.0

def cluster_strength(levels):
    try:
        if not levels: return 0.0
        levels = sorted(levels)
        clusters = 0
        i = 0
        while i < len(levels):
            j, count = i+1, 1
            while j < len(levels) and abs(levels[j]-levels[i]) <= 0.1: 
                count += 1
                j += 1
            if count > 1: 
                clusters += count
            i = j
        return float(clusters)
    except:
        return 0.0

# ============================================================================
# INTEGRATED FEATURE ENHANCER WITH META FEATURES
# ============================================================================
class IntegratedFeatureEnhancer:
    def __init__(self, ably_client, agent_names, window_size=100):
        self.ably = ably_client
        self.agent_names = agent_names
        self.window_size = window_size
        
        self.price_buffers = {name: deque(maxlen=window_size) for name in agent_names}
        
        # Single shared feature channel for all 62 features
        self.features_channel = ably_client.channels.get("integrated_features_all")
        
        # Meta feature channels remain if you want
        self.meta_channels = {
            name: ably_client.channels.get(f"meta_features-{name}") 
            for name in agent_names
        }
        
        self.latest_computed_features = {}
        self.features_lock = threading.Lock()
        logger.info(f"IntegratedFeatureEnhancer initialized for {len(agent_names)} agents (single shared channel)")

    def compute_core_technical_features(self, df):
        """Compute 18 core technical indicators"""
        df = df.copy()
        eps = 1e-10
        
        # Log return + basic stats
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1)).replace([np.inf, -np.inf], 0)
        df['rolling_mean_5'] = df['Close'].rolling(5, min_periods=1).mean()
        df['rolling_std_5'] = df['Close'].rolling(5, min_periods=1).std().replace(0, eps)
        df['zscore_5'] = (df['Close'] - df['rolling_mean_5']) / df['rolling_std_5']
        
        # RSI
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + eps)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].ewm(span=5, adjust=False).mean()
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        tr = np.maximum.reduce([high_low, high_close, low_close])
        df['atr'] = pd.Series(tr).rolling(14, min_periods=1).mean()
        
        # CDF features with better NaN handling
        window = min(100, len(df))
        if window >= 20:
            df['cdf_value'] = df['log_return'].rolling(window).apply(
                lambda x: percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 10 else 0.5
            )
        else:
            df['cdf_value'] = 0.5
            
        df['cdf_value'] = df['cdf_value'].ffill().bfill().fillna(0.5)
        df['cdf_slope'] = df['cdf_value'].diff().ewm(span=5, adjust=False).mean()
        df['cdf_diff'] = df['cdf_value'] - df['cdf_value'].shift(10)
        df['cdf_diff'] = df['cdf_diff'].ewm(span=5, adjust=False).mean()
        
        # Volatility quantiles + entropy
        df['volatility_quantile_90'] = df['rolling_std_5'].rolling(min(100, len(df)), min_periods=20).quantile(0.9)
        df['volatility_ratio'] = df['rolling_std_5'] / (df['volatility_quantile_90'] + eps)
        df['volatility_ratio'] = df['volatility_ratio'].clip(0, 3)
        
        df['entropy_50'] = df['log_return'].rolling(min(50, len(df)), min_periods=20).apply(safe_entropy)
        
        # Autocorrelation + momentum
        df['autocorr_3'] = df['log_return'].rolling(20, min_periods=5).apply(
            lambda x: x.autocorr(lag=3) if len(x) > 3 else 0
        )
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Volume features
        df['volume_change_rate'] = df['Volume'].pct_change().replace([np.inf, -np.inf], 0)
        df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + eps)
        df['volume_zscore'] = df['volume_zscore'].clip(-3, 3)
        
        return df
    
    def compute_derivative_features(self, df, window=10):
        """Compute price derivatives and their statistics"""
        df = df.copy()
        
        df['price_vel'] = df['Close'].diff()
        df['price_acc'] = df['price_vel'].diff()
        df['price_jrk'] = df['price_acc'].diff()
        
        numeric = ['price_vel', 'price_acc', 'price_jrk']
        for col in numeric:
            try:
                df[f'{col}_mean'] = df[col].rolling(window, min_periods=3).mean()
                df[f'{col}_std'] = df[col].rolling(window, min_periods=3).std()
                df[f'{col}_skew'] = df[col].rolling(window, min_periods=3).apply(safe_skew, raw=True)
                df[f'{col}_kurtosis'] = df[col].rolling(window, min_periods=3).apply(safe_kurtosis, raw=True)
            except:
                pass
        
        return df
    
    def compute_additional_technical(self, df):
        """Compute additional technical indicators"""
        df = df.copy()
        eps = 1e-10
        
        df['ma10'] = df['Close'].rolling(10, min_periods=1).mean()
        df['ma20'] = df['Close'].rolling(20, min_periods=1).mean()
        df['std20'] = df['Close'].rolling(20, min_periods=1).std()
        
        df['bollinger_upper'] = df['ma20'] + 2 * df['std20']
        df['bollinger_lower'] = df['ma20'] - 2 * df['std20']
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / (df['ma20'] + eps)
        df['bollinger_position'] = (df['Close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'] + eps)
        df['bollinger_position'] = df['bollinger_position'].clip(0, 1)
        
        return df
    
    def compute_candlestick_patterns(self, df):
        """Compute candlestick patterns"""
        df = df.copy()
        
        if 'Open' not in df.columns:
            df['Open'] = df['Close']
        
        patterns = [
            ('gravestone_doji', gravestone_doji),
            ('four_price_doji', four_price_doji),
            ('doji', doji),
            ('spinning_top', spinning_top),
            ('bullish_candle', bullish_candle),
            ('bearish_candle', bearish_candle),
            ('dragonfly_candle', dragonfly_candle)
        ]
        
        for name, func in patterns:
            df[name] = df.apply(
                lambda r: func(r['Open'], r['High'], r['Low'], r['Close']), 
                axis=1
            )
        
        df['spinning_top_bearish_followup'] = 0
        df['bullish_then_dragonfly'] = 0
        
        for i in range(1, len(df)):
            c1 = tuple(df.iloc[i-1][['Open', 'High', 'Low', 'Close']])
            c2 = tuple(df.iloc[i][['Open', 'High', 'Low', 'Close']])
            
            if spinning_top_bearish_followup(c1, c2):
                df.at[df.index[i], 'spinning_top_bearish_followup'] = 1
            if bullish_candle_followed_by_dragonfly(c1, c2):
                df.at[df.index[i], 'bullish_then_dragonfly'] = 1
        
        return df
    
    def compute_support_resistance_features(self, df):
        """Compute support/resistance features"""
        df = df.copy()
        
        if len(df) < 10:
            df['distance_to_nearest_support'] = 0.0
            df['distance_to_nearest_resistance'] = 0.0
            df['near_support'] = 0
            df['near_resistance'] = 0
            df['distance_to_stop_loss'] = 0.5
            df['support_strength'] = 0.0
            df['resistance_strength'] = 0.0
            return df
        
        current_price = df['Close'].iloc[-1]
        supports = find_supports(current_price, df)
        resistances = find_resistances(current_price, df)
        stop_level = find_stop_level(current_price, df)
        
        min_p, max_p = df['Low'].min(), df['High'].max()
        rng = max_p - min_p if max_p > min_p else 1
        
        df['distance_to_nearest_support'] = dist_to_nearest(current_price, supports)
        df['distance_to_nearest_resistance'] = dist_to_nearest(current_price, resistances)
        df['near_support'] = int(any(abs(current_price - s) < 0.3 for s in supports)) if supports else 0
        df['near_resistance'] = int(any(abs(current_price - r) < 0.3 for r in resistances)) if resistances else 0
        df['distance_to_stop_loss'] = (current_price - stop_level) / rng if stop_level else 0.5
        df['support_strength'] = cluster_strength([s/rng for s in supports])
        df['resistance_strength'] = cluster_strength([r/rng for r in resistances])
        
        return df
    
    def compute_all_features(self, df):
        """Compute all integrated features with robust error handling"""
        try:
            if len(df) < 10:
                logger.debug(f"Insufficient data: {len(df)} rows")
                return pd.DataFrame()
            
            df = self.compute_core_technical_features(df)
            df = self.compute_derivative_features(df)
            df = self.compute_additional_technical(df)
            df = self.compute_candlestick_patterns(df)
            df = self.compute_support_resistance_features(df)
            
            # Clean up infinities and NaNs
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            return pd.DataFrame()
    
    def extract_meta_features(self, df, current_price):
        """Extract meta voting features for separate publishing - 24 features total
        
        Structure matches original:
        - 8 voting features (7 support/resistance + 1 close_price)
        - 15 technical features (filtered and scaled)
        - 1 timestamp (added during publish)
        Total: 24 features
        """
        try:
            if len(df) < 10:
                return {}
            
            supports = find_supports(current_price, df)
            resistances = find_resistances(current_price, df)
            stop_level = find_stop_level(current_price, df)
            
            min_p, max_p = df['Low'].min(), df['High'].max()
            rng = max_p - min_p if max_p > min_p else 1
            
            # VOTING FEATURES (8: 7 support/resistance + 1 close_price)
            voting = {
                'distance_to_nearest_support_scaled': dist_to_nearest(current_price, supports) / rng if rng > 0 else 0.0,
                'distance_to_nearest_resistance_scaled': dist_to_nearest(current_price, resistances) / rng if rng > 0 else 0.0,
                'near_support': int(any(abs(current_price - s) < 0.3 for s in supports)) if supports else 0,
                'near_resistance': int(any(abs(current_price - r) < 0.3 for r in resistances)) if resistances else 0,
                'distance_to_stop_loss_scaled': (current_price - stop_level) / rng if stop_level and rng > 0 else 0.5,
                'support_strength_scaled': cluster_strength([s/rng for s in supports]) if rng > 0 else 0.0,
                'resistance_strength_scaled': cluster_strength([r/rng for r in resistances]) if rng > 0 else 0.0,
                'close_price': float(current_price)
            }
            
            # FILTERED TECHNICAL FEATURES (15 specific features)
            latest = df.iloc[-1]
            
            # Define mappings: (dataframe_column_name, meta_feature_name)
            feature_mappings = [
                ('price_vel', 'price_vel_scaled'),
                ('price_acc', 'price_acc_scaled'),
                ('price_jrk', 'price_jrk_scaled'),
                ('price_vel_mean', 'price_vel_mean_scaled'),
                ('price_acc_mean', 'price_acc_mean_scaled'),
                ('price_jrk_mean', 'price_jrk_mean_scaled'),
                ('ma10', 'ma10_scaled'),
                ('ma20', 'ma20_scaled'),
                ('bollinger_upper', 'bollinger_upper_scaled'),
                ('bollinger_lower', 'bollinger_lower_scaled'),
                ('macd', 'macd_scaled'),
                ('macd_signal', 'macd_signal_scaled'),
                ('macd_hist', 'macd_hist_scaled'),
                ('rsi_14', 'rsi_scaled'),
                ('std20', 'std20_scaled')
            ]
            
            filtered = {}
            for df_col, meta_col in feature_mappings:
                if df_col in latest.index:
                    val = float(latest[df_col])
                    filtered[meta_col] = val
                else:
                    filtered[meta_col] = 0.0
            
            # COMBINE: voting (8) + filtered technical (15) + timestamp (added in publish) = 24
            meta_features = {**filtered, **voting}
            
            return meta_features
            
        except Exception as e:
            logger.error(f"Meta feature extraction failed: {e}")
            return {}
    
    def process_raw_tick(self, agent_name, price_data):
        """Process incoming raw price tick - FIXED VERSION WITH PRICE"""
        try:
            close_price = price_data.get('close', 0)
            
            self.price_buffers[agent_name].append({
                'Close': close_price,
                'High': price_data.get('high', close_price),
                'Low': price_data.get('low', close_price),
                'Volume': price_data.get('volume', 0),
                'Open': price_data.get('open', close_price)
            })
            
            if len(self.price_buffers[agent_name]) < 30:
                return
            
            df = pd.DataFrame(list(self.price_buffers[agent_name]))
            enhanced_df = self.compute_all_features(df)
            
            if enhanced_df.empty:
                return
            
            latest_features = enhanced_df.iloc[-1].to_dict()
            
            # ✅ FIX: ADD ALL PRICE VARIANTS (covers all use cases)
            latest_features['price'] = float(close_price)
            latest_features['close_scaled'] = float(close_price)  # ← Primary expected name
            latest_features['close_price'] = float(close_price)   # ← Used by meta-model
            
            with self.features_lock:
                self.latest_computed_features[agent_name] = latest_features.copy()
            
        except Exception as e:
            logger.error(f"[{agent_name}] Feature enhancement failed: {e}")
    
    async def publish_features(self, agent_name, features_dict):
        """Publish all 62 integrated features to the single shared channel"""
        try:
            clean_features = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in features_dict.items()
            }
            
            payload = {
                'agent': agent_name,
                'features': clean_features,
                'timestamp': datetime.utcnow().isoformat(),
                'feature_count': len(clean_features)
            }
            
            await self.features_channel.publish("integrated-features", payload)
            
        except Exception as e:
            logger.error(f"[{agent_name}] Feature publish failed: {e}")
    
        
    async def publish_meta_features(self, agent_name, meta_features):
        """Async publish meta features to separate channel"""
        try:
            channel = self.meta_channels[agent_name]
            
            clean_meta = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in meta_features.items()
            }
            
            # Add metadata (agent + timestamp)
            clean_meta['agent'] = agent_name
            clean_meta['timestamp'] = datetime.utcnow().isoformat()
            
            await channel.publish("meta_features", clean_meta)
            
        except Exception as e:
            logger.error(f"[{agent_name}] Meta feature publish failed: {e}")
    
    def get_latest_state_features(self, agent_name=None):
        """Get latest computed state features"""
        with self.features_lock:
            if agent_name:
                return self.latest_computed_features.get(agent_name, {})
            
            if not self.latest_computed_features:
                return {}
            
            all_features = list(self.latest_computed_features.values())
            if not all_features:
                return {}
            
            avg_features = {}
            feature_keys = all_features[0].keys()
            
            for key in feature_keys:
                values = [f[key] for f in all_features if key in f]
                if values:
                    avg_features[key] = float(np.mean(values))
            
            return avg_features
    
    def get_feature_summary(self):
        """Get summary of available features"""
        with self.features_lock:
            if not self.latest_computed_features:
                return "No features computed yet"
            
            sample_agent = list(self.latest_computed_features.keys())[0]
            features = self.latest_computed_features[sample_agent]
            
            summary = f"Total Features: {len(features)}\n\n"
            summary += "Feature Categories:\n"
            summary += f"  • Core Technical: ~19 features\n"
            summary += f"  • Derivatives: ~15 features\n"
            summary += f"  • Additional Technical: ~7 features\n"
            summary += f"  • Candlestick Patterns: 9 features\n"
            summary += f"  • Support/Resistance: 7 features\n"
            summary += f"  • Price Data: 3 features (price, close_scaled, close_price)\n"
            summary += f"\nMeta Features (published separately to meta_features-{{tf}}):\n"
            summary += f"  • Voting Features: 8 (support/resistance + close_price)\n"
            summary += f"  • Filtered Technical: 15 (scaled)\n"
            summary += f"  • Timestamp: 1\n"
            summary += f"  • Total Meta: 24 features\n"
            
            return summary

# ============================================================================
# ASYNC WRAPPER
# ============================================================================
class AsyncIntegratedFeatureEnhancer:
    """Async wrapper for integrated feature enhancement with robust Ably listeners"""

    def __init__(self, ably_client, agent_names, window_size=100):
        self.enhancer = IntegratedFeatureEnhancer(ably_client, agent_names, window_size)
        self.ably = ably_client
        self.agents = agent_names
        self.running = False
        self.channels = {}
        
    def get_latest_state_features(self, agent_name=None):
        return self.enhancer.get_latest_state_features(agent_name)
    
    async def start(self):
        self.running = True
        logger.info("AsyncIntegratedFeatureEnhancer started")
        logger.info(self.enhancer.get_feature_summary())
        await self._start_ably_listeners()

    async def _start_ably_listeners(self):
        """Attach and subscribe to all feature/meta channels with retries"""
        if not self.ably:
            logger.error("No Ably client available")
            return

        if hasattr(self.ably, 'connection') and self.ably.connection.state != 'connected':
            try:
                self.ably.connection.connect()
                for _ in range(20):
                    await asyncio.sleep(0.5)
                    if self.ably.connection.state == 'connected':
                        break
                else:
                    logger.error("Failed to connect to Ably within timeout")
                    return
            except Exception as e:
                logger.error(f"Ably connection attempt failed: {e}")
                return

        logger.info(f"Starting Ably listeners (connection state: {self.ably.connection.state})")

        for agent in self.agents:
            agent_str = agent.decode('utf-8') if isinstance(agent, bytes) else str(agent)

            feature_ok = await self._subscribe_with_retry(agent_str, "integrated-features",
                                                          lambda msg, name=agent_str: self._handle_feature_message(name, msg))
            meta_ok = await self._subscribe_with_retry(agent_str, "meta_features",
                                                       lambda msg, name=agent_str: self._handle_meta_features_message(name, msg),
                                                       channel_suffix="meta_features-")
            
            if feature_ok:
                logger.info(f"✓ [{agent_str}] Feature channel attached")
            if meta_ok:
                logger.info(f"✓ [{agent_str}] Meta features channel attached")

    async def _subscribe_with_retry(self, agent_name, event_name, callback, max_retries=3, timeout=10, channel_suffix=""):
        """Attach and subscribe to a channel with retries"""
        channel_name = f"{channel_suffix}{agent_name}" if channel_suffix else agent_name

        for attempt in range(max_retries):
            try:
                channel = self.ably.channels.get(channel_name)
                self.channels[channel_name] = channel

                attach_task = asyncio.create_task(channel.attach())
                try:
                    await asyncio.wait_for(attach_task, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Attach timeout for {channel_name} (attempt {attempt+1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False

                subscribe_task = asyncio.create_task(channel.subscribe(event_name, callback))
                try:
                    await asyncio.wait_for(subscribe_task, timeout=timeout)
                    return True
                except asyncio.TimeoutError:
                    logger.warning(f"Subscribe timeout for {channel_name} (attempt {attempt+1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False

            except Exception as e:
                logger.warning(f"[{channel_name}] Subscription attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
        return False

    async def process_tick(self, agent_name, price_data):
        """Process tick and publish features + meta features separately per agent"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.enhancer.process_raw_tick, agent_name, price_data)
    
        features = self.enhancer.get_latest_state_features(agent_name)
        if features:
            await self._publish_with_retry(agent_name, features, meta=False)
    
            df = pd.DataFrame(list(self.enhancer.price_buffers[agent_name]))
            if len(df) >= 10:
                current_price = price_data.get('close', 0)
                meta_features = self.enhancer.extract_meta_features(df, current_price)
                if meta_features:
                    await self._publish_with_retry(agent_name, meta_features, meta=True)
    
    async def _publish_with_retry(self, agent_name, features_dict, meta=False):
        """Publish agent features or meta features with retry"""
        channel_name = f"meta_features-{agent_name}" if meta else agent_name
        event_name = "meta_features" if meta else "feature"
    
        if channel_name not in self.channels:
            self.channels[channel_name] = self.ably.channels.get(channel_name)
    
        channel = self.channels[channel_name]
    
        payload = {
            'agent': agent_name,
            'features' if not meta else 'meta_features': features_dict,
            'timestamp': datetime.utcnow().isoformat()
        }
    
        for attempt in range(3):
            try:
                await channel.publish(event_name, payload)
                break
            except Exception as e:
                logger.warning(f"Publish attempt {attempt+1} failed on {channel_name}: {e}")
                await asyncio.sleep(2 ** attempt)

    def _handle_feature_message(self, agent_name, msg):
        logger.debug(f"[{agent_name}] Feature message received: {msg}")


    def _handle_meta_features_message(self, agent_name, msg):
        logging.debug(f"[{agent_name}] Meta feature message received: {msg}")
# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution loop"""
    
    nest_asyncio.apply()
    
    logger.info("=" * 80)
    logger.info("🚀 INTEGRATED FEATURE ENHANCER WITH META - STARTING")
    logger.info("=" * 80)
    
    # Initialize MT5
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        error = mt5.last_error()
        raise RuntimeError(f"❌ MT5 initialization failed: {error}")
    
    logger.info(f"✅ MT5 initialized: {MT5_SERVER}")
    logger.info(f"   Symbol: {SYMBOL}")
    
    # Verify symbol
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        mt5.shutdown()
        raise RuntimeError(f"❌ Symbol {SYMBOL} not found")
    
    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            mt5.shutdown()
            raise RuntimeError(f"❌ Failed to select symbol {SYMBOL}")
    
    logger.info(f"✅ Symbol verified")
    
    # Initialize Ably
    logger.info("\n📡 Connecting to Ably...")
    try:
        ably_client = AblyRealtime(ABLY_API_KEY)
        await asyncio.sleep(1)
        logger.info("✅ Ably connected")
    except Exception as e:
        mt5.shutdown()
        raise RuntimeError(f"❌ Ably connection failed: {e}")
    
    # Initialize Feature Enhancers
    logger.info("\n🔧 Initializing feature enhancers...")
    agent_names = list(TIMEFRAMES.keys())
    logger.info(f"   Timeframes: {TIMEFRAMES}")
    
    enhancer = AsyncIntegratedFeatureEnhancer(
        ably_client=ably_client,
        agent_names=agent_names,
        window_size=100
    )
    
    await enhancer.start()
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 FEATURE SUMMARY")
    logger.info("=" * 80)
    logger.info(enhancer.enhancer.get_feature_summary())
    
    # Create channels
    agent_channels = {tf: ably_client.channels.get(tf) for tf in TIMEFRAMES}
    
    logger.info("\n✅ All systems initialized - Starting tick processing...")
    logger.info("   Publishing to: agent channels + meta_features channels\n")
    
    # Main loop
    tick_count = 0
    last_summary_time = time.time()
    feature_counts = {tf: 0 for tf in TIMEFRAMES}
    meta_counts = {tf: 0 for tf in TIMEFRAMES}
    
    try:
        while True:
            try:
                tick = mt5.symbol_info_tick(SYMBOL)
                
                if tick is None:
                    logger.warning("⚠️ No tick received")
                    await asyncio.sleep(1)
                    continue
                
                tick_count += 1
                mid_price = (tick.bid + tick.ask) / 2.0
                timestamp = datetime.utcnow()
                
                for timeframe_name, window in TIMEFRAMES.items():
                    try:
                        price_data = {
                            'close': mid_price,
                            'high': tick.ask,
                            'low': tick.bid,
                            'open': mid_price,
                            'volume': getattr(tick, 'volume', 0)
                        }
                        
                        await enhancer.process_tick(timeframe_name, price_data)
                        
                        features = enhancer.get_latest_state_features(timeframe_name)
                        
                        if features:
                            feature_counts[timeframe_name] += 1
                            meta_counts[timeframe_name] += 1
                            
                            features_with_meta = {
                                **features,
                                'timestamp': timestamp.isoformat(),
                                'tick_count': tick_count,
                                'timeframe': timeframe_name
                            }
                            
                            await agent_channels[timeframe_name].publish(
                                "integrated-features",
                                {
                                    "agent": timeframe_name,
                                    "features": features_with_meta,
                                    "feature_count": len(features)
                                }
                            )
                            
                            if feature_counts[timeframe_name] % 10 == 0:
                                logger.info(
                                    f"✅ [{timeframe_name}] Tick #{tick_count}: "
                                    f"{len(features)} features + meta | Price: {mid_price:.5f}"
                                )
                    
                    except Exception as e:
                        logger.error(f"❌ [{timeframe_name}] Error: {e}")
                
                # Summary every 60 seconds
                if time.time() - last_summary_time > 60:
                    logger.info("\n" + "=" * 80)
                    logger.info(f"📊 SUMMARY (Tick #{tick_count})")
                    logger.info("=" * 80)
                    logger.info(f"Price: {mid_price:.5f} | Time: {timestamp.strftime('%H:%M:%S')}")
                    logger.info("\nFeature Sets Published:")
                    for tf in TIMEFRAMES:
                        logger.info(f"  {tf}: {feature_counts[tf]} agent + {meta_counts[tf]} meta")
                    logger.info("=" * 80 + "\n")
                    last_summary_time = time.time()
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("\n⚠️ Shutting down...")
                break
                
            except Exception as e:
                logger.error(f"❌ Tick error: {e}")
                await asyncio.sleep(1)
    
    finally:
        logger.info("\n" + "=" * 80)
        logger.info("🛑 SHUTTING DOWN")
        logger.info("=" * 80)
        
        mt5.shutdown()
        
        logger.info(f"\nTotal Ticks: {tick_count}")
        logger.info("Feature Sets Published:")
        for tf in TIMEFRAMES:
            logger.info(f"  {tf}: {feature_counts[tf]} agent + {meta_counts[tf]} meta")
        logger.info("\n✅ Shutdown complete")

if __name__ == "__main__":
    try:
        nest_asyncio.apply()
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Program exited")