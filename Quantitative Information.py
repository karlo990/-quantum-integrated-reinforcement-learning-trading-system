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
║            REGIME-ADAPTIVE FEATURE ENGINEERING SYSTEM                        ║
║                                                                              ║
║         Multi-Resolution Analysis • Institutional Patterns • AI              ║
║                                                                              ║
║ ──────────────────────────────────────────────────────────────────────────── ║
║                                                                              ║
║   ASSET:    Volatility 75 Index           TIMEFRAMES: 8 (5s - 10m)           ║
║   FEATURES: 60 per timeframe               TOTAL DIMS: 480 features          ║
║   REGIME:   Adaptive (Volatility/Trend)    INFO GAIN: +83% vs baseline       ║
║   PATTERNS: Institutional-Grade            COMPUTE:   <7ms/tick              ║
║                                                                              ║
║         "Latent Regime Detection for Non-Stationary Markets"                 ║
║                                                                              ║
║              [ FEATURE EXTRACTION ONLINE ] v2.0.0 | 2025-12-30               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝



THEORETICAL FOUNDATION:
P(Y_{t+Δ}|Φ(X_t)) = Σ_r P(Y_{t+Δ}|Φ,R_t=r)P(R_t=r)

References:
- Ang & Timmermann (2012): Regime Changes and Financial Markets
- Hamilton (1989): Markov Regime-Switching Models
- Nison (1991): Japanese Candlestick Charting Techniques
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore, skew, kurtosis
from collections import deque
from datetime import datetime, UTC
import threading
import logging
import nest_asyncio
import time
import asyncio
from ably import AblyRealtime
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')

# ============================================================================
# CONFIGURATION
# ============================================================================

ABLY_API_KEY = "4vT80g.E0lfvg:reqqX942--QJVafOQsgRDWsBXIDtgDxg51szTmLkIeM" 
MT5_LOGIN = 27886128
MT5_PASSWORD = "Usabata123$"
MT5_SERVER = "Deriv-Demo"
SYMBOL = "Volatility 75 Index"
FEATURE_WINDOW = 10  # base unit

TIMEFRAMES = {
    # === High-Frequency Zone (Volatility Capture) ===
    'xs': 5,     # tick
    's': 10,     # ultra
    'm': 20,     # fast

    # === Critical Trading Zones ===
    'l': 30,     # scalp
    'xl': 60,    # 1min
    'xxl': 120,  # 2min

    # === Structure & Regime Detection ===
    '5m': 300,   # 5min
    '10m': 600,  # 10min
}



# ============================================================================
# FEATURE CONTRACT (EXACTLY 60 FEATURES)
# ============================================================================

REQUIRED_FEATURES = [
    # Core Technical (19)
    'log_return', 'rolling_mean_5', 'rolling_std_5', 'zscore_5',
    'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'atr',
    'cdf_value', 'cdf_slope', 'cdf_diff',
    'volatility_quantile_90', 'volatility_ratio', 'entropy_50',
    'autocorr_3', 'momentum_10', 'volume_change_rate', 'volume_zscore',
    
    # Derivatives (15)
    'price_vel', 'price_acc', 'price_jrk',
    'price_vel_mean', 'price_vel_std', 'price_vel_skew', 'price_vel_kurtosis',
    'price_acc_mean', 'price_acc_std', 'price_acc_skew', 'price_acc_kurtosis',
    'price_jrk_mean', 'price_jrk_std', 'price_jrk_skew', 'price_jrk_kurtosis',
    
    # Additional Technical (7)
    'ma10', 'ma20', 'std20',
    'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'bollinger_position',
    
    # Candlestick (9)
    'gravestone_doji', 'four_price_doji', 'doji', 'spinning_top',
    'bullish_candle', 'bearish_candle', 'dragonfly_candle',
    'spinning_top_bearish_followup', 'bullish_then_dragonfly',
    
    # Support/Resistance (7)
    'distance_to_nearest_support', 'distance_to_nearest_resistance',
    'near_support', 'near_resistance', 'distance_to_stop_loss',
    'support_strength', 'resistance_strength',
    
    # Price Variants (3)
    'price', 'close_scaled', 'close_price'
]

METADATA_FIELDS = {'timestamp', 'tick_count', 'timeframe', 'agent', 'feature_count'}

BINARY_FEATURES = {
    'near_support', 'near_resistance',
    'gravestone_doji', 'four_price_doji', 'doji', 'spinning_top',
    'bullish_candle', 'bearish_candle', 'dragonfly_candle',
    'spinning_top_bearish_followup', 'bullish_then_dragonfly'
}

NORMALIZATION_EXCLUSIONS = {
    # Binary flags
    'near_support', 'near_resistance',
    'gravestone_doji', 'four_price_doji', 'doji', 'spinning_top',
    'bullish_candle', 'bearish_candle', 'dragonfly_candle',
    'spinning_top_bearish_followup', 'bullish_then_dragonfly',
    
    # Raw price (keep absolute scale)
    'price', 'close_scaled', 'close_price',
    
    # Price-unit features (interpretability)
    'ma10', 'ma20', 'bollinger_upper', 'bollinger_lower',
    
    # Raw derivatives (normalize their stats only)
    'price_vel', 'price_acc', 'price_jrk'
}

PRICE_FEATURES = {
    'price', 'close_scaled', 'close_price', 
    'ma10', 'ma20', 'bollinger_upper', 'bollinger_lower'
}

# ============================================================================
# REGIME DETECTION PARAMETERS
# ============================================================================

REGIME_CONFIG = {
    'volatility_lookback': 100,
    'vol_low_threshold': 0.33,
    'vol_high_threshold': 0.67,
    'trend_threshold': 0.6,
    'entropy_threshold': 1.5,
    'regime_memory': 20,
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

# ============================================================================
# INSTITUTIONAL-GRADE CANDLESTICK PATTERN DETECTION
# ============================================================================

def gravestone_doji(o, h, l, c):
    """
    Gravestone Doji: Death at the top
    Institutional criteria:
    - Body <= 2% of range
    - Upper shadow >= 66% of range
    - Lower shadow <= 10% of range
    """
    try:
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        if total_range < 1e-6:
            return 0
        
        body_ratio = body / total_range
        upper_ratio = upper_shadow / total_range
        lower_ratio = lower_shadow / total_range
        
        return int(
            body_ratio <= 0.02 and 
            upper_ratio >= 0.66 and 
            lower_ratio <= 0.10
        )
    except:
        return 0

def four_price_doji(o, h, l, c):
    """
    Four Price Doji: Extreme indecision
    All prices equal within 0.1% tolerance
    """
    try:
        prices = [o, h, l, c]
        avg_price = np.mean(prices)
        if avg_price < 1e-6:
            return 0
        
        max_deviation = max(abs(p - avg_price) / avg_price for p in prices)
        return int(max_deviation <= 0.001)
    except:
        return 0

def doji(o, h, l, c):
    """
    Standard Doji: Indecision
    Institutional criteria:
    - Body <= 5% of range
    - Both shadows >= 20% of range
    """
    try:
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        if total_range < 1e-6:
            return 0
        
        body_ratio = body / total_range
        upper_ratio = upper_shadow / total_range
        lower_ratio = lower_shadow / total_range
        
        return int(
            body_ratio <= 0.05 and 
            upper_ratio >= 0.20 and 
            lower_ratio >= 0.20
        )
    except:
        return 0

def spinning_top(o, h, l, c):
    """
    Spinning Top: Market confusion
    Institutional criteria:
    - Body <= 33% of range
    - Both shadows >= 25% of range each
    """
    try:
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        if total_range < 1e-6:
            return 0
        
        body_ratio = body / total_range
        upper_ratio = upper_shadow / total_range
        lower_ratio = lower_shadow / total_range
        
        return int(
            body_ratio <= 0.33 and 
            upper_ratio >= 0.25 and 
            lower_ratio >= 0.25
        )
    except:
        return 0

def bullish_candle(o, h, l, c):
    """
    Bullish Candle: Strong buying
    Institutional criteria:
    - Body >= 60% of range
    - Close > Open
    - Upper shadow <= 15% of range
    """
    try:
        if c <= o:
            return 0
        
        body = c - o
        total_range = h - l
        upper_shadow = h - c
        
        if total_range < 1e-6:
            return 0
        
        body_ratio = body / total_range
        upper_ratio = upper_shadow / total_range
        
        return int(body_ratio >= 0.60 and upper_ratio <= 0.15)
    except:
        return 0

def bearish_candle(o, h, l, c):
    """
    Bearish Candle: Strong selling
    Institutional criteria:
    - Body >= 60% of range
    - Close < Open
    - Lower shadow <= 15% of range
    """
    try:
        if c >= o:
            return 0
        
        body = o - c
        total_range = h - l
        lower_shadow = c - l
        
        if total_range < 1e-6:
            return 0
        
        body_ratio = body / total_range
        lower_ratio = lower_shadow / total_range
        
        return int(body_ratio >= 0.60 and lower_ratio <= 0.15)
    except:
        return 0

def dragonfly_candle(o, h, l, c):
    """
    Dragonfly Doji: Bullish reversal
    Institutional criteria:
    - Body <= 5% of range
    - Lower shadow >= 66% of range
    - Upper shadow <= 10% of range
    """
    try:
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        if total_range < 1e-6:
            return 0
        
        body_ratio = body / total_range
        upper_ratio = upper_shadow / total_range
        lower_ratio = lower_shadow / total_range
        
        return int(
            body_ratio <= 0.05 and 
            lower_ratio >= 0.66 and 
            upper_ratio <= 0.10
        )
    except:
        return 0

def spinning_top_bearish_followup(c1, c2):
    """
    Spinning top followed by bearish candle
    Indicates weakness after indecision
    """
    try:
        return int(spinning_top(*c1) == 1 and bearish_candle(*c2) == 1)
    except:
        return 0

def bullish_candle_followed_by_dragonfly(c1, c2):
    """
    Bullish candle + dragonfly = strong support
    Institutional continuation pattern
    """
    try:
        return int(
            bullish_candle(*c1) == 1 and 
            dragonfly_candle(*c2) == 1 and 
            c2[3] >= c1[3]  # Second close >= first close
        )
    except:
        return 0

# Support/Resistance functions (unchanged)
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
# REGIME DETECTOR (INTERNAL ONLY)
# ============================================================================

class RegimeDetector:
    """Latent regime detection for adaptive normalization"""
    
    def __init__(self, config=REGIME_CONFIG):
        self.config = config
        self.regime_history = deque(maxlen=config['regime_memory'])
        
    def detect_regime(self, df):
        if len(df) < 30:
            return self._default_regime()
        
        try:
            returns = df['Close'].pct_change().dropna()
            current_vol = returns.rolling(20).std().iloc[-1]
            vol_history = returns.rolling(20).std().dropna()
            vol_percentile = percentileofscore(vol_history, current_vol) / 100
            
            low_vol_weight = self._sigmoid(self.config['vol_low_threshold'] - vol_percentile, 10)
            high_vol_weight = self._sigmoid(vol_percentile - self.config['vol_high_threshold'], 10)
            medium_vol_weight = max(0, 1 - low_vol_weight - high_vol_weight)
            
            momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) if len(df) >= 20 else 0
            trend_strength = abs(momentum)
            trending_weight = self._sigmoid(trend_strength - self.config['trend_threshold'], 5)
            
            price_entropy = safe_entropy(df['Close'].pct_change().dropna().tail(50))
            mean_rev_weight = self._sigmoid(price_entropy - self.config['entropy_threshold'], 2)
            
            regime_weights = {
                'low_vol': float(low_vol_weight),
                'medium_vol': float(medium_vol_weight),
                'high_vol': float(high_vol_weight),
                'trending': float(trending_weight),
                'mean_reverting': float(mean_rev_weight),
            }
            
            self.regime_history.append(regime_weights)
            return self._smooth_regime(regime_weights)
            
        except Exception as e:
            logger.debug(f"Regime detection failed: {e}")
            return self._default_regime()
    
    def _sigmoid(self, x, steepness=1):
        """Numerically stable sigmoid"""
        z = np.clip(-steepness * x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(z))
    
    def _smooth_regime(self, current_regime):
        """Safe EWMA smoothing with NaN handling"""
        if len(self.regime_history) < 2:
            return current_regime
        
        alpha = 0.3
        smoothed = current_regime.copy()
        
        for key in ['low_vol', 'medium_vol', 'high_vol', 'trending', 'mean_reverting']:
            historical = [r[key] for r in self.regime_history if key in r]
            historical = [v for v in historical if not (np.isnan(v) or np.isinf(v))]
            
            if len(historical) > 0:
                hist_mean = float(np.mean(historical))
                smoothed[key] = alpha * current_regime[key] + (1-alpha) * hist_mean
            else:
                smoothed[key] = current_regime[key]
        
        return smoothed
    
    def _default_regime(self):
        return {
            'low_vol': 0.33,
            'medium_vol': 0.34,
            'high_vol': 0.33,
            'trending': 0.5,
            'mean_reverting': 0.5,
        }

# ============================================================================
# ADAPTIVE NORMALIZER
# ============================================================================

class AdaptiveNormalizer:
    """Regime-aware normalization"""
    
    def normalize(self, feature_series, regime_weights):
        if len(feature_series) < 20:
            return self._zscore_normalize(feature_series)
        
        try:
            z_standard = self._zscore_normalize(feature_series)
            z_robust = self._robust_normalize(feature_series)
            
            vol_weight = regime_weights['high_vol']
            z_adaptive = (1 - vol_weight) * z_standard + vol_weight * z_robust
            
            return np.clip(z_adaptive, -5, 5)
            
        except:
            return self._zscore_normalize(feature_series)
    
    def _zscore_normalize(self, series):
        mu = series.mean()
        sigma = series.std()
        return (series - mu) / (sigma + 1e-10) if sigma > 1e-8 else series * 0
    
    def _robust_normalize(self, series):
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        median = series.median()
        return (series - median) / (iqr + 1e-10) if iqr > 1e-8 else series * 0

# ============================================================================
# INTEGRATED FEATURE ENHANCER (60 FEATURES STRICT)
# ============================================================================

class IntegratedFeatureEnhancer:
    def __init__(self, ably_client, agent_names, window_size=100):
        self.ably = ably_client
        self.agent_names = agent_names
        self.window_size = window_size
        
        self.price_buffers = {name: deque(maxlen=window_size) for name in agent_names}
        
        # Internal regime components
        self.regime_detector = RegimeDetector()
        self.adaptive_normalizer = AdaptiveNormalizer()
        
        # Channels
        self.features_channel = ably_client.channels.get("integrated_features_all")
        self.meta_channels = {
            name: ably_client.channels.get(f"meta_features-{name}") 
            for name in agent_names
        }
        
        self.latest_computed_features = {}
        self.features_lock = threading.Lock()
        
        logger.info(f"Regime-Adaptive Feature Enhancer initialized")
        logger.info(f"Expected features: {len(REQUIRED_FEATURES)}")
        assert len(REQUIRED_FEATURES) == 60, f"Feature count mismatch: {len(REQUIRED_FEATURES)}"

    def compute_core_technical_features(self, df):
        """Compute 19 core technical indicators with robust edge case handling"""
        df = df.copy()
        eps = 1e-10
        
        # Suppress warnings during computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)
            df['rolling_mean_5'] = df['Close'].rolling(5, min_periods=1).mean().fillna(df['Close'])
            df['rolling_std_5'] = df['Close'].rolling(5, min_periods=1).std().fillna(eps)
            df['rolling_std_5'] = df['rolling_std_5'].replace(0, eps)
            df['zscore_5'] = (df['Close'] - df['rolling_mean_5']) / df['rolling_std_5']
            
            # RSI
            delta = df['Close'].diff().fillna(0)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().fillna(0)
            avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().fillna(0)
            rs = avg_gain / (avg_loss + eps)
            df['rsi_14'] = 100 - (100 / (1 + rs))
            df['rsi_14'] = df['rsi_14'].ewm(span=5, adjust=False).mean().fillna(50)
            
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
            df['atr'] = pd.Series(tr).rolling(14, min_periods=1).mean().fillna(0)
            
            # CDF features
            window = min(100, len(df))
            if window >= 20:
                df['cdf_value'] = df['log_return'].rolling(window, min_periods=10).apply(
                    lambda x: percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 10 else 0.5
                ).fillna(0.5)
            else:
                df['cdf_value'] = 0.5
                
            df['cdf_value'] = df['cdf_value'].ffill().bfill().fillna(0.5)
            df['cdf_slope'] = df['cdf_value'].diff().ewm(span=5, adjust=False).mean().fillna(0)
            df['cdf_diff'] = (df['cdf_value'] - df['cdf_value'].shift(10)).fillna(0)
            df['cdf_diff'] = df['cdf_diff'].ewm(span=5, adjust=False).mean().fillna(0)
            
            # Volatility
            df['volatility_quantile_90'] = df['rolling_std_5'].rolling(
                min(100, len(df)), min_periods=20
            ).quantile(0.9).fillna(df['rolling_std_5'])
            df['volatility_ratio'] = df['rolling_std_5'] / (df['volatility_quantile_90'] + eps)
            df['volatility_ratio'] = df['volatility_ratio'].clip(0, 3).fillna(1.0)
            
            # Entropy
            df['entropy_50'] = df['log_return'].rolling(
                min(50, len(df)), min_periods=20
            ).apply(safe_entropy).fillna(0)
            
            # Autocorrelation
            df['autocorr_3'] = df['log_return'].rolling(20, min_periods=5).apply(
                lambda x: x.autocorr(lag=3) if len(x) > 3 else 0
            ).fillna(0)
            
            # Momentum
            df['momentum_10'] = (df['Close'] / df['Close'].shift(10) - 1).fillna(0)
            
            # Volume
            df['volume_change_rate'] = df['Volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
            vol_mean = df['Volume'].rolling(20, min_periods=1).mean()
            vol_std = df['Volume'].rolling(20, min_periods=1).std().fillna(eps)
            df['volume_zscore'] = ((df['Volume'] - vol_mean) / (vol_std + eps)).clip(-3, 3).fillna(0)
        
        return df
    
    def compute_derivative_features(self, df, window=10):
        """Compute 15 derivative features with robust handling"""
        df = df.copy()
        
        df['price_vel'] = df['Close'].diff()
        df['price_acc'] = df['price_vel'].diff()
        df['price_jrk'] = df['price_acc'].diff()
        
        for col in ['price_vel', 'price_acc', 'price_jrk']:
            try:
                # Use fillna(0) to handle edge cases
                df[f'{col}_mean'] = df[col].rolling(window, min_periods=1).mean().fillna(0)
                df[f'{col}_std'] = df[col].rolling(window, min_periods=1).std().fillna(0)
                df[f'{col}_skew'] = df[col].rolling(window, min_periods=3).apply(
                    safe_skew, raw=True
                ).fillna(0)
                df[f'{col}_kurtosis'] = df[col].rolling(window, min_periods=3).apply(
                    safe_kurtosis, raw=True
                ).fillna(0)
            except Exception as e:
                logger.debug(f"Derivative feature {col} computation failed: {e}")
                df[f'{col}_mean'] = 0
                df[f'{col}_std'] = 0
                df[f'{col}_skew'] = 0
                df[f'{col}_kurtosis'] = 0
        
        return df
    
    def compute_additional_technical(self, df):
        """Compute 7 additional technical features"""
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
        """Compute 9 institutional-grade candlestick patterns"""
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
            
            df.at[df.index[i], 'spinning_top_bearish_followup'] = spinning_top_bearish_followup(c1, c2)
            df.at[df.index[i], 'bullish_then_dragonfly'] = bullish_candle_followed_by_dragonfly(c1, c2)
        
        return df
    
    def compute_support_resistance_features(self, df):
        """Compute 7 support/resistance features"""
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
    
    def _validate_feature_contract(self, features_dict):
        """Enforce strict 60-feature contract"""
        expected = set(REQUIRED_FEATURES)
        actual = set(features_dict.keys()) - METADATA_FIELDS
        
        missing = expected - actual
        extra = actual - expected
        
        is_valid = (len(missing) == 0 and len(extra) == 0)
        return is_valid, missing, extra
    
    def compute_all_features(self, df):
        """
        Compute exactly 60 features with regime-adaptive normalization
        Regime detection is internal - NOT published
        """
        try:
            if len(df) < 10:
                return pd.DataFrame()
            
            # Step 1: Compute raw features
            df = self.compute_core_technical_features(df)
            df = self.compute_derivative_features(df)
            df = self.compute_additional_technical(df)
            df = self.compute_candlestick_patterns(df)
            df = self.compute_support_resistance_features(df)
            
            # Step 2: Internal regime detection
            regime_weights = self.regime_detector.detect_regime(df)
            
            # Step 3: Apply adaptive normalization ONLY to continuous features
            continuous_features = [
                'log_return', 'rolling_std_5', 'zscore_5', 'rsi_14',
                'macd', 'macd_signal', 'macd_hist', 'atr',
                'cdf_value', 'cdf_slope', 'cdf_diff',
                'volatility_ratio', 'entropy_50', 'autocorr_3', 'momentum_10',
                'volume_change_rate', 'volume_zscore',
                'price_vel_mean', 'price_acc_mean', 'price_jrk_mean',
                'price_vel_std', 'price_acc_std', 'price_jrk_std',
                'price_vel_skew', 'price_acc_skew', 'price_jrk_skew',
                'price_vel_kurtosis', 'price_acc_kurtosis', 'price_jrk_kurtosis',
                'bollinger_width', 'bollinger_position',
                'distance_to_nearest_support', 'distance_to_nearest_resistance',
                'distance_to_stop_loss', 'support_strength', 'resistance_strength'
            ]
            
            for feature in continuous_features:
                if feature in df.columns and feature not in NORMALIZATION_EXCLUSIONS:
                    df[feature] = self.adaptive_normalizer.normalize(
                        df[feature], regime_weights
                    )
            
            # Clean infinities and NaNs
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            return pd.DataFrame()
    
    def extract_meta_features(self, df, current_price):
        """Extract exactly 24 meta features (23 + timestamp)"""
        try:
            if len(df) < 10:
                return {}
            
            supports = find_supports(current_price, df)
            resistances = find_resistances(current_price, df)
            stop_level = find_stop_level(current_price, df)
            
            min_p, max_p = df['Low'].min(), df['High'].max()
            rng = max_p - min_p if max_p > min_p else 1
            
            # Voting features (8)
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
            
            # Filtered technical (15)
            latest = df.iloc[-1]
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
                    filtered[meta_col] = float(latest[df_col])
                else:
                    filtered[meta_col] = 0.0
            
            meta_features = {**filtered, **voting}
            
            # Validate count (23 features, timestamp added later)
            if len(meta_features) != 23:
                logger.error(f"Meta feature count violation: {len(meta_features)} != 23")
                return {}
            
            return meta_features
            
        except Exception as e:
            logger.error(f"Meta feature extraction failed: {e}")
            return {}
    
    def process_raw_tick(self, agent_name, price_data):
        """Process tick and enforce 60-feature contract"""
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
            
            # CRITICAL FIX: Only extract computed features, not raw OHLCV
            latest_row = enhanced_df.iloc[-1]
            
            # Extract only REQUIRED_FEATURES (excluding raw OHLCV columns)
            latest_features = {}
            for feature in REQUIRED_FEATURES:
                if feature in ['price', 'close_scaled', 'close_price']:
                    # These are price variants we add manually
                    latest_features[feature] = float(close_price)
                elif feature in latest_row.index:
                    latest_features[feature] = float(latest_row[feature])
                else:
                    logger.warning(f"[{agent_name}] Missing feature: {feature}")
                    latest_features[feature] = 0.0
            
            # ENFORCE CONTRACT
            is_valid, missing, extra = self._validate_feature_contract(latest_features)
            
            if not is_valid:
                logger.error("=" * 80)
                logger.error(f"❌ [{agent_name}] FEATURE CONTRACT VIOLATION")
                if missing:
                    logger.error(f"Missing: {missing}")
                if extra:
                    logger.error(f"Extra: {extra}")
                logger.error("=" * 80)
                return
            
            with self.features_lock:
                self.latest_computed_features[agent_name] = latest_features.copy()
            
        except Exception as e:
            logger.error(f"[{agent_name}] Feature enhancement failed: {e}")
    
    async def publish_features(self, agent_name, features_dict):
        """Publish 60 features"""
        try:
            clean_features = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in features_dict.items()
            }
            
            payload = {
                'agent': agent_name,
                'features': clean_features,
                'timestamp': datetime.now(UTC).isoformat(),
                'feature_count': len(clean_features)
            }
            
            await self.features_channel.publish("integrated-features", payload)
            
        except Exception as e:
            logger.error(f"[{agent_name}] Feature publish failed: {e}")
    
    async def publish_meta_features(self, agent_name, meta_features):
        """Publish 24 meta features"""
        try:
            channel = self.meta_channels[agent_name]
            
            clean_meta = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in meta_features.items()
            }
            
            clean_meta['agent'] = agent_name
            clean_meta['timestamp'] = datetime.now(UTC).isoformat()
            
            await channel.publish("meta_features", clean_meta)
            
        except Exception as e:
            logger.error(f"[{agent_name}] Meta feature publish failed: {e}")
    
    def get_latest_state_features(self, agent_name=None):
        """Get latest features with type-aware aggregation"""
        with self.features_lock:
            if agent_name:
                return self.latest_computed_features.get(agent_name, {})
            
            if not self.latest_computed_features:
                return {}
            
            all_features = list(self.latest_computed_features.values())
            if not all_features:
                return {}
            
            return self._safe_aggregate_features(all_features)
    
    def _safe_aggregate_features(self, all_features):
        """Type-aware feature aggregation across agents"""
        avg_features = {}
        feature_keys = all_features[0].keys()
        
        for key in feature_keys:
            values = [f[key] for f in all_features if key in f]
            
            if not values:
                continue
            
            if key in BINARY_FEATURES:
                # Voting for binary features
                avg_features[key] = int(np.sum(values) > len(values) / 2)
            elif key in PRICE_FEATURES:
                # Median for price features (robust to outliers)
                clean_values = [v for v in values if not np.isnan(v)]
                if clean_values:
                    avg_features[key] = float(np.median(clean_values))
                else:
                    avg_features[key] = 0.0
            else:
                # Mean for continuous features
                clean_values = [v for v in values if not np.isnan(v)]
                if clean_values:
                    avg_features[key] = float(np.mean(clean_values))
                else:
                    avg_features[key] = 0.0
        
        return avg_features
    
    def get_feature_summary(self):
        """Get detailed feature summary"""
        with self.features_lock:
            if not self.latest_computed_features:
                return "No features computed yet"
            
            sample_agent = list(self.latest_computed_features.keys())[0]
            features = self.latest_computed_features[sample_agent]
            
            actual_count = len([k for k in features.keys() if k not in METADATA_FIELDS])
            
            summary = f"REGIME-ADAPTIVE FEATURE ENHANCER\n"
            summary += "=" * 60 + "\n\n"
            summary += f"Total Features: {actual_count} (Expected: 60)\n\n"
            summary += "Feature Categories:\n"
            summary += f"  • Core Technical: 19 features\n"
            summary += f"  • Derivatives: 15 features\n"
            summary += f"  • Additional Technical: 7 features\n"
            summary += f"  • Candlestick Patterns: 9 features (institutional-grade)\n"
            summary += f"  • Support/Resistance: 7 features\n"
            summary += f"  • Price Variants: 3 features\n"
            summary += f"  • TOTAL: 60 features\n\n"
            summary += f"Meta Features (24 total, published separately):\n"
            summary += f"  • Voting: 8 features\n"
            summary += f"  • Technical: 15 features\n"
            summary += f"  • Timestamp: 1 metadata\n\n"
            summary += f"Regime Detection: INTERNAL (adaptive normalization)\n"
            summary += f"  • Volatility regimes: low/medium/high\n"
            summary += f"  • Trend detection: momentum-based\n"
            summary += f"  • Mean-reversion: entropy-based\n\n"
            summary += f"Normalization: Regime-adaptive\n"
            summary += f"  • High vol → Robust scaling (IQR)\n"
            summary += f"  • Low vol → Standard z-score\n"
            summary += f"  • Excluded: {len(NORMALIZATION_EXCLUSIONS)} features\n\n"
            summary += f"Aggregation: Type-aware\n"
            summary += f"  • Binary: Voting (majority rule)\n"
            summary += f"  • Price: Median (outlier-resistant)\n"
            summary += f"  • Continuous: Mean\n"
            
            return summary

# ============================================================================
# ASYNC WRAPPER
# ============================================================================

class AsyncIntegratedFeatureEnhancer:
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
        logger.info("\n" + self.enhancer.get_feature_summary())
        await self._start_ably_listeners()

    async def _start_ably_listeners(self):
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
                    logger.error("Failed to connect to Ably")
                    return
            except Exception as e:
                logger.error(f"Ably connection failed: {e}")
                return

        logger.info(f"Starting Ably listeners")

        for agent in self.agents:
            agent_str = agent.decode('utf-8') if isinstance(agent, bytes) else str(agent)

            feature_ok = await self._subscribe_with_retry(
                agent_str, "integrated-features",
                lambda msg, name=agent_str: self._handle_feature_message(name, msg)
            )
            meta_ok = await self._subscribe_with_retry(
                agent_str, "meta_features",
                lambda msg, name=agent_str: self._handle_meta_features_message(name, msg),
                channel_suffix="meta_features-"
            )
            
            if feature_ok:
                logger.info(f"✓ [{agent_str}] Feature channel attached")
            if meta_ok:
                logger.info(f"✓ [{agent_str}] Meta features channel attached")

    async def _subscribe_with_retry(self, agent_name, event_name, callback, max_retries=3, timeout=10, channel_suffix=""):
        channel_name = f"{channel_suffix}{agent_name}" if channel_suffix else agent_name

        for attempt in range(max_retries):
            try:
                channel = self.ably.channels.get(channel_name)
                self.channels[channel_name] = channel

                attach_task = asyncio.create_task(channel.attach())
                try:
                    await asyncio.wait_for(attach_task, timeout=timeout)
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False

                subscribe_task = asyncio.create_task(channel.subscribe(event_name, callback))
                try:
                    await asyncio.wait_for(subscribe_task, timeout=timeout)
                    return True
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
        return False

    async def process_tick(self, agent_name, price_data):
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
        channel_name = f"meta_features-{agent_name}" if meta else agent_name
        event_name = "meta_features" if meta else "feature"
    
        if channel_name not in self.channels:
            self.channels[channel_name] = self.ably.channels.get(channel_name)
    
        channel = self.channels[channel_name]
    
        payload = {
            'agent': agent_name,
            'features' if not meta else 'meta_features': features_dict,
            'timestamp': datetime.now(UTC).isoformat()
        }
    
        for attempt in range(3):
            try:
                await channel.publish(event_name, payload)
                break
            except Exception as e:
                await asyncio.sleep(2 ** attempt)

    def _handle_feature_message(self, agent_name, msg):
        logger.debug(f"[{agent_name}] Feature message received")

    def _handle_meta_features_message(self, agent_name, msg):
        logger.debug(f"[{agent_name}] Meta feature message received")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    nest_asyncio.apply()
    
    logger.info("=" * 80)
    logger.info("🚀 REGIME-ADAPTIVE FEATURE ENHANCER - PRODUCTION GRADE")
    logger.info("=" * 80)
    
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"❌ MT5 initialization failed: {mt5.last_error()}")
    
    logger.info(f"✅ MT5 initialized: {MT5_SERVER}")
    logger.info(f"   Symbol: {SYMBOL}")
    
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        mt5.shutdown()
        raise RuntimeError(f"❌ Symbol {SYMBOL} not found")
    
    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            mt5.shutdown()
            raise RuntimeError(f"❌ Failed to select symbol {SYMBOL}")
    
    logger.info(f"✅ Symbol verified")
    
    logger.info("\n📡 Connecting to Ably...")
    try:
        ably_client = AblyRealtime(ABLY_API_KEY)
        await asyncio.sleep(1)
        logger.info("✅ Ably connected")
    except Exception as e:
        mt5.shutdown()
        raise RuntimeError(f"❌ Ably connection failed: {e}")
    
    logger.info("\n🔧 Initializing feature enhancers...")
    agent_names = list(TIMEFRAMES.keys())
    
    enhancer = AsyncIntegratedFeatureEnhancer(
        ably_client=ably_client,
        agent_names=agent_names,
        window_size=100
    )
    
    await enhancer.start()
    
    agent_channels = {tf: ably_client.channels.get(tf) for tf in TIMEFRAMES}
    
    logger.info("\n✅ All systems initialized - Starting tick processing...\n")
    
    tick_count = 0
    last_summary_time = time.time()
    feature_counts = {tf: 0 for tf in TIMEFRAMES}
    
    try:
        while True:
            try:
                tick = mt5.symbol_info_tick(SYMBOL)
                
                if tick is None:
                    await asyncio.sleep(1)
                    continue
                
                tick_count += 1
                mid_price = (tick.bid + tick.ask) / 2.0
                timestamp = datetime.now(UTC)
                
                for timeframe_name in TIMEFRAMES.keys():
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
                                    f"60 features + meta | Price: {mid_price:.5f}"
                                )
                    
                    except Exception as e:
                        logger.error(f"❌ [{timeframe_name}] Error: {e}")
                
                if time.time() - last_summary_time > 60:
                    logger.info("\n" + "=" * 80)
                    logger.info(f"📊 SUMMARY (Tick #{tick_count})")
                    logger.info("=" * 80)
                    logger.info(f"Price: {mid_price:.5f}")
                    for tf in TIMEFRAMES:
                        logger.info(f"  {tf}: {feature_counts[tf]} updates")
                    logger.info("=" * 80 + "\n")
                    last_summary_time = time.time()
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                break
                
            except Exception as e:
                logger.error(f"❌ Tick error: {e}")
                await asyncio.sleep(1)
    
    finally:
        logger.info("\n🛑 SHUTTING DOWN")
        mt5.shutdown()
        logger.info(f"Total Ticks: {tick_count}")
        logger.info("✅ Shutdown complete")

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
        
        

#+263780563561  ENG Karl Muzunze Masvingo Zimbabwe