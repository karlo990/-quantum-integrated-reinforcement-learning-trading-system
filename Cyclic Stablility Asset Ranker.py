import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import json
import time
from pathlib import Path

# ===============================================================
# 📈 CONFIGURATION
# ===============================================================
class Config:
    # MT5 Connection Settings

    
    MT5_LOGIN = 	40014331
    MT5_PASSWORD = "W0*9-9=Q+Eo+"
    MT5_SERVER = "RCGMarkets-Real"
    
    ASSETS = [
        # ---- Major Currency Pairs ----
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    
        # ---- Cross Currency Pairs ----
        "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "AUDCAD",
    
        # ---- Precious Metals ----
        "XAUUSD",  # Gold
        "XAGUSD",  # Silver
        "XPTUSD",  # Platinum
    
     
    ]

    TIMEFRAME = mt5.TIMEFRAME_M5
    CANDLES = 1000
    
    # Buffering Settings
    BUFFER_SIZE = 100  # Keep last 100 rankings
    SCAN_INTERVAL = 300  # Scan every 5 minutes (in seconds)
    
    # Scoring Weights
    WEIGHTS = {
        'cycle': 0.25,
        'trend': 0.25,
        'volatility': 0.20,
        'liquidity': 0.15,
        'pullback': 0.15
    }
    
    # Alert Settings
    ALERT_ON_RANK_CHANGE = True
    ALERT_THRESHOLD = 0.05  # Alert if score changes by more than 5%
    
    # File Settings
    HISTORY_FILE = "asset_rankings_history.json"
    LOG_FILE = "asset_ranker.log"

# ===============================================================
# 🗄️ BUFFER MANAGER
# ===============================================================
class RankingBuffer:
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)
        self.history_file = Config.HISTORY_FILE
        self.load_history()
    
    def add_ranking(self, ranking_data):
        """Add new ranking to buffer"""
        ranking_data['timestamp'] = datetime.now().isoformat()
        self.buffer.append(ranking_data)
        self.save_history()
    
    def get_recent_rankings(self, n=10):
        """Get last n rankings"""
        return list(self.buffer)[-n:]
    
    def get_asset_history(self, asset_name, n=20):
        """Get historical scores for a specific asset"""
        history = []
        for ranking in list(self.buffer)[-n:]:
            asset_data = next((item for item in ranking['rankings'] 
                             if item['Asset'] == asset_name), None)
            if asset_data:
                history.append({
                    'timestamp': ranking['timestamp'],
                    'score': asset_data['final'],
                    'rank': asset_data.get('rank', None)
                })
        return history
    
    def detect_rank_changes(self):
        """Detect if top asset has changed"""
        if len(self.buffer) < 2:
            return None
        
        current = self.buffer[-1]['rankings'][0]['Asset']
        previous = self.buffer[-2]['rankings'][0]['Asset']
        
        if current != previous:
            return {
                'previous_top': previous,
                'new_top': current,
                'timestamp': self.buffer[-1]['timestamp']
            }
        return None
    
    def get_score_volatility(self, asset_name, n=10):
        """Calculate score volatility for an asset"""
        history = self.get_asset_history(asset_name, n)
        if len(history) < 2:
            return 0
        scores = [h['score'] for h in history]
        return np.std(scores)
    
    def save_history(self):
        """Save buffer to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(list(self.buffer), f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save history: {e}")
    
    def load_history(self):
        """Load buffer from file"""
        try:
            if Path(self.history_file).exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.buffer = deque(data, maxlen=self.buffer.maxlen)
                print(f"📚 Loaded {len(self.buffer)} historical rankings")
        except Exception as e:
            print(f"⚠️ Failed to load history: {e}")

# ===============================================================
# 🧩 HELPER FUNCTIONS
# ===============================================================
def get_data(symbol, candles=Config.CANDLES, timeframe=Config.TIMEFRAME):
    """Fetch OHLC data from MT5"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, candles)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data received for {symbol}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def sign_changes(series):
    """Count directional changes in price"""
    diff = np.sign(np.diff(series))
    return np.sum(diff[1:] != diff[:-1])

def adx(df, period=14):
    """Calculate Average Directional Index"""
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'],
                   np.maximum(abs(df['high'] - df['close'].shift()),
                              abs(df['low'] - df['close'].shift())))
    df['up_move'] = df['high'] - df['high'].shift()
    df['down_move'] = df['low'].shift() - df['low']
    df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), 
                         df['up_move'], 0.0)
    df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), 
                         df['down_move'], 0.0)
    tr14 = df['tr'].rolling(window=period).sum()
    plus14 = df['+dm'].rolling(window=period).sum()
    minus14 = df['-dm'].rolling(window=period).sum()
    plusDI = 100 * (plus14 / tr14)
    minusDI = 100 * (minus14 / tr14)
    dx = 100 * abs(plusDI - minusDI) / (plusDI + minusDI + 1e-6)
    adx_value = dx.rolling(window=period).mean().iloc[-1]
    return adx_value if not pd.isna(adx_value) else 0

def atr(df, period=14):
    """Calculate Average True Range"""
    tr = np.maximum(df['high'] - df['low'],
             np.maximum(abs(df['high'] - df['close'].shift()),
                        abs(df['low'] - df['close'].shift())))
    atr_value = tr.rolling(period).mean().iloc[-1]
    return atr_value if not pd.isna(atr_value) else 0

def body_ratio(df):
    """Calculate average body-to-range ratio"""
    ratio = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
    return np.mean(ratio)

def rsi(df, period=14):
    """Calculate Relative Strength Index"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-6)
    rsi_value = 100 - (100 / (1 + rs))
    return rsi_value.iloc[-1] if not pd.isna(rsi_value.iloc[-1]) else 50

# ===============================================================
# 🧮 SCORING FUNCTIONS
# ===============================================================
def compute_scores(df):
    """Calculate all scoring metrics for an asset"""
    scores = {}
    
    # 1. Cycle Stability
    changes = sign_changes(df['close'])
    scores['cycle'] = 1 / (1 + changes / (len(df) / 100))
    
    # 2. Trend Strength
    scores['trend'] = adx(df) / 100
    
    # 3. Volatility Control
    vol = atr(df) / (np.mean(df['close']) + 1e-6)
    target_vol = 0.01
    scores['volatility'] = 1 / (1 + abs(vol - target_vol) * 10)
    
    # 4. Liquidity / Body Ratio
    scores['liquidity'] = body_ratio(df)
    
    # 5. Pullback Structure
    ema_fast = df['close'].ewm(span=10).mean()
    ema_slow = df['close'].ewm(span=30).mean()
    retracement_ratio = abs(ema_fast - ema_slow) / (df['close'] + 1e-6)
    structured_pullbacks = np.sum((retracement_ratio > 0.002) & 
                                 (retracement_ratio < 0.01))
    scores['pullback'] = structured_pullbacks / len(df)
    
    # 6. Additional: Momentum (RSI deviation from 50)
    rsi_val = rsi(df)
    scores['momentum'] = 1 - abs(rsi_val - 50) / 50
    
    return scores

def calculate_final_score(scores):
    """Calculate weighted final score"""
    return (
        Config.WEIGHTS['cycle'] * scores['cycle'] +
        Config.WEIGHTS['trend'] * scores['trend'] +
        Config.WEIGHTS['volatility'] * scores['volatility'] +
        Config.WEIGHTS['liquidity'] * scores['liquidity'] +
        Config.WEIGHTS['pullback'] * scores['pullback']
    )

# ===============================================================
# 📊 ANALYSIS ENGINE
# ===============================================================
class AssetRanker:
    def __init__(self):
        self.buffer = RankingBuffer(max_size=Config.BUFFER_SIZE)
        self.initialize_mt5()
    
    def initialize_mt5(self):
        """Initialize MT5 connection with authentication"""
        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            print(f"Error code: {mt5.last_error()}")
            raise ConnectionError("MT5 initialization failed")
        
        # Login to MT5 account
        authorized = mt5.login(
            login=Config.MT5_LOGIN,
            password=Config.MT5_PASSWORD,
            server=Config.MT5_SERVER
        )
        
        if not authorized:
            error = mt5.last_error()
            mt5.shutdown()
            print(f"❌ MT5 login failed")
            print(f"Error code: {error}")
            raise ConnectionError(f"MT5 login failed: {error}")
        
        # Display account info
        account_info = mt5.account_info()
        if account_info:
            print("✅ MT5 connected successfully")
            print(f"   Account: {account_info.login}")
            print(f"   Server: {account_info.server}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Leverage: 1:{account_info.leverage}")
        else:
            print("✅ MT5 connected (account info unavailable)")
    
    def analyze_asset(self, symbol):
        """Analyze a single asset"""
        try:
            df = get_data(symbol)
            scores = compute_scores(df)
            final_score = calculate_final_score(scores)
            
            return {
                "Asset": symbol,
                **scores,
                "final": final_score,
                "last_price": df['close'].iloc[-1],
                "data_points": len(df)
            }
        except Exception as e:
            print(f"⚠️ Error analyzing {symbol}: {e}")
            return None
    
    def scan_all_assets(self):
        """Scan and rank all assets"""
        print(f"\n{'='*60}")
        print(f"🔍 Scanning Assets - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        results = []
        for sym in Config.ASSETS:
            print(f"  ⏳ Analyzing {sym}...")
            result = self.analyze_asset(sym)
            if result:
                results.append(result)
        
        if not results:
            print("❌ No valid results obtained")
            return None
        
        # Sort by final score
        df_result = pd.DataFrame(results).sort_values(
            by='final', ascending=False
        ).reset_index(drop=True)
        
        # Add rank column
        df_result['rank'] = range(1, len(df_result) + 1)
        
        # Store in buffer
        ranking_data = {
            'rankings': df_result.to_dict('records')
        }
        self.buffer.add_ranking(ranking_data)
        
        # Check for rank changes
        rank_change = self.buffer.detect_rank_changes()
        
        # Display results
        self.display_results(df_result, rank_change)
        
        return df_result
    
    def display_results(self, df_result, rank_change=None):
        """Display ranking results"""
        print("\n🏆 Current Rankings:")
        print("-" * 100)
        display_cols = ['rank', 'Asset', 'final', 'cycle', 'trend', 
                       'volatility', 'liquidity', 'pullback', 'momentum']
        print(df_result[display_cols].to_string(index=False, 
                                                float_format=lambda x: f'{x:.4f}'))
        print("-" * 100)
        
        best = df_result.iloc[0]['Asset']
        best_score = df_result.iloc[0]['final']
        print(f"\n✅ Top Ranked Asset: {best} (Score: {best_score:.4f})")
        
        if rank_change and Config.ALERT_ON_RANK_CHANGE:
            print(f"\n🚨 RANK CHANGE ALERT!")
            print(f"   Previous Top: {rank_change['previous_top']}")
            print(f"   New Top: {rank_change['new_top']}")
        
        # Show score trends
        print(f"\n📈 Recent Score Trends (Top 3):")
        for i in range(min(3, len(df_result))):
            asset = df_result.iloc[i]['Asset']
            history = self.buffer.get_asset_history(asset, n=5)
            if len(history) > 1:
                scores = [h['score'] for h in history]
                trend = "📈" if scores[-1] > scores[0] else "📉"
                volatility = self.buffer.get_score_volatility(asset, n=10)
                print(f"   {trend} {asset}: {scores[-1]:.4f} "
                      f"(σ={volatility:.4f})")
    
    def run_continuous(self):
        """Run continuous monitoring"""
        print(f"\n🔄 Starting continuous monitoring...")
        print(f"   Scan interval: {Config.SCAN_INTERVAL} seconds")
        print(f"   Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.scan_all_assets()
                print(f"\n⏸️  Waiting {Config.SCAN_INTERVAL} seconds until next scan...")
                time.sleep(Config.SCAN_INTERVAL)
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
        finally:
            self.shutdown()
    
    def get_statistics(self):
        """Get historical statistics"""
        if len(self.buffer.buffer) == 0:
            print("No historical data available")
            return
        
        print(f"\n📊 Historical Statistics:")
        print(f"   Total scans: {len(self.buffer.buffer)}")
        
        # Count how many times each asset was #1
        top_counts = {}
        for ranking in self.buffer.buffer:
            top_asset = ranking['rankings'][0]['Asset']
            top_counts[top_asset] = top_counts.get(top_asset, 0) + 1
        
        print(f"\n   Times Ranked #1:")
        for asset, count in sorted(top_counts.items(), 
                                   key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.buffer.buffer)) * 100
            print(f"      {asset}: {count} ({percentage:.1f}%)")
    
    def shutdown(self):
        """Cleanup and shutdown"""
        mt5.shutdown()
        print("✅ MT5 connection closed")

# ===============================================================
# 🚀 MAIN EXECUTION
# ===============================================================
def main():
    ranker = AssetRanker()
    
    # Run initial scan
    ranker.scan_all_assets()
    
    # Show statistics if we have history
    if len(ranker.buffer.buffer) > 1:
        ranker.get_statistics()
    
    # Ask user if they want continuous monitoring
    print("\n" + "="*60)
    choice = input("Run continuous monitoring? (y/n): ").strip().lower()
    
    if choice == 'y':
        ranker.run_continuous()
    else:
        ranker.shutdown()
        print("\n✅ Single scan completed")

if __name__ == "__main__":
    main()