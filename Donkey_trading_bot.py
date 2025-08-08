import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import json
import talib
from typing import Dict, List, Tuple, Optional
import logging
import sqlite3

import os
from dotenv import load_dotenv

# Load environment variables for deployment
load_dotenv()

class TradingStrategyAnalyzer:
    """Analyzes traditional trading strategies to create the Donkey Strategy parameters"""
    
    def __init__(self):
        self.strategies = {
            'support_resistance': {
                'entry_offset': 0.0005,  # 5 pips from S/R level
                'tp_ratio': 2.0,         # 2:1 RR
                'sl_ratio': 1.0
            },
            'chart_patterns': {
                'entry_offset': 0.0008,  # 8 pips for pattern confirmation
                'tp_ratio': 1.8,
                'sl_ratio': 1.0
            },
            'indicators': {
                'entry_offset': 0.0003,  # 3 pips for indicator signal
                'tp_ratio': 1.5,
                'sl_ratio': 1.0
            },
            'smart_money': {
                'entry_offset': 0.0012,  # 12 pips for accumulation zone
                'tp_ratio': 3.0,
                'sl_ratio': 1.0
            },
            'ict_concepts': {
                'entry_offset': 0.0010,  # 10 pips for order blocks
                'tp_ratio': 2.5,
                'sl_ratio': 1.0
            }
        }
    
    def calculate_donkey_parameters(self) -> Dict:
        """Calculate average entry, TP, and SL from all strategies"""
        total_strategies = len(self.strategies)
        
        # Calculate averages
        avg_entry_offset = sum(s['entry_offset'] for s in self.strategies.values()) / total_strategies
        avg_tp_ratio = sum(s['tp_ratio'] for s in self.strategies.values()) / total_strategies
        avg_sl_ratio = sum(s['sl_ratio'] for s in self.strategies.values()) / total_strategies
        
        # Donkey Strategy: Entry at average, TP at 80% of average SL distance
        donkey_params = {
            'entry_offset': round(avg_entry_offset, 6),
            'tp_distance': round(avg_sl_ratio * 0.8, 6),  # 80% of average SL
            'sl_distance': round(avg_sl_ratio * 0.3, 6),  # Tight SL (30% of normal)
            'original_avg_tp': round(avg_tp_ratio, 2),
            'original_avg_sl': round(avg_sl_ratio, 2)
        }
        
        return donkey_params

class NewsDetector:
    """Detects high-impact news events"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.high_impact_keywords = [
            'NFP', 'Interest Rate', 'GDP', 'Inflation', 'CPI', 'PPI',
            'Employment', 'FOMC', 'ECB', 'BOJ', 'BOE', 'Fed'
        ]
    
    def is_high_volatility_time(self) -> Tuple[bool, str]:
        """Check if current time is during high volatility news"""
        now = datetime.now()
        
        # Major news release times (UTC)
        news_times = [
            (8, 30),   # EUR news
            (12, 30),  # GBP news
            (13, 30),  # USD news
            (23, 30),  # JPY news
        ]
        
        current_time = (now.hour, now.minute)
        
        for news_hour, news_minute in news_times:
            if (abs(current_time[0] - news_hour) < 1 and 
                abs(current_time[1] - news_minute) < 30):
                return True, f"High volatility period detected at {news_hour}:{news_minute:02d} UTC"
        
        return False, ""

class NotificationManager:
    """Handles all notifications (email, telegram, etc.)"""
    
    def __init__(self, email_config: Dict = None, telegram_config: Dict = None):
        self.email_config = email_config
        self.telegram_config = telegram_config
        
    def send_email(self, subject: str, message: str):
        """Send email notification"""
        if not self.email_config:
            return
            
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = subject
            
            msg.attach(MimeText(message, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['from_email'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logging.error(f"Email notification failed: {e}")
    
    def send_telegram(self, message: str):
        """Send Telegram notification"""
        if not self.telegram_config:
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            data = {
                'chat_id': self.telegram_config['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            requests.post(url, data=data)
        except Exception as e:
            logging.error(f"Telegram notification failed: {e}")
    
    def notify(self, subject: str, message: str):
        """Send notification via all configured methods"""
        self.send_email(subject, message)
        self.send_telegram(f"<b>{subject}</b>\n{message}")

class RiskManager:
    """Manages risk and position sizing"""
    
    def __init__(self, max_daily_loss_percent: float = 3.0):
        self.max_daily_loss_percent = max_daily_loss_percent
        self.daily_start_balance = 0
        self.current_daily_loss = 0
        self.last_reset_date = datetime.now().date()
    
    def reset_daily_tracking(self, current_balance: float):
        """Reset daily loss tracking"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_start_balance = current_balance
            self.current_daily_loss = 0
            self.last_reset_date = today
    
    def can_trade(self, current_balance: float) -> bool:
        """Check if trading is allowed based on daily loss limit"""
        self.reset_daily_tracking(current_balance)
        
        daily_loss_percent = (self.current_daily_loss / self.daily_start_balance) * 100
        return daily_loss_percent < self.max_daily_loss_percent
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float = 1.0) -> float:
        """Calculate position size based on account balance and risk"""
        return (account_balance * risk_per_trade / 100)
    
    def update_daily_loss(self, loss_amount: float):
        """Update current daily loss"""
        if loss_amount > 0:
            self.current_daily_loss += loss_amount

class TechnicalAnalyzer:
    """Performs technical analysis for entry signals"""
    
    def __init__(self):
        self.timeframes = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15]
    
    def get_market_data(self, symbol: str, timeframe: int, count: int = 100) -> pd.DataFrame:
        """Get historical market data"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        if df.empty:
            return {'support': 0, 'resistance': 0}
        
        # Simple pivot point calculation
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        support = 2 * pivot - high
        resistance = 2 * pivot - low
        
        return {'support': support, 'resistance': resistance, 'pivot': pivot}
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect chart patterns using price action"""
        if len(df) < 20:
            return {'pattern': 'none', 'signal': 'none'}
        
        # Simple pattern detection based on recent price action
        recent_highs = df['high'].tail(10).max()
        recent_lows = df['low'].tail(10).min()
        current_price = df['close'].iloc[-1]
        
        # Bullish pattern detection
        if current_price > (recent_lows + (recent_highs - recent_lows) * 0.7):
            return {'pattern': 'bullish_breakout', 'signal': 'buy'}
        
        # Bearish pattern detection
        if current_price < (recent_lows + (recent_highs - recent_lows) * 0.3):
            return {'pattern': 'bearish_breakdown', 'signal': 'sell'}
        
        return {'pattern': 'consolidation', 'signal': 'none'}
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        if len(df) < 50:
            return {}
        
        close_prices = df['close'].values
        
        indicators = {}
        
        try:
            # RSI
            indicators['rsi'] = talib.RSI(close_prices, timeperiod=14)[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macd_signal[-1]
            indicators['macd_histogram'] = macd_hist[-1]
            
            # Moving Averages
            indicators['sma_20'] = talib.SMA(close_prices, timeperiod=20)[-1]
            indicators['sma_50'] = talib.SMA(close_prices, timeperiod=50)[-1]
            indicators['ema_12'] = talib.EMA(close_prices, timeperiod=12)[-1]
            indicators['ema_26'] = talib.EMA(close_prices, timeperiod=26)[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            
        except Exception as e:
            logging.error(f"Indicator calculation error: {e}")
        
        return indicators
    
    def generate_signal(self, symbol: str) -> Dict:
        """Generate trading signal using Donkey Strategy logic"""
        signal = {
            'symbol': symbol,
            'action': 'none',
            'entry_price': 0,
            'take_profit': 0,
            'stop_loss': 0,
            'confidence': 0,
            'analysis': {}
        }
        
        # Get data for multiple timeframes
        for timeframe in self.timeframes:
            df = self.get_market_data(symbol, timeframe)
            if df.empty:
                continue
            
            # Technical analysis
            sr_levels = self.calculate_support_resistance(df)
            patterns = self.detect_patterns(df)
            indicators = self.calculate_indicators(df)
            
            current_price = df['close'].iloc[-1]
            
            # Donkey Strategy Logic
            # Entry: Average of all strategy entry points
            # TP: 80% of average stop loss distance
            # SL: Very tight (30% of normal)
            
            donkey_params = TradingStrategyAnalyzer().calculate_donkey_parameters()
            
            # Calculate entry points based on support/resistance and patterns
            if patterns['signal'] == 'buy':
                entry_price = current_price + donkey_params['entry_offset']
                take_profit = entry_price + donkey_params['tp_distance']
                stop_loss = entry_price - donkey_params['sl_distance']
                action = 'buy'
                confidence = 75
                
            elif patterns['signal'] == 'sell':
                entry_price = current_price - donkey_params['entry_offset']
                take_profit = entry_price - donkey_params['tp_distance']
                stop_loss = entry_price + donkey_params['sl_distance']
                action = 'sell'
                confidence = 75
            else:
                continue
            
            # Update signal with highest confidence timeframe
            if confidence > signal['confidence']:
                signal.update({
                    'action': action,
                    'entry_price': round(entry_price, 5),
                    'take_profit': round(take_profit, 5),
                    'stop_loss': round(stop_loss, 5),
                    'confidence': confidence,
                    'analysis': {
                        'timeframe': timeframe,
                        'support_resistance': sr_levels,
                        'patterns': patterns,
                        'indicators': indicators,
                        'donkey_params': donkey_params
                    }
                })
        
        return signal

class DonkeyTradingBot:
    """Main trading bot class implementing the Donkey Strategy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        self.risk_manager = RiskManager(config.get('max_daily_loss_percent', 3.0))
        self.notification_manager = NotificationManager(
            config.get('email_config'),
            config.get('telegram_config')
        )
        self.technical_analyzer = TechnicalAnalyzer()
        self.news_detector = NewsDetector(config.get('news_api_key'))
        
        # Trading symbols
        self.symbols = config.get('symbols', [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURJPY', 'GBPJPY', 'EURGBP', 'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD'
        ])
        
        # Database for trade tracking
        self.init_database()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('donkey_bot.log'),
                logging.StreamHandler()
            ]
        )
    
    def init_database(self):
        """Initialize SQLite database for trade tracking"""
        conn = sqlite3.connect('donkey_trades.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                action TEXT,
                entry_price REAL,
                take_profit REAL,
                stop_loss REAL,
                volume REAL,
                status TEXT,
                profit_loss REAL,
                mt5_ticket INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 platform"""
        if not mt5.initialize():
            logging.error("MT5 initialization failed")
            return False
        
        # Login to account
        if not mt5.login(
            login=self.config['mt5_login'],
            password=self.config['mt5_password'],
            server=self.config['mt5_server']
        ):
            logging.error("MT5 login failed")
            return False
        
        logging.info("Successfully connected to MT5")
        return True
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        account_info = mt5.account_info()
        if account_info is None:
            return {}
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'profit': account_info.profit
        }
    
    def place_order(self, signal: Dict) -> bool:
        """Place order based on signal"""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return False
            
            # Check risk management
            if not self.risk_manager.can_trade(account_info['balance']):
                logging.warning("Daily loss limit reached. Trading suspended.")
                return False
            
            # Calculate position size
            risk_amount = self.risk_manager.calculate_position_size(account_info['balance'])
            
            # Get symbol info
            symbol_info = mt5.symbol_info(signal['symbol'])
            if symbol_info is None:
                logging.error(f"Symbol {signal['symbol']} not found")
                return False
            
            # Calculate lot size
            point = symbol_info.point
            tick_value = symbol_info.trade_tick_value
            sl_distance = abs(signal['entry_price'] - signal['stop_loss'])
            lot_size = round(risk_amount / (sl_distance * tick_value * 100000), 2)
            
            # Minimum lot size check
            lot_size = max(lot_size, symbol_info.volume_min)
            lot_size = min(lot_size, symbol_info.volume_max)
            
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if signal['action'] == 'buy' else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal['symbol'],
                "volume": lot_size,
                "type": order_type,
                "price": signal['entry_price'],
                "sl": signal['stop_loss'],
                "tp": signal['take_profit'],
                "deviation": 20,
                "magic": 987654321,
                "comment": "Donkey Strategy Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Save trade to database
                self.save_trade_to_db(signal, lot_size, result.order)
                
                # Send notification
                message = f"""
🚀 DONKEY STRATEGY TRADE OPENED
Symbol: {signal['symbol']}
Action: {signal['action'].upper()}
Entry: {signal['entry_price']}
Take Profit: {signal['take_profit']}
Stop Loss: {signal['stop_loss']}
Volume: {lot_size}
Ticket: {result.order}
Confidence: {signal['confidence']}%
                """
                
                self.notification_manager.notify("New Donkey Trade", message)
                logging.info(f"Order placed successfully: {result.order}")
                return True
            else:
                logging.error(f"Order failed: {result.retcode} - {result.comment}")
                return False
                
        except Exception as e:
            logging.error(f"Order placement error: {e}")
            return False
    
    def save_trade_to_db(self, signal: Dict, volume: float, ticket: int):
        """Save trade information to database"""
        conn = sqlite3.connect('donkey_trades.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, action, entry_price, take_profit, 
                              stop_loss, volume, status, profit_loss, mt5_ticket)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            signal['symbol'],
            signal['action'],
            signal['entry_price'],
            signal['take_profit'],
            signal['stop_loss'],
            volume,
            'open',
            0.0,
            ticket
        ))
        
        conn.commit()
        conn.close()
    
    def check_open_positions(self):
        """Monitor and update open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return
        
        for position in positions:
            if position.magic == 987654321:  # Our bot's magic number
                # Check if position was closed
                if position.profit != 0:
                    # Update database
                    conn = sqlite3.connect('donkey_trades.db')
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        UPDATE trades SET status = ?, profit_loss = ?
                        WHERE mt5_ticket = ?
                    ''', ('closed', position.profit, position.ticket))
                    
                    conn.commit()
                    conn.close()
                    
                    # Update risk manager
                    if position.profit < 0:
                        self.risk_manager.update_daily_loss(abs(position.profit))
                    
                    # Send notification
                    status = "✅ PROFIT" if position.profit > 0 else "❌ LOSS"
                    message = f"""
{status} - DONKEY TRADE CLOSED
Symbol: {position.symbol}
Profit/Loss: ${position.profit:.2f}
Volume: {position.volume}
Ticket: {position.ticket}
                    """
                    
                    self.notification_manager.notify("Trade Closed", message)
    
    def scan_for_signals(self):
        """Scan all symbols for trading signals"""
        signals = []
        
        for symbol in self.symbols:
            try:
                signal = self.technical_analyzer.generate_signal(symbol)
                
                if signal['action'] != 'none':
                    # Check for high volatility news
                    is_volatile, news_msg = self.news_detector.is_high_volatility_time()
                    
                    if is_volatile:
                        volatility_message = f"""
⚠️ HIGH VOLATILITY DETECTED
{news_msg}
Symbol: {signal['symbol']}
Proposed Entry: {signal['entry_price']}
Take Profit: {signal['take_profit']}
Stop Loss: {signal['stop_loss']}
Confidence: {signal['confidence']}%
                        """
                        self.notification_manager.notify("High Volatility Trade", volatility_message)
                    
                    signals.append(signal)
                    
            except Exception as e:
                logging.error(f"Signal generation error for {symbol}: {e}")
        
        return signals
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            # Check open positions
            self.check_open_positions()
            
            # Scan for new signals
            signals = self.scan_for_signals()
            
            # Place orders for valid signals
            for signal in signals:
                if signal['confidence'] >= 70:  # Minimum confidence threshold
                    self.place_order(signal)
                    time.sleep(1)  # Small delay between orders
            
        except Exception as e:
            logging.error(f"Trading cycle error: {e}")
    
    def start(self):
        """Start the trading bot"""
        if not self.connect_mt5():
            return False
        
        self.is_running = True
        
        # Send startup notification
        self.notification_manager.notify(
            "Donkey Bot Started", 
            "🐴 Donkey Strategy Bot is now running and monitoring markets!"
        )
        
        logging.info("Donkey Trading Bot started successfully")
        
        # Main trading loop
        while self.is_running:
            try:
                self.run_trading_cycle()
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                time.sleep(60)
        
        mt5.shutdown()
        logging.info("Donkey Trading Bot stopped")
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        self.notification_manager.notify(
            "Donkey Bot Stopped", 
            "🛑 Donkey Strategy Bot has been stopped."
        )

def create_donkey_bot_config():
    config = {
        # MT5 Connection - Now using environment variables
        'mt5_login': int(os.getenv('MT5_LOGIN', '12345678')),
        'mt5_password': os.getenv('MT5_PASSWORD', 'your_password'),
        'mt5_server': os.getenv('MT5_SERVER', 'your_broker_server'),
        
        # Keep everything else the same...
        'max_daily_loss_percent': 3.0,
        'symbols': [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURJPY', 'GBPJPY', 'EURGBP', 'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD'
        ],
        # ... rest of your config
    }
    return config

# Configuration and usage example


# Main execution
if __name__ == "__main__":
    # Analyze traditional strategies and show Donkey parameters
    analyzer = TradingStrategyAnalyzer()
    donkey_params = analyzer.calculate_donkey_parameters()
    
    print("=== DONKEY STRATEGY PARAMETERS ===")
    print(f"Average Entry Offset: {donkey_params['entry_offset']} pips")
    print(f"Take Profit Distance: {donkey_params['tp_distance']} (80% of avg SL)")
    print(f"Stop Loss Distance: {donkey_params['sl_distance']} (30% of normal - TIGHT!)")
    print(f"Original Average TP Ratio: {donkey_params['original_avg_tp']}")
    print(f"Original Average SL Ratio: {donkey_params['original_avg_sl']}")
    
    # Create and start the bot
    config = create_donkey_bot_config()
    donkey_bot = DonkeyTradingBot(config)
    
    try:
        donkey_bot.start()
    except KeyboardInterrupt:
        print("\nStopping Donkey Bot...")
        donkey_bot.stop()
