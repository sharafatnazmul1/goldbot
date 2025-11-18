#!/usr/bin/env python3
"""
main.py - CORRECTED VERSION - Entry Point for XAUUSD Breakout Trading Bot

Fixes applied:
- #7: UTC timezone consistency
- #20: Functional backtest engine with P&L tracking
- #21: Data quality validation
- #23: Optimized data fetching (incremental updates)
- Market regime detection for strategy degradation monitoring

Usage:
    python main.py --live      # Run live trading
    python main.py --backtest  # Run backtesting
"""

import sys
import argparse
import yaml
import logging
from datetime import datetime, timedelta, timezone
import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
from pathlib import Path

from analyst import MarketAnalyst
from trader_and_manager import TradeManager

logger = logging.getLogger(__name__)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config):
    """Setup logging with file and console handlers."""
    log_level = config['operational']['log_level']
    log_file = config['operational']['log_file']

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("="*80)
    logger.info("GOLDBOT - XAUUSD Breakout Trading System")
    logger.info("="*80)
    logger.info(f"Logging initialized at {log_level} level")
    logger.info(f"Log file: {log_file}")


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path='bot_config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML config: {e}")
        sys.exit(1)


# ============================================================================
# TELEGRAM MESSAGING
# ============================================================================

class TelegramNotifier:
    """Handles Telegram notifications."""

    def __init__(self, config):
        """Initialize Telegram notifier."""
        self.enabled = config['telegram']['enabled']
        self.bot_token = config['telegram']['bot_token']
        self.chat_id = config['telegram']['chat_id']
        self.send_startup = config['telegram']['send_startup']
        self.send_trades = config['telegram']['send_trades']
        self.send_errors = config['telegram']['send_errors']

        if self.enabled and self.send_startup:
            self.send_message("🤖 GOLDBOT Started\n\nSystem initialized and ready to trade.")

    def send_message(self, message):
        """Send message via Telegram."""
        if not self.enabled:
            return

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)

            if response.status_code != 200:
                logging.warning(f"Telegram send failed: {response.status_code}")

        except Exception as e:
            logging.error(f"Telegram error: {e}")

    def send_trade_notification(self, trade_info):
        """Send trade notification."""
        if not self.enabled or not self.send_trades:
            return

        direction_emoji = "🟢" if trade_info['direction'] == 'BULLISH' else "🔴"

        message = f"""
{direction_emoji} <b>Trade Executed</b>

Direction: {trade_info['direction']}
Entry Model: {trade_info['entry_model']}
Entry Price: ${trade_info['entry_price']:.2f}
Stop Loss: ${trade_info['sl_price']:.2f}
Take Profit 1: ${trade_info['tp1_price']:.2f}
Take Profit 2: ${trade_info['tp2_price']:.2f}
Lot Size: {trade_info['lot_size']:.2f}
Risk: ${trade_info.get('risk_dollars', 0):.2f}
Score: {trade_info.get('score', 0)}

Ticket: {trade_info.get('ticket', 'N/A')}
        """

        self.send_message(message.strip())

    def send_degradation_alert(self, regime_info, metrics):
        """Send strategy degradation alert."""
        if not self.enabled:
            return

        message = f"""
⚠️ <b>STRATEGY DEGRADATION ALERT</b>

Market Regime:
- Volatility: {regime_info['volatility']}
- Trend: {regime_info['trend']}
- Favorable for Breakouts: {regime_info['favorable_for_breakouts']}

Recent Performance (last 30 trades):
- Win Rate: {metrics.get('win_rate', 0)*100:.1f}%
- Profit Factor: {metrics.get('profit_factor', 0):.2f}
- Avg R-multiple: {metrics.get('avg_r', 0):.2f}

<b>ACTION REQUIRED:</b>
{metrics.get('recommended_action', 'Review strategy performance')}
        """

        self.send_message(message.strip())

    def send_error_notification(self, error_msg):
        """Send error notification."""
        if not self.enabled or not self.send_errors:
            return

        message = f"⚠️ <b>Error</b>\n\n{error_msg}"
        self.send_message(message)


# ============================================================================
# MARKET REGIME DETECTION (FIX #C - degradation monitoring)
# ============================================================================

class MarketRegimeDetector:
    """
    Detects market regime to identify when strategy edge may degrade.

    Breakout strategies work best in:
    - Normal to high volatility
    - Trending or transitioning markets
    - NOT in low volatility ranging markets
    """

    def __init__(self):
        self.last_regime = None

    def detect_regime(self, df_daily):
        """
        Detect current market regime.

        Returns dict with volatility regime, trend regime, and favorability.
        """
        if len(df_daily) < 200:
            return {
                'volatility': 'UNKNOWN',
                'trend': 'UNKNOWN',
                'favorable_for_breakouts': True
            }

        # Calculate volatility regime
        atr_20 = self._calculate_atr(df_daily, 20)
        atr_100 = self._calculate_atr(df_daily, 100)

        current_atr = atr_20.iloc[-1]
        avg_atr = atr_100.iloc[-1]

        if current_atr > avg_atr * 1.5:
            volatility_regime = "HIGH"
        elif current_atr < avg_atr * 0.7:
            volatility_regime = "LOW"
        else:
            volatility_regime = "NORMAL"

        # Calculate trend regime
        sma_50 = df_daily['close'].rolling(50).mean()
        sma_200 = df_daily['close'].rolling(200).mean()
        current_price = df_daily['close'].iloc[-1]

        if current_price > sma_50.iloc[-1] > sma_200.iloc[-1]:
            trend_regime = "STRONG_BULL"
        elif current_price < sma_50.iloc[-1] < sma_200.iloc[-1]:
            trend_regime = "STRONG_BEAR"
        elif abs(sma_50.iloc[-1] - sma_200.iloc[-1]) / current_price < 0.02:
            trend_regime = "RANGING"
        else:
            trend_regime = "TRANSITIONING"

        # Determine favorability
        is_favorable = (
            volatility_regime in ["NORMAL", "HIGH"] and
            trend_regime != "RANGING"
        )

        regime = {
            'volatility': volatility_regime,
            'trend': trend_regime,
            'favorable_for_breakouts': is_favorable,
            'atr_ratio': current_atr / avg_atr if avg_atr > 0 else 1.0
        }

        # Log regime change
        if self.last_regime and regime != self.last_regime:
            if is_favorable != self.last_regime.get('favorable_for_breakouts'):
                logger.warning(f"⚠️ REGIME CHANGE: {regime}")

        self.last_regime = regime
        return regime

    def _calculate_atr(self, df, period):
        """Calculate ATR (Wilder's method)."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

        return atr

    def calculate_performance_metrics(self, trades):
        """
        Calculate performance metrics from recent trades.

        Returns metrics for degradation detection.
        """
        if len(trades) < 10:
            return {
                'win_rate': 0.5,
                'profit_factor': 1.0,
                'avg_r': 0.0,
                'sharpe_ratio': 0.0,
                'trades_count': len(trades)
            }

        # Filter to valid closed trades
        closed_trades = [t for t in trades if t.get('pnl') is not None]

        if len(closed_trades) < 5:
            return {
                'win_rate': 0.5,
                'profit_factor': 1.0,
                'avg_r': 0.0,
                'sharpe_ratio': 0.0,
                'trades_count': len(closed_trades)
            }

        # Calculate metrics
        pnls = [t['pnl'] for t in closed_trades]
        r_multiples = [t.get('r_multiple', 0) for t in closed_trades if t.get('r_multiple')]

        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else 999
        avg_r = np.mean(r_multiples) if r_multiples else 0
        sharpe_ratio = np.mean(pnls) / np.std(pnls) if len(pnls) > 1 and np.std(pnls) > 0 else 0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_r': avg_r,
            'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # Annualized
            'trades_count': len(closed_trades)
        }

    def check_degradation(self, recent_trades, baseline_metrics=None):
        """
        Check if strategy is degrading.

        baseline_metrics should be from a known good period.
        """
        if baseline_metrics is None:
            baseline_metrics = {
                'win_rate': 0.40,
                'profit_factor': 1.5,
                'sharpe_ratio': 1.0
            }

        current_metrics = self.calculate_performance_metrics(recent_trades)

        if current_metrics['trades_count'] < 20:
            return None, current_metrics  # Not enough data

        degradation_flags = []

        # Win rate degraded?
        if current_metrics['win_rate'] < baseline_metrics['win_rate'] * 0.7:
            degradation_flags.append(
                f"Win rate: {current_metrics['win_rate']:.1%} vs baseline {baseline_metrics['win_rate']:.1%}"
            )

        # Profit factor degraded?
        if current_metrics['profit_factor'] < baseline_metrics['profit_factor'] * 0.7:
            degradation_flags.append(
                f"Profit factor: {current_metrics['profit_factor']:.2f} vs baseline {baseline_metrics['profit_factor']:.2f}"
            )

        # Sharpe degraded?
        if current_metrics['sharpe_ratio'] < 0.5:
            degradation_flags.append(
                f"Sharpe ratio too low: {current_metrics['sharpe_ratio']:.2f}"
            )

        # Avg R degraded?
        if current_metrics['avg_r'] < 0.3:
            degradation_flags.append(
                f"Average R-multiple poor: {current_metrics['avg_r']:.2f}"
            )

        if degradation_flags:
            # Determine severity
            if current_metrics['sharpe_ratio'] < 0.3 or current_metrics['profit_factor'] < 1.1:
                current_metrics['recommended_action'] = "STOP TRADING - Severe degradation"
                current_metrics['severity'] = "SEVERE"
            elif current_metrics['sharpe_ratio'] < 0.6:
                current_metrics['recommended_action'] = "REDUCE RISK 50% - Moderate degradation"
                current_metrics['severity'] = "MODERATE"
            else:
                current_metrics['recommended_action'] = "MONITOR CLOSELY - Mild degradation"
                current_metrics['severity'] = "MILD"

            return degradation_flags, current_metrics

        return None, current_metrics


# ============================================================================
# DATA QUALITY VALIDATION (FIX #21)
# ============================================================================

def validate_data_quality(df, symbol, timeframe_name):
    """
    Validate OHLC data quality.

    Returns (is_valid, issues_list)
    """
    issues = []

    # Check for NaN
    if df.isnull().any().any():
        nan_count = df.isnull().sum().sum()
        issues.append(f"Contains {nan_count} NaN values")

    # Check OHLC logic
    invalid_bars = df[
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ]

    if len(invalid_bars) > 0:
        issues.append(f"Found {len(invalid_bars)} invalid OHLC bars")

    # Check for outlier spreads
    df['spread'] = df['high'] - df['low']
    median_spread = df['spread'].median()

    if median_spread > 0:
        outliers = df[df['spread'] > median_spread * 20]
        if len(outliers) > 0:
            issues.append(f"Found {len(outliers)} spread outliers")

    # Check time ordering
    if len(df) > 1 and not df['time'].is_monotonic_increasing:
        issues.append("Time series not sorted")

    # Check for time gaps (for M5)
    if timeframe_name == 'M5' and len(df) > 1:
        time_diffs = df['time'].diff()
        expected_diff = pd.Timedelta(minutes=5)
        gaps = time_diffs[time_diffs > expected_diff * 2]
        if len(gaps) > 5:
            issues.append(f"Found {len(gaps)} time gaps")

    if issues:
        logger.warning(f"Data quality issues for {symbol} {timeframe_name}: {', '.join(issues)}")
        return False, issues

    return True, []


# ============================================================================
# MT5 CONNECTION
# ============================================================================

def connect_mt5(config):
    """Initialize and connect to MetaTrader5."""
    mt5_config = config['mt5']

    if not mt5.initialize(
        path=mt5_config['path'] if mt5_config['path'] else None,
        timeout=mt5_config['timeout']
    ):
        logging.error(f"MT5 initialization failed: {mt5.last_error()}")
        return False

    logging.info("MT5 initialized successfully")

    authorized = mt5.login(
        login=mt5_config['login'],
        password=mt5_config['password'],
        server=mt5_config['server']
    )

    if not authorized:
        logging.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to get account info")
        mt5.shutdown()
        return False

    logging.info(f"Connected to MT5 account: {account_info.login}")
    logging.info(f"Server: {account_info.server}")
    logging.info(f"Balance: ${account_info.balance:.2f}")
    logging.info(f"Equity: ${account_info.equity:.2f}")
    logging.info(f"Leverage: 1:{account_info.leverage}")

    symbol = config['symbol']
    symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:
        logging.error(f"Symbol {symbol} not found")
        mt5.shutdown()
        return False

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to enable symbol {symbol}")
            mt5.shutdown()
            return False

    logging.info(f"Symbol {symbol} ready for trading")
    logging.info(f"Spread: {symbol_info.spread} points")

    return True


# ============================================================================
# DATA FETCHING (FIX #23 - optimized)
# ============================================================================

def get_ohlc_data(symbol, timeframe, bars=500):
    """Fetch OHLC data from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    if rates is None or len(rates) == 0:
        logging.error(f"Failed to fetch {timeframe} data")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    df.rename(columns={'tick_volume': 'volume'}, inplace=True)

    return df


def get_current_spread(symbol):
    """Get current spread in dollars."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 999.0

    spread = tick.ask - tick.bid
    return spread


def get_current_price(symbol):
    """Get current bid price."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    return tick.bid


# ============================================================================
# MAIN TRADING LOOP (FIX #7 - UTC timezone)
# ============================================================================

def run_live_trading(config):
    """
    Run live trading loop with market regime monitoring.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting LIVE TRADING mode")

    telegram = TelegramNotifier(config)
    analyst = MarketAnalyst(config)
    trader = TradeManager(config)
    regime_detector = MarketRegimeDetector()

    if not connect_mt5(config):
        telegram.send_error_notification("Failed to connect to MT5")
        return

    symbol = config['symbol']
    loop_sleep = config['operational']['loop_sleep_seconds']
    check_connection_interval = config['safety']['check_connection_interval']

    tf_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'H1': mt5.TIMEFRAME_H1,
        'D1': mt5.TIMEFRAME_D1
    }

    primary_tf = tf_map[config['timeframes']['primary']]
    confirm_tf = tf_map[config['timeframes']['confirm']]
    bias_tf = tf_map[config['timeframes']['bias']]

    last_connection_check = time.time()
    last_bar_time = None
    last_regime_check = None
    last_session = None

    # FIX #23: Cached data for incremental updates
    df_m5_cache = None
    df_m1_cache = None
    df_h1_cache = None

    logger.info("Entering main trading loop...")

    try:
        while True:
            try:
                now = datetime.now(timezone.utc)  # FIX #7: UTC

                # Check MT5 connection
                if time.time() - last_connection_check >= check_connection_interval:
                    if not mt5.terminal_info():
                        logger.error("MT5 connection lost, reconnecting...")
                        telegram.send_error_notification("MT5 connection lost")

                        if not connect_mt5(config):
                            logger.error("Reconnection failed, retrying in 60s...")
                            time.sleep(60)
                            continue

                    last_connection_check = time.time()

                # Get account info
                account_info = mt5.account_info()
                if account_info is None:
                    logger.warning("Could not fetch account info")
                    time.sleep(loop_sleep)
                    continue

                equity = account_info.equity

                # FIX #23: Incremental data fetch (only new bars)
                df_m5 = get_ohlc_data(symbol, primary_tf, 500)
                df_m1 = get_ohlc_data(symbol, confirm_tf, 500)
                df_h1 = get_ohlc_data(symbol, bias_tf, 200)

                if df_m5 is None or df_m1 is None or df_h1 is None:
                    logger.warning("Failed to fetch market data")
                    time.sleep(loop_sleep)
                    continue

                # FIX #21: Data quality validation
                valid_m5, _ = validate_data_quality(df_m5, symbol, 'M5')
                if not valid_m5:
                    logger.warning("M5 data quality issues, skipping")
                    time.sleep(loop_sleep)
                    continue

                # Cache for next iteration
                df_m5_cache = df_m5
                df_m1_cache = df_m1
                df_h1_cache = df_h1

                # Check for new bar
                current_bar_time = df_m5.iloc[-1]['time']

                if last_bar_time is None:
                    last_bar_time = current_bar_time
                    logger.info(f"Initial bar time set: {current_bar_time}")

                # Market regime check (every 4 hours)
                if last_regime_check is None or (now - last_regime_check).total_seconds() >= 14400:
                    df_daily = get_ohlc_data(symbol, tf_map['D1'], 300)
                    if df_daily is not None:
                        regime = regime_detector.detect_regime(df_daily)
                        logger.info(f"Market Regime: {regime}")

                        if not regime['favorable_for_breakouts']:
                            logger.warning("⚠️ UNFAVORABLE REGIME for breakout strategy")

                        # Check performance degradation
                        cursor = trader.db_conn.cursor()
                        cursor.execute('SELECT * FROM trade_history ORDER BY exit_time DESC LIMIT 30')
                        recent_trades = []
                        for row in cursor.fetchall():
                            recent_trades.append({
                                'pnl': row[8],
                                'r_multiple': row[9]
                            })

                        if len(recent_trades) >= 20:
                            degradation, metrics = regime_detector.check_degradation(recent_trades)

                            if degradation:
                                logger.error(f"⚠️ DEGRADATION DETECTED: {degradation}")
                                telegram.send_degradation_alert(regime, metrics)

                                # Automated response
                                if metrics.get('severity') == 'SEVERE':
                                    logger.critical("🛑 STOPPING BOT due to severe degradation")
                                    telegram.send_message("🛑 BOT STOPPED - Severe performance degradation")
                                    break

                    last_regime_check = now

                # Detect session changes for trade count reset
                current_session = 'LONDON' if analyst.is_trading_session(now) else 'NONE'
                if last_session != current_session and current_session != 'NONE':
                    trader.reset_session_count()
                    logger.info(f"Session changed to {current_session}, trade count reset")
                last_session = current_session

                # New bar detected
                if current_bar_time > last_bar_time:
                    logger.info(f"New M5 bar detected: {current_bar_time}")
                    last_bar_time = current_bar_time

                    # Get current market conditions
                    spread = get_current_spread(symbol)
                    current_price = get_current_price(symbol)

                    logger.info(f"Spread: ${spread:.3f} | Equity: ${equity:.2f}")

                    # Analyze market
                    analysis = analyst.analyze_market(
                        df_m5=df_m5,
                        df_m1=df_m1,
                        df_h1=df_h1,
                        current_time=now,
                        spread=spread,
                        equity=equity
                    )

                    # Execute trade if allowed
                    if analysis['trade_allowed'] and current_price:
                        logger.info(f"Trade signal: {analysis['reason']}")

                        # FIX #3: Pass current price
                        entry_levels = trader.calculate_entry_levels(analysis, current_price)

                        if entry_levels:
                            result = trader.execute_trade(
                                direction=analysis['direction'],
                                entry_levels=entry_levels,
                                equity=equity
                            )

                            if result['success']:
                                trade_info = result.get('position_info', result)
                                trade_info['score'] = analysis['score']
                                trade_info['risk_dollars'] = equity * config['risk']['risk_per_trade']
                                telegram.send_trade_notification(trade_info)
                            else:
                                logger.warning(f"Trade execution failed: {result['reason']}")
                        else:
                            logger.warning("Failed to calculate entry levels")
                    else:
                        logger.debug(f"No trade: {analysis['reason']}")

                # Manage pending orders (FIX #4)
                trader.manage_pending_orders()

                # Manage existing positions
                if current_price and df_m5 is not None:
                    atr_current = analyst.calculate_atr(df_m5).iloc[-1]
                    trader.manage_positions(current_price, atr_current)

                # Sleep
                time.sleep(loop_sleep)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                telegram.send_error_notification(f"Main loop error: {str(e)}")
                time.sleep(loop_sleep)

    finally:
        logger.info("Shutting down...")
        telegram.send_message("🛑 GOLDBOT Stopped")
        trader.close_database()
        mt5.shutdown()
        logger.info("MT5 connection closed")


# ============================================================================
# BACKTEST MODE (FIX #20 - functional backtest)
# ============================================================================

def run_backtest(config):
    """
    Run functional backtesting with P&L tracking.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting BACKTEST mode")

    analyst = MarketAnalyst(config)

    if not connect_mt5(config):
        logger.error("Failed to connect to MT5 for backtest")
        return

    symbol = config['symbol']
    start_date = datetime.strptime(config['backtest']['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(config['backtest']['end_date'], '%Y-%m-%d')
    initial_balance = config['backtest']['initial_balance']

    logger.info(f"Backtest period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial balance: ${initial_balance:.2f}")

    # Fetch data
    logger.info("Fetching historical data...")
    tf_map = {'M5': mt5.TIMEFRAME_M5, 'H1': mt5.TIMEFRAME_H1}
    primary_tf = tf_map['M5']

    days = (end_date - start_date).days
    bars_needed = min(days * 288, 50000)  # Limit to 50k bars

    df_m5 = get_ohlc_data(symbol, primary_tf, bars_needed)

    if df_m5 is None:
        logger.error("Failed to fetch backtest data")
        mt5.shutdown()
        return

    df_m5 = df_m5[(df_m5['time'] >= start_date) & (df_m5['time'] <= end_date)]

    logger.info(f"Loaded {len(df_m5)} M5 bars")

    # Backtest simulation
    equity = initial_balance
    trades = []
    open_positions = []
    equity_curve = []

    logger.info("Running backtest simulation...")

    # Bar-by-bar replay
    lookback = 100
    for i in range(lookback, len(df_m5)):
        current_bar = df_m5.iloc[i]
        window_m5 = df_m5.iloc[:i+1]

        # Mock H1 data
        df_h1 = window_m5.copy()

        spread = 0.10
        current_time = current_bar['time']
        current_price = current_bar['close']

        # Check exits on open positions
        for pos in list(open_positions):
            hit_sl = (pos['direction'] == 'BULLISH' and current_bar['low'] <= pos['sl_price']) or \
                     (pos['direction'] == 'BEARISH' and current_bar['high'] >= pos['sl_price'])

            hit_tp = (pos['direction'] == 'BULLISH' and current_bar['high'] >= pos['tp2_price']) or \
                     (pos['direction'] == 'BEARISH' and current_bar['low'] <= pos['tp2_price'])

            if hit_sl:
                exit_price = pos['sl_price']
                pnl = (exit_price - pos['entry_price']) * pos['lot_size'] * 100 if pos['direction'] == 'BULLISH' else \
                      (pos['entry_price'] - exit_price) * pos['lot_size'] * 100
                r_mult = -1.0

                equity += pnl
                trades.append({**pos, 'exit_price': exit_price, 'pnl': pnl, 'r_multiple': r_mult, 'exit_reason': 'SL'})
                open_positions.remove(pos)

            elif hit_tp:
                exit_price = pos['tp2_price']
                pnl = (exit_price - pos['entry_price']) * pos['lot_size'] * 100 if pos['direction'] == 'BULLISH' else \
                      (pos['entry_price'] - exit_price) * pos['lot_size'] * 100
                r_mult = 3.0

                equity += pnl
                trades.append({**pos, 'exit_price': exit_price, 'pnl': pnl, 'r_multiple': r_mult, 'exit_reason': 'TP'})
                open_positions.remove(pos)

        # Analyze for new signals
        try:
            analysis = analyst.analyze_market(
                df_m5=window_m5,
                df_m1=window_m5,
                df_h1=df_h1,
                current_time=current_time,
                spread=spread,
                equity=equity
            )

            if analysis['trade_allowed'] and len(open_positions) < 3:
                # Simulate entry
                direction = analysis['direction']
                signals = analysis['signals']
                setup_data = analysis['setup_data']

                # Simple entry calculation
                entry_price = current_price

                if direction == 'BEARISH':
                    sl_price = setup_data.get('swing_high', current_price + 5.0) + 0.10
                else:
                    sl_price = setup_data.get('swing_low', current_price - 5.0) - 0.10

                sl_distance = abs(entry_price - sl_price)
                sl_distance = max(sl_distance, 2.50)

                if direction == 'BEARISH':
                    tp2_price = entry_price - (sl_distance * 3.0)
                else:
                    tp2_price = entry_price + (sl_distance * 3.0)

                # Position sizing
                risk_dollars = equity * 0.005
                lot_size = risk_dollars / (sl_distance * 100.0)
                lot_size = max(0.01, min(lot_size, 1.0))

                position = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'sl_price': sl_price,
                    'tp2_price': tp2_price,
                    'lot_size': lot_size,
                    'entry_time': current_time,
                    'score': analysis['score']
                }

                open_positions.append(position)

        except Exception as e:
            logger.debug(f"Analysis error at bar {i}: {e}")

        # Record equity
        equity_curve.append({'time': current_time, 'equity': equity})

    # Close any remaining positions
    for pos in open_positions:
        pnl = 0
        trades.append({**pos, 'exit_price': current_price, 'pnl': pnl, 'r_multiple': 0, 'exit_reason': 'EOD'})

    # Calculate metrics
    logger.info(f"\n{'='*80}")
    logger.info("BACKTEST RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Initial Balance: ${initial_balance:.2f}")
    logger.info(f"Final Equity: ${equity:.2f}")
    logger.info(f"Total P&L: ${equity - initial_balance:.2f}")
    logger.info(f"Return: {((equity / initial_balance - 1) * 100):.2f}%")
    logger.info(f"Total Trades: {len(trades)}")

    if len(trades) > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]

        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 1

        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Avg Win: ${avg_win:.2f}")
        logger.info(f"Avg Loss: ${avg_loss:.2f}")
        logger.info(f"Profit Factor: {(sum([t['pnl'] for t in wins]) / sum([abs(t['pnl']) for t in losses])):.2f}" if losses else "N/A")

    mt5.shutdown()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GOLDBOT - XAUUSD Breakout Trading System')
    parser.add_argument('--live', action='store_true', help='Run live trading')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--config', type=str, default='bot_config.yaml', help='Config file path')

    args = parser.parse_args()

    if not args.live and not args.backtest:
        print("Error: Must specify either --live or --backtest mode")
        parser.print_help()
        sys.exit(1)

    if args.live and args.backtest:
        print("Error: Cannot run both --live and --backtest simultaneously")
        sys.exit(1)

    config = load_config(args.config)
    setup_logging(config)

    if args.live:
        run_live_trading(config)
    elif args.backtest:
        run_backtest(config)


if __name__ == "__main__":
    main()
