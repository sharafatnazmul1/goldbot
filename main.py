#!/usr/bin/env python3
"""
main.py - Entry Point for XAUUSD Breakout Trading Bot

Handles:
- Command-line argument parsing (--live, --backtest)
- Configuration loading
- MetaTrader5 connection
- Telegram messaging
- Main trading loop (24/7)
- Logging setup
- Backtest mode execution

Usage:
    python main.py --live      # Run live trading
    python main.py --backtest  # Run backtesting
"""

import sys
import argparse
import yaml
import logging
from datetime import datetime, timedelta
import time
import MetaTrader5 as mt5
import pandas as pd
import requests
from pathlib import Path

# Import our modules
from analyst import MarketAnalyst
from trader_and_manager import TradeManager


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config):
    """
    Setup logging with both file and console handlers.

    Args:
        config: Configuration dictionary
    """
    log_level = config['operational']['log_level']
    log_file = config['operational']['log_file']

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers
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
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with configuration
    """
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
        """
        Initialize Telegram notifier.

        Args:
            config: Configuration dictionary
        """
        self.enabled = config['telegram']['enabled']
        self.bot_token = config['telegram']['bot_token']
        self.chat_id = config['telegram']['chat_id']
        self.send_startup = config['telegram']['send_startup']
        self.send_trades = config['telegram']['send_trades']
        self.send_errors = config['telegram']['send_errors']

        if self.enabled and self.send_startup:
            self.send_message("🤖 GOLDBOT Started\n\nSystem initialized and ready to trade.")

    def send_message(self, message):
        """
        Send message via Telegram.

        Args:
            message: Message text
        """
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
        """
        Send trade notification.

        Args:
            trade_info: Dictionary with trade details
        """
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

    def send_error_notification(self, error_msg):
        """
        Send error notification.

        Args:
            error_msg: Error message
        """
        if not self.enabled or not self.send_errors:
            return

        message = f"⚠️ <b>Error</b>\n\n{error_msg}"
        self.send_message(message)


# ============================================================================
# MT5 CONNECTION
# ============================================================================

def connect_mt5(config):
    """
    Initialize and connect to MetaTrader5.

    Args:
        config: Configuration dictionary

    Returns:
        Boolean indicating success
    """
    mt5_config = config['mt5']

    # Initialize MT5
    if not mt5.initialize(
        path=mt5_config['path'] if mt5_config['path'] else None,
        timeout=mt5_config['timeout']
    ):
        logging.error(f"MT5 initialization failed: {mt5.last_error()}")
        return False

    logging.info("MT5 initialized successfully")

    # Login to account
    authorized = mt5.login(
        login=mt5_config['login'],
        password=mt5_config['password'],
        server=mt5_config['server']
    )

    if not authorized:
        logging.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    # Get account info
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

    # Check symbol
    symbol = config['symbol']
    symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:
        logging.error(f"Symbol {symbol} not found")
        mt5.shutdown()
        return False

    # Enable symbol if not enabled
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to enable symbol {symbol}")
            mt5.shutdown()
            return False

    logging.info(f"Symbol {symbol} ready for trading")
    logging.info(f"Spread: {symbol_info.spread} points")

    return True


# ============================================================================
# DATA FETCHING
# ============================================================================

def get_ohlc_data(symbol, timeframe, bars=500):
    """
    Fetch OHLC data from MT5.

    Args:
        symbol: Trading symbol
        timeframe: MT5 timeframe constant
        bars: Number of bars to fetch

    Returns:
        DataFrame with OHLC data
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    if rates is None or len(rates) == 0:
        logging.error(f"Failed to fetch {timeframe} data")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Rename columns to lowercase
    df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume'
    }, inplace=True)

    return df


def get_current_spread(symbol):
    """
    Get current spread in dollars.

    Args:
        symbol: Trading symbol

    Returns:
        Float: Spread in dollars
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 999.0  # Return high value on error

    spread = tick.ask - tick.bid
    return spread


def get_current_price(symbol):
    """
    Get current bid price.

    Args:
        symbol: Trading symbol

    Returns:
        Float: Current bid price
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    return tick.bid


# ============================================================================
# MAIN TRADING LOOP
# ============================================================================

def run_live_trading(config):
    """
    Run live trading loop.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting LIVE TRADING mode")

    # Initialize components
    telegram = TelegramNotifier(config)
    analyst = MarketAnalyst(config)
    trader = TradeManager(config)

    # Connect to MT5
    if not connect_mt5(config):
        telegram.send_error_notification("Failed to connect to MT5")
        return

    symbol = config['symbol']
    loop_sleep = config['operational']['loop_sleep_seconds']
    check_connection_interval = config['safety']['check_connection_interval']

    # Timeframe mapping
    tf_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'H1': mt5.TIMEFRAME_H1
    }

    primary_tf = tf_map[config['timeframes']['primary']]
    confirm_tf = tf_map[config['timeframes']['confirm']]
    bias_tf = tf_map[config['timeframes']['bias']]

    last_connection_check = time.time()
    last_bar_time = None

    logger.info("Entering main trading loop...")

    try:
        while True:
            try:
                # Check MT5 connection periodically
                if time.time() - last_connection_check >= check_connection_interval:
                    if not mt5.terminal_info():
                        logger.error("MT5 connection lost, attempting reconnect...")
                        telegram.send_error_notification("MT5 connection lost, reconnecting...")

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

                # Fetch market data
                df_m5 = get_ohlc_data(symbol, primary_tf, 500)
                df_m1 = get_ohlc_data(symbol, confirm_tf, 500)
                df_h1 = get_ohlc_data(symbol, bias_tf, 200)

                if df_m5 is None or df_m1 is None or df_h1 is None:
                    logger.warning("Failed to fetch market data")
                    time.sleep(loop_sleep)
                    continue

                # Check for new bar
                current_bar_time = df_m5.iloc[-1]['time']

                if last_bar_time is None:
                    last_bar_time = current_bar_time
                    logger.info(f"Initial bar time set: {current_bar_time}")

                if current_bar_time > last_bar_time:
                    logger.info(f"New M5 bar detected: {current_bar_time}")
                    last_bar_time = current_bar_time

                    # Get current market conditions
                    spread = get_current_spread(symbol)
                    current_time = datetime.utcnow()

                    logger.info(f"Spread: ${spread:.3f} | Equity: ${equity:.2f}")

                    # Analyze market
                    analysis = analyst.analyze_market(
                        df_m5=df_m5,
                        df_m1=df_m1,
                        df_h1=df_h1,
                        current_time=current_time,
                        spread=spread,
                        equity=equity
                    )

                    # Execute trade if allowed
                    if analysis['trade_allowed']:
                        logger.info(f"Trade signal: {analysis['reason']}")

                        # Calculate entry levels
                        entry_levels = trader.calculate_entry_levels(analysis, df_m5)

                        if entry_levels:
                            # Execute trade
                            result = trader.execute_trade(
                                direction=analysis['direction'],
                                entry_levels=entry_levels,
                                equity=equity
                            )

                            if result['success']:
                                # Send notification
                                trade_info = result['position_info'].copy()
                                trade_info['score'] = analysis['score']
                                trade_info['risk_dollars'] = equity * config['risk']['risk_per_trade']
                                telegram.send_trade_notification(trade_info)
                            else:
                                logger.warning(f"Trade execution failed: {result['reason']}")
                        else:
                            logger.warning("Failed to calculate entry levels")
                    else:
                        logger.debug(f"No trade: {analysis['reason']}")

                # Manage existing positions
                current_price = get_current_price(symbol)
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
        mt5.shutdown()
        logger.info("MT5 connection closed")


# ============================================================================
# BACKTEST MODE
# ============================================================================

def run_backtest(config):
    """
    Run backtesting mode.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting BACKTEST mode")

    # Initialize components
    analyst = MarketAnalyst(config)

    # Connect to MT5 (for data access)
    if not connect_mt5(config):
        logger.error("Failed to connect to MT5 for backtest")
        return

    symbol = config['symbol']
    start_date = datetime.strptime(config['backtest']['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(config['backtest']['end_date'], '%Y-%m-%d')
    initial_balance = config['backtest']['initial_balance']

    logger.info(f"Backtest period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial balance: ${initial_balance:.2f}")

    # Fetch historical data
    logger.info("Fetching historical data...")

    tf_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'H1': mt5.TIMEFRAME_H1
    }

    primary_tf = tf_map[config['timeframes']['primary']]

    # Calculate number of bars needed
    days = (end_date - start_date).days
    bars_needed = days * 288  # M5 bars per day

    df_m5 = get_ohlc_data(symbol, primary_tf, bars_needed)

    if df_m5 is None:
        logger.error("Failed to fetch backtest data")
        mt5.shutdown()
        return

    # Filter to date range
    df_m5 = df_m5[(df_m5['time'] >= start_date) & (df_m5['time'] <= end_date)]

    logger.info(f"Loaded {len(df_m5)} M5 bars")

    # Backtest simulation
    balance = initial_balance
    equity = initial_balance
    trades = []

    logger.info("Running backtest simulation...")
    logger.warning("NOTE: Full backtest simulation with trade execution is complex.")
    logger.warning("This is a simplified skeleton. Full implementation requires:")
    logger.warning("  - Bar-by-bar replay")
    logger.warning("  - Position tracking")
    logger.warning("  - Order fills simulation")
    logger.warning("  - P&L calculation")
    logger.warning("  - Performance metrics")

    # Placeholder: Count signals
    signal_count_bull = 0
    signal_count_bear = 0

    for i in range(100, len(df_m5)):
        # Get data window
        window_m5 = df_m5.iloc[:i+1]
        current_time = window_m5.iloc[-1]['time']

        # Simplified analysis (no M1/H1 for demo)
        df_h1_dummy = window_m5.copy()  # Placeholder

        # Mock spread
        spread = 0.10

        try:
            analysis = analyst.analyze_market(
                df_m5=window_m5,
                df_m1=window_m5,  # Mock with M5
                df_h1=df_h1_dummy,
                current_time=current_time,
                spread=spread,
                equity=equity
            )

            if analysis['trade_allowed']:
                if analysis['direction'] == 'BULLISH':
                    signal_count_bull += 1
                else:
                    signal_count_bear += 1

        except Exception as e:
            logger.debug(f"Analysis error at bar {i}: {e}")
            continue

    logger.info(f"Backtest complete")
    logger.info(f"Bullish signals detected: {signal_count_bull}")
    logger.info(f"Bearish signals detected: {signal_count_bear}")
    logger.warning("Full backtest execution engine not implemented in this version")
    logger.info("To implement full backtest, extend this function with:")
    logger.info("  1. Trade execution simulation")
    logger.info("  2. Position management")
    logger.info("  3. P&L tracking")
    logger.info("  4. Performance report generation")

    mt5.shutdown()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='GOLDBOT - XAUUSD Breakout Trading System')
    parser.add_argument('--live', action='store_true', help='Run live trading')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--config', type=str, default='bot_config.yaml', help='Config file path')

    args = parser.parse_args()

    # Check mode
    if not args.live and not args.backtest:
        print("Error: Must specify either --live or --backtest mode")
        parser.print_help()
        sys.exit(1)

    if args.live and args.backtest:
        print("Error: Cannot run both --live and --backtest simultaneously")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)

    # Run appropriate mode
    if args.live:
        run_live_trading(config)
    elif args.backtest:
        run_backtest(config)


if __name__ == "__main__":
    main()
