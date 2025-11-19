#!/usr/bin/env python3
"""XAUUSD Breakout Trading Bot - Clean, efficient, production-ready."""

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

# Constants
CHECK_CONNECTION_INTERVAL = 300  # 5 minutes
REGIME_CHECK_INTERVAL = 14400  # 4 hours


def setup_logging(config):
    """Setup logging with clean, minimal output."""
    log_level = config['operational']['log_level']
    log_file = config['operational']['log_file']

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level))
    root.handlers = []

    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, log_level))
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root.addHandler(fh)

    # Console handler (minimal)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    root.addHandler(ch)

    logger.info("═" * 80)
    logger.info("GOLDBOT - XAUUSD Breakout Trading System")
    logger.info("═" * 80)


def load_config(path='bot_config.yaml'):
    """Load configuration from YAML."""
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logging.error(f"Config error: {e}")
        sys.exit(1)


class TelegramNotifier:
    def __init__(self, config):
        self.enabled = config['telegram']['enabled']
        self.bot_token = config['telegram']['bot_token']
        self.chat_id = config['telegram']['chat_id']
        self.send_trades = config['telegram']['send_trades']
        self.send_errors = config['telegram']['send_errors']

        if self.enabled and config['telegram']['send_startup']:
            self._send("🤖 GOLDBOT started")

    def _send(self, message):
        """Send telegram message."""
        if not self.enabled:
            return

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            requests.post(url, json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    def send_trade(self, trade_info):
        """Send trade notification."""
        if not self.enabled or not self.send_trades:
            return

        emoji = "🟢" if trade_info['direction'] == 'BULLISH' else "🔴"
        msg = f"{emoji} <b>{trade_info['direction']}</b> | {trade_info['entry_model']}\n" \
              f"Entry: ${trade_info['entry_price']:.2f} | SL: ${trade_info['sl_price']:.2f}\n" \
              f"TP1: ${trade_info['tp1_price']:.2f} | TP2: ${trade_info['tp2_price']:.2f}\n" \
              f"Size: {trade_info['lot_size']:.2f} lots | Score: {trade_info.get('score', 0)}\n" \
              f"Ticket: {trade_info.get('ticket', 'N/A')}"
        self._send(msg)

    def send_degradation_alert(self, regime, metrics):
        """Send degradation alert."""
        if not self.enabled:
            return

        msg = f"⚠️ <b>DEGRADATION ALERT</b>\n\n" \
              f"Regime: {regime['volatility']} vol, {regime['trend']} trend\n" \
              f"Win rate: {metrics.get('win_rate', 0)*100:.1f}%\n" \
              f"Profit factor: {metrics.get('profit_factor', 0):.2f}\n" \
              f"Action: {metrics.get('recommended_action', 'Review strategy')}"
        self._send(msg)

    def send_error(self, error_msg):
        """Send error notification."""
        if self.enabled and self.send_errors:
            self._send(f"⚠️ Error: {error_msg}")


class MarketRegimeDetector:
    def __init__(self):
        self.last_regime = None

    def detect_regime(self, df_daily):
        """Detect market regime for degradation monitoring."""
        if len(df_daily) < 200:
            return {'volatility': 'UNKNOWN', 'trend': 'UNKNOWN', 'favorable_for_breakouts': True}

        atr_20 = self._calc_atr(df_daily, 20)
        atr_100 = self._calc_atr(df_daily, 100)

        current_atr = atr_20.iloc[-1]
        avg_atr = atr_100.iloc[-1]

        volatility = "HIGH" if current_atr > avg_atr * 1.5 else "LOW" if current_atr < avg_atr * 0.7 else "NORMAL"

        sma_50 = df_daily['close'].rolling(50).mean()
        sma_200 = df_daily['close'].rolling(200).mean()
        price = df_daily['close'].iloc[-1]

        if price > sma_50.iloc[-1] > sma_200.iloc[-1]:
            trend = "STRONG_BULL"
        elif price < sma_50.iloc[-1] < sma_200.iloc[-1]:
            trend = "STRONG_BEAR"
        elif abs(sma_50.iloc[-1] - sma_200.iloc[-1]) / price < 0.02:
            trend = "RANGING"
        else:
            trend = "TRANSITIONING"

        is_favorable = volatility in ["NORMAL", "HIGH"] and trend != "RANGING"

        regime = {
            'volatility': volatility,
            'trend': trend,
            'favorable_for_breakouts': is_favorable,
            'atr_ratio': current_atr / avg_atr if avg_atr > 0 else 1.0
        }

        if self.last_regime and regime != self.last_regime:
            if is_favorable != self.last_regime.get('favorable_for_breakouts'):
                logger.warning(f"⚠️ Regime change: {regime}")

        self.last_regime = regime
        return regime

    def _calc_atr(self, df, period):
        """Calculate ATR."""
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def calculate_performance_metrics(self, trades):
        """Calculate performance metrics from recent trades."""
        if len(trades) < 10:
            return {'win_rate': 0.5, 'profit_factor': 1.0, 'avg_r': 0.0, 'sharpe_ratio': 0.0, 'trades_count': len(trades)}

        closed = [t for t in trades if t.get('pnl') is not None]
        if len(closed) < 5:
            return {'win_rate': 0.5, 'profit_factor': 1.0, 'avg_r': 0.0, 'sharpe_ratio': 0.0, 'trades_count': len(closed)}

        pnls = [t['pnl'] for t in closed]
        r_multiples = [t.get('r_multiple', 0) for t in closed if t.get('r_multiple')]

        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else 999
        avg_r = np.mean(r_multiples) if r_multiples else 0
        sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if len(pnls) > 1 and np.std(pnls) > 0 else 0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_r': avg_r,
            'sharpe_ratio': sharpe,
            'trades_count': len(closed)
        }

    def check_degradation(self, recent_trades, baseline_metrics=None):
        """Check if strategy is degrading."""
        if baseline_metrics is None:
            baseline_metrics = {'win_rate': 0.40, 'profit_factor': 1.5, 'sharpe_ratio': 1.0}

        current = self.calculate_performance_metrics(recent_trades)

        if current['trades_count'] < 20:
            return None, current

        flags = []

        if current['win_rate'] < baseline_metrics['win_rate'] * 0.7:
            flags.append(f"Win rate: {current['win_rate']:.1%} vs {baseline_metrics['win_rate']:.1%}")

        if current['profit_factor'] < baseline_metrics['profit_factor'] * 0.7:
            flags.append(f"PF: {current['profit_factor']:.2f} vs {baseline_metrics['profit_factor']:.2f}")

        if current['sharpe_ratio'] < 0.5:
            flags.append(f"Sharpe too low: {current['sharpe_ratio']:.2f}")

        if current['avg_r'] < 0.3:
            flags.append(f"Avg R poor: {current['avg_r']:.2f}")

        if flags:
            if current['sharpe_ratio'] < 0.3 or current['profit_factor'] < 1.1:
                current['recommended_action'] = "STOP TRADING - Severe degradation"
                current['severity'] = "SEVERE"
            elif current['sharpe_ratio'] < 0.6:
                current['recommended_action'] = "REDUCE RISK 50%"
                current['severity'] = "MODERATE"
            else:
                current['recommended_action'] = "MONITOR CLOSELY"
                current['severity'] = "MILD"

            return flags, current

        return None, current


def connect_mt5(config):
    """Connect to MetaTrader5."""
    mt5_cfg = config['mt5']

    if not mt5.initialize(path=mt5_cfg['path'] if mt5_cfg['path'] else None, timeout=mt5_cfg['timeout']):
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    if not mt5.login(mt5_cfg['login'], mt5_cfg['password'], mt5_cfg['server']):
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    account = mt5.account_info()
    if not account:
        logger.error("Failed to get account info")
        mt5.shutdown()
        return False

    logger.info(f"Connected: {account.server} | Account: {account.login} | Balance: ${account.balance:.2f}")

    symbol = config['symbol']
    info = mt5.symbol_info(symbol)
    if not info:
        logger.error(f"Symbol {symbol} not found")
        mt5.shutdown()
        return False

    if not info.visible and not mt5.symbol_select(symbol, True):
        logger.error(f"Failed to enable {symbol}")
        mt5.shutdown()
        return False

    logger.info(f"Symbol {symbol} ready | Spread: {info.spread} pts")
    return True


def get_ohlc_data(symbol, timeframe, bars=500):
    """Fetch OHLC data from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df


def run_live_trading(config):
    """Run live trading with clean, minimal logging."""
    logger.info("Starting LIVE TRADING mode")

    telegram = TelegramNotifier(config)
    analyst = MarketAnalyst(config)
    trader = TradeManager(config)
    regime_detector = MarketRegimeDetector()

    if not connect_mt5(config):
        telegram.send_error("Failed to connect to MT5")
        return

    symbol = config['symbol']
    loop_sleep = config['operational']['loop_sleep_seconds']

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

    logger.info("Entering main loop...")

    try:
        while True:
            try:
                now = datetime.now(timezone.utc)

                # Connection check
                if time.time() - last_connection_check >= CHECK_CONNECTION_INTERVAL:
                    if not mt5.terminal_info():
                        logger.error("MT5 connection lost, reconnecting...")
                        telegram.send_error("MT5 connection lost")
                        if not connect_mt5(config):
                            logger.error("Reconnection failed, retrying in 60s...")
                            time.sleep(60)
                            continue
                    last_connection_check = time.time()

                account = mt5.account_info()
                if not account:
                    time.sleep(loop_sleep)
                    continue

                equity = account.equity

                # Fetch data
                df_m5 = get_ohlc_data(symbol, primary_tf, 500)
                df_m1 = get_ohlc_data(symbol, confirm_tf, 500)
                df_h1 = get_ohlc_data(symbol, bias_tf, 200)

                if df_m5 is None or df_m1 is None or df_h1 is None:
                    logger.warning("Data fetch failed, retrying...")
                    time.sleep(loop_sleep)
                    continue

                current_bar_time = df_m5.iloc[-1]['time']

                if last_bar_time is None:
                    last_bar_time = current_bar_time

                # Regime check (every 4 hours)
                if last_regime_check is None or (now - last_regime_check).total_seconds() >= REGIME_CHECK_INTERVAL:
                    df_daily = get_ohlc_data(symbol, tf_map['D1'], 300)
                    if df_daily is not None:
                        regime = regime_detector.detect_regime(df_daily)
                        logger.info(f"Regime: {regime['volatility']} vol, {regime['trend']} trend, "
                                    f"favorable={regime['favorable_for_breakouts']}")

                        if not regime['favorable_for_breakouts']:
                            logger.warning("⚠️ Unfavorable regime for breakouts")

                        # Check degradation
                        cursor = trader.db_conn.cursor()
                        cursor.execute('SELECT * FROM trade_history ORDER BY exit_time DESC LIMIT 30')
                        recent_trades = [{'pnl': row[8], 'r_multiple': row[9]} for row in cursor.fetchall()]

                        if len(recent_trades) >= 20:
                            degradation, metrics = regime_detector.check_degradation(recent_trades)
                            if degradation:
                                logger.error(f"⚠️ Degradation detected: {degradation}")
                                telegram.send_degradation_alert(regime, metrics)

                                if metrics.get('severity') == 'SEVERE':
                                    logger.critical("🛑 STOPPING due to severe degradation")
                                    telegram._send("🛑 BOT STOPPED - Severe degradation")
                                    break

                    last_regime_check = now

                # Session management
                current_session = 'LONDON' if analyst.is_trading_session(now) else 'NONE'
                if last_session != current_session and current_session != 'NONE':
                    trader.reset_session_count()
                last_session = current_session

                # Get current price for position management
                tick = mt5.symbol_info_tick(symbol)
                current_price = tick.bid if tick else None
                spread = tick.ask - tick.bid if tick else 999.0

                # New bar check
                if current_bar_time > last_bar_time:
                    last_bar_time = current_bar_time

                    if current_price:
                        # Analyze
                        analysis = analyst.analyze_market(
                            df_m5=df_m5,
                            df_m1=df_m1,
                            df_h1=df_h1,
                            current_time=now,
                            spread=spread,
                            equity=equity
                        )

                        # Execute if signal
                        if analysis['trade_allowed']:
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
                                    telegram.send_trade(trade_info)
                                else:
                                    logger.debug(f"Trade blocked: {result['reason']}")

                # Manage orders and positions
                trader.manage_pending_orders()

                if current_price and df_m5 is not None:
                    atr_current = analyst.calculate_atr(df_m5).iloc[-1]
                    trader.manage_positions(current_price, atr_current)

                time.sleep(loop_sleep)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt, shutting down...")
                break

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                telegram.send_error(f"Loop error: {str(e)}")
                time.sleep(loop_sleep)

    finally:
        logger.info("Shutting down...")
        telegram._send("🛑 GOLDBOT stopped")
        trader.close_database()
        mt5.shutdown()


def run_backtest(config):
    """Run scalping backtest using dedicated backtest module."""
    from backtest import ScalpingBacktest

    logger.info("Starting BACKTEST mode")

    if not connect_mt5(config):
        logger.error("MT5 connection failed")
        return

    start_date = datetime.strptime(config['backtest']['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(config['backtest']['end_date'], '%Y-%m-%d')
    initial_balance = config['backtest']['initial_balance']

    start_date = start_date.replace(tzinfo=timezone.utc)
    end_date = end_date.replace(tzinfo=timezone.utc)

    backtester = ScalpingBacktest(config)
    results = backtester.run(start_date, end_date, initial_balance)

    if results is None:
        logger.error("Backtest failed")
        mt5.shutdown()
        return

    logger.info("=" * 80)
    logger.info("SCALPING BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Initial Balance: ${results['initial_balance']:.2f}")
    logger.info(f"Final Equity: ${results['final_equity']:.2f}")
    logger.info(f"Total Return: ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)")
    logger.info(f"Total Trades: {results['total_trades']}")
    logger.info(f"Winning Trades: {results['winning_trades']} ({results['win_rate']:.1f}%)")
    logger.info(f"Losing Trades: {results['losing_trades']}")
    logger.info(f"Avg Win: ${results['avg_win']:.2f} ({results['avg_win_r']:.2f}R)")
    logger.info(f"Avg Loss: ${results['avg_loss']:.2f} ({results['avg_loss_r']:.2f}R)")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: ${results['max_drawdown']:.2f} ({results['max_drawdown_pct']:.2f}%)")
    logger.info(f"Max Consecutive Wins: {results['max_consecutive_wins']}")
    logger.info(f"Max Consecutive Losses: {results['max_consecutive_losses']}")
    logger.info(f"Partial Closes (TP1): {results['partial_closes']}")
    logger.info(f"Avg Trade Duration: {results['avg_trade_duration_minutes']:.1f} minutes")
    logger.info(f"Exit Reasons: {results['exit_reasons']}")
    logger.info("=" * 80)

    mt5.shutdown()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='GOLDBOT - XAUUSD Breakout Trading')
    parser.add_argument('--live', action='store_true', help='Run live trading')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--config', type=str, default='bot_config.yaml', help='Config file')

    args = parser.parse_args()

    if not args.live and not args.backtest:
        print("Error: Specify --live or --backtest")
        parser.print_help()
        sys.exit(1)

    if args.live and args.backtest:
        print("Error: Cannot run both modes simultaneously")
        sys.exit(1)

    config = load_config(args.config)
    setup_logging(config)

    if args.live:
        run_live_trading(config)
    else:
        run_backtest(config)


if __name__ == "__main__":
    main()
