"""
Scalping Backtest Engine for XAUUSD Trading Bot

Simulates realistic M5 scalping with:
- Dynamic risk scaling based on account size
- Partial close at TP1 (70%)
- Realistic OHLC-based fills
- 40-minute max hold time
- Spread and slippage modeling
"""

import MetaTrader5 as mt5
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from analyst import MarketAnalyst

logger = logging.getLogger(__name__)


class ScalpingBacktest:
    def __init__(self, config):
        self.config = config
        self.symbol = config['symbol']
        self.point_value = config['point_value']

        # Risk parameters
        self.dynamic_risk_tiers = config['risk'].get('dynamic_risk_tiers', [])
        self.risk_per_trade = config['risk']['risk_per_trade']

        # Position sizing
        self.min_lot = config['position']['min_lot']
        self.max_lot = config['position']['max_lot']
        self.lot_step = config['position']['lot_step']

        # Scalping parameters
        self.tp1_mult = config['take_profit']['tp1_multiplier']
        self.tp2_mult = config['take_profit']['tp2_multiplier']
        self.partial_close_pct = config['take_profit']['partial_close_pct']
        self.max_hold_bars = config['trade_management']['max_hold_bars']
        self.sl_min = config['stop_loss']['minimum']

        self.analyst = MarketAnalyst(config)

    def get_dynamic_risk(self, equity):
        """Get risk percentage based on account size tiers."""
        if not self.dynamic_risk_tiers:
            return self.risk_per_trade

        for tier in self.dynamic_risk_tiers:
            if equity <= tier['max_equity']:
                return tier['risk_pct']

        return self.risk_per_trade

    def calculate_position_size(self, equity, entry_price, sl_price):
        """Calculate lot size with dynamic risk."""
        risk_pct = self.get_dynamic_risk(equity)
        risk_dollars = equity * risk_pct
        sl_distance = abs(entry_price - sl_price)

        if sl_distance == 0:
            return 0.0, 0.0, 0.0

        lot = risk_dollars / (sl_distance * self.point_value)
        lot = round(lot / self.lot_step) * self.lot_step
        lot = max(self.min_lot, min(lot, self.max_lot))

        return lot, risk_pct, risk_dollars

    def check_bar_fill(self, bar, position, current_time):
        """
        Check if bar hits TP1, TP2, SL, or max hold time.
        Returns: (fill_type, exit_price, r_multiple)
        fill_type: 'TP1', 'TP2', 'SL', 'TIME', None
        """
        direction = position['direction']
        entry_price = position['entry_price']
        sl_price = position['sl_price']
        tp1_price = position['tp1_price']
        tp2_price = position['tp2_price']
        entry_time = position['entry_time']
        partial_closed = position.get('partial_closed', False)

        # Check max hold time
        bars_held = (current_time - entry_time).total_seconds() / 300
        if bars_held >= self.max_hold_bars:
            exit_price = bar['close']
            sl_dist = abs(entry_price - sl_price)
            profit = (exit_price - entry_price) if direction == 'BULLISH' else (entry_price - exit_price)
            r_mult = profit / sl_dist if sl_dist > 0 else 0
            return ('TIME', exit_price, r_mult)

        if direction == 'BULLISH':
            # Check SL first (worst case)
            if bar['low'] <= sl_price:
                return ('SL', sl_price, -1.0)

            # Check TP2 (if not partial closed, or remaining 30%)
            if bar['high'] >= tp2_price:
                return ('TP2', tp2_price, self.tp2_mult)

            # Check TP1 (70% partial close)
            if not partial_closed and bar['high'] >= tp1_price:
                return ('TP1', tp1_price, self.tp1_mult)

        else:  # BEARISH
            # Check SL first
            if bar['high'] >= sl_price:
                return ('SL', sl_price, -1.0)

            # Check TP2
            if bar['low'] <= tp2_price:
                return ('TP2', tp2_price, self.tp2_mult)

            # Check TP1
            if not partial_closed and bar['low'] <= tp1_price:
                return ('TP1', tp1_price, self.tp1_mult)

        return (None, None, None)

    def run(self, start_date, end_date, initial_balance):
        """Run backtest simulation with realistic scalping fills."""
        logger.info(f"Starting scalping backtest: {start_date.date()} to {end_date.date()}")
        logger.info(f"Initial balance: ${initial_balance:.2f}")

        # Fetch data
        days = (end_date - start_date).days
        bars_needed = min(days * 288, 50000)

        logger.info("Fetching historical data...")
        df_m5 = self.fetch_ohlc_data(self.symbol, mt5.TIMEFRAME_M5, bars_needed)
        df_h1 = self.fetch_ohlc_data(self.symbol, mt5.TIMEFRAME_H1, min(days * 24, 10000))

        if df_m5 is None or df_h1 is None:
            logger.error("Failed to fetch historical data")
            return None

        df_m5 = df_m5[(df_m5['time'] >= start_date) & (df_m5['time'] <= end_date)]
        df_h1 = df_h1[(df_h1['time'] >= start_date) & (df_h1['time'] <= end_date)]

        logger.info(f"Loaded {len(df_m5)} M5 bars, {len(df_h1)} H1 bars")

        # Backtest state
        equity = initial_balance
        balance = initial_balance
        peak_equity = initial_balance
        trades = []
        open_positions = []
        equity_curve = []

        # Spread simulation (realistic Exness XAUUSD spread)
        avg_spread = 0.15

        logger.info("Running scalping simulation...")

        lookback = 100
        for i in range(lookback, len(df_m5)):
            bar = df_m5.iloc[i]
            window_m5 = df_m5.iloc[:i+1]
            window_h1 = df_h1[df_h1['time'] <= bar['time']]

            current_time = bar['time']
            current_price = bar['close']

            # Update equity curve
            equity_curve.append({
                'time': current_time,
                'equity': equity,
                'balance': balance
            })

            # Check exits on open positions FIRST (using OHLC)
            for pos in list(open_positions):
                fill_type, exit_price, r_mult = self.check_bar_fill(bar, pos, current_time)

                if fill_type == 'TP1':
                    # Partial close: 70% at TP1
                    partial_volume = pos['lot_size'] * self.partial_close_pct
                    remaining_volume = pos['lot_size'] * (1 - self.partial_close_pct)

                    sl_dist = abs(pos['entry_price'] - pos['sl_price'])
                    profit_partial = (exit_price - pos['entry_price']) if pos['direction'] == 'BULLISH' else (pos['entry_price'] - exit_price)
                    pnl_partial = profit_partial * partial_volume * self.point_value

                    equity += pnl_partial
                    balance += pnl_partial
                    peak_equity = max(peak_equity, equity)

                    # Mark partial closed, let 30% run to TP2
                    pos['partial_closed'] = True
                    pos['lot_size'] = remaining_volume

                    logger.info(f"PARTIAL CLOSE: {pos['direction']} @ ${exit_price:.2f} | 70% | PnL: ${pnl_partial:.2f} | R: {r_mult:.2f}")

                elif fill_type in ['TP2', 'SL', 'TIME']:
                    # Full exit (or remaining 30% after partial)
                    sl_dist = abs(pos['entry_price'] - pos['sl_price'])
                    profit = (exit_price - pos['entry_price']) if pos['direction'] == 'BULLISH' else (pos['entry_price'] - exit_price)
                    pnl = profit * pos['lot_size'] * self.point_value

                    equity += pnl
                    balance += pnl
                    peak_equity = max(peak_equity, equity)

                    trade_record = {
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'direction': pos['direction'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'sl_price': pos['sl_price'],
                        'tp1_price': pos['tp1_price'],
                        'tp2_price': pos['tp2_price'],
                        'lot_size': pos['initial_lot_size'],
                        'pnl': pnl,
                        'r_multiple': r_mult,
                        'exit_reason': fill_type,
                        'partial_closed': pos.get('partial_closed', False),
                        'equity': equity,
                        'risk_pct': pos['risk_pct']
                    }

                    trades.append(trade_record)
                    open_positions.remove(pos)

                    logger.info(f"EXIT {fill_type}: {pos['direction']} @ ${exit_price:.2f} | PnL: ${pnl:.2f} | R: {r_mult:.2f} | Equity: ${equity:.2f}")

            # New signals (limit concurrent positions)
            if len(open_positions) < 2:
                try:
                    analysis = self.analyst.analyze_market(
                        df_m5=window_m5,
                        df_m1=window_m5,
                        df_h1=window_h1,
                        current_time=current_time,
                        spread=avg_spread,
                        equity=equity
                    )

                    if analysis['trade_allowed']:
                        direction = analysis['direction']
                        setup_data = analysis['setup_data']

                        # Realistic slippage (0.5 pip = $0.05)
                        slippage = 0.05 if direction == 'BULLISH' else -0.05
                        entry_price = current_price + slippage

                        # Calculate SL from sweep
                        if direction == 'BULLISH':
                            sweep_low = setup_data.get('sweep_low', current_price - 3.0)
                            sl_price = sweep_low - 0.10
                        else:
                            sweep_high = setup_data.get('sweep_high', current_price + 3.0)
                            sl_price = sweep_high + 0.10

                        sl_distance = abs(entry_price - sl_price)
                        sl_distance = max(sl_distance, self.sl_min)

                        # Recalculate SL with minimum
                        if direction == 'BULLISH':
                            sl_price = entry_price - sl_distance
                        else:
                            sl_price = entry_price + sl_distance

                        # Calculate TP levels with SCALPING parameters
                        if direction == 'BULLISH':
                            tp1_price = entry_price + (sl_distance * self.tp1_mult)
                            tp2_price = entry_price + (sl_distance * self.tp2_mult)
                        else:
                            tp1_price = entry_price - (sl_distance * self.tp1_mult)
                            tp2_price = entry_price - (sl_distance * self.tp2_mult)

                        # Dynamic position sizing
                        lot_size, risk_pct, risk_dollars = self.calculate_position_size(equity, entry_price, sl_price)

                        if lot_size >= self.min_lot:
                            position = {
                                'direction': direction,
                                'entry_price': entry_price,
                                'entry_time': current_time,
                                'sl_price': sl_price,
                                'tp1_price': tp1_price,
                                'tp2_price': tp2_price,
                                'lot_size': lot_size,
                                'initial_lot_size': lot_size,
                                'partial_closed': False,
                                'risk_pct': risk_pct,
                                'sl_distance': sl_distance
                            }

                            open_positions.append(position)

                            logger.info(f"ENTRY: {direction} @ ${entry_price:.2f} | SL: ${sl_price:.2f} | TP1: ${tp1_price:.2f} ({self.tp1_mult}R) | TP2: ${tp2_price:.2f} ({self.tp2_mult}R) | Size: {lot_size:.2f} lots | Risk: {risk_pct*100:.1f}% (${risk_dollars:.2f})")

                except Exception as e:
                    logger.error(f"Analysis error at {current_time}: {e}")

        # Close any remaining positions at end
        if open_positions:
            logger.warning(f"Closing {len(open_positions)} positions at end of backtest")
            final_bar = df_m5.iloc[-1]
            for pos in open_positions:
                exit_price = final_bar['close']
                profit = (exit_price - pos['entry_price']) if pos['direction'] == 'BULLISH' else (pos['entry_price'] - exit_price)
                pnl = profit * pos['lot_size'] * self.point_value
                equity += pnl

                sl_dist = abs(pos['entry_price'] - pos['sl_price'])
                r_mult = profit / sl_dist if sl_dist > 0 else 0

                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': final_bar['time'],
                    'direction': pos['direction'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'sl_price': pos['sl_price'],
                    'tp1_price': pos['tp1_price'],
                    'tp2_price': pos['tp2_price'],
                    'lot_size': pos['initial_lot_size'],
                    'pnl': pnl,
                    'r_multiple': r_mult,
                    'exit_reason': 'END',
                    'partial_closed': pos.get('partial_closed', False),
                    'equity': equity,
                    'risk_pct': pos['risk_pct']
                })

        # Generate results
        results = self.generate_report(trades, equity_curve, initial_balance, equity)

        return results

    def fetch_ohlc_data(self, symbol, timeframe, num_bars):
        """Fetch OHLC data from MT5."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df

    def generate_report(self, trades, equity_curve, initial_balance, final_equity):
        """Generate comprehensive backtest statistics."""
        if not trades:
            logger.warning("No trades executed in backtest")
            return {
                'total_trades': 0,
                'initial_balance': initial_balance,
                'final_equity': final_equity,
                'total_return': 0.0,
                'total_return_pct': 0.0
            }

        df_trades = pd.DataFrame(trades)

        # Basic metrics
        total_trades = len(df_trades)
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0

        total_pnl = df_trades['pnl'].sum()
        total_return_pct = ((final_equity - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0

        avg_win = winning_trades['pnl'].mean() if num_wins > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if num_losses > 0 else 0

        avg_win_r = winning_trades['r_multiple'].mean() if num_wins > 0 else 0
        avg_loss_r = losing_trades['r_multiple'].mean() if num_losses > 0 else 0

        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if num_wins > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if num_losses > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        # Drawdown
        df_equity = pd.DataFrame(equity_curve)
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = df_equity['equity'] - df_equity['peak']
        df_equity['drawdown_pct'] = (df_equity['drawdown'] / df_equity['peak'] * 100)

        max_drawdown = df_equity['drawdown'].min()
        max_drawdown_pct = df_equity['drawdown_pct'].min()

        # Exit reason breakdown
        exit_reasons = df_trades['exit_reason'].value_counts().to_dict()

        # Partial close stats
        partial_closed_count = df_trades[df_trades['partial_closed'] == True].shape[0]

        # Trade duration
        df_trades['duration_minutes'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 60
        avg_duration = df_trades['duration_minutes'].mean()

        # Consecutive wins/losses
        df_trades['win'] = df_trades['pnl'] > 0
        df_trades['streak'] = (df_trades['win'] != df_trades['win'].shift()).cumsum()
        streaks = df_trades.groupby('streak')['win'].agg(['first', 'count'])

        max_consecutive_wins = streaks[streaks['first'] == True]['count'].max() if len(streaks[streaks['first'] == True]) > 0 else 0
        max_consecutive_losses = streaks[streaks['first'] == False]['count'].max() if len(streaks[streaks['first'] == False]) > 0 else 0

        results = {
            'initial_balance': initial_balance,
            'final_equity': final_equity,
            'total_return': total_pnl,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_r': avg_win_r,
            'avg_loss_r': avg_loss_r,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'exit_reasons': exit_reasons,
            'partial_closes': partial_closed_count,
            'avg_trade_duration_minutes': avg_duration,
            'trades': df_trades,
            'equity_curve': df_equity
        }

        return results
