"""Trade execution, position management, and state persistence for XAUUSD bot."""

import MetaTrader5 as mt5
import logging
import sqlite3
import json
from datetime import datetime, timezone
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class OrderState(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    CLOSED = "closed"


class TradeManager:
    def __init__(self, config):
        self.config = config
        self.symbol = config['symbol']
        self.point_value = config['point_value']

        # Risk parameters
        self.risk_per_trade = config['risk']['risk_per_trade']
        self.dynamic_risk_tiers = config['risk'].get('dynamic_risk_tiers', [])
        self.max_position_risk = config['risk']['max_position_risk']
        self.max_concurrent = config['risk']['max_concurrent_trades']
        self.daily_loss_limit = config['risk']['daily_loss_limit']
        self.weekly_loss_limit = config['risk'].get('weekly_loss_limit', 0.08)
        self.max_drawdown = config['risk'].get('max_drawdown_limit', 0.15)
        self.max_consecutive_losses = config['risk'].get('consecutive_loss_limit', 5)

        # Position sizing
        self.min_lot = config['position']['min_lot']
        self.max_lot = config['position']['max_lot']
        self.lot_step = config['position']['lot_step']

        # SL/TP
        self.sl_buffer = config['stop_loss']['buffer']
        self.sl_min = config['stop_loss']['minimum']
        self.tp1_mult = config['take_profit']['tp1_multiplier']
        self.tp2_mult = config['take_profit']['tp2_multiplier']
        self.partial_pct = config['take_profit']['partial_close_pct']

        # Management
        self.breakeven_at_r = config['trade_management']['breakeven_at_r']
        self.trail_at_r = config['trade_management']['trail_at_r']
        self.trail_atr_mult = config['trade_management']['trail_atr_mult']
        self.max_hold_bars = config['trade_management']['max_hold_bars']
        self.min_trade_interval = config['trade_management'].get('min_time_between_trades', 300)
        self.max_trades_per_session = config['trade_management'].get('max_trades_per_session', 4)
        self.cooloff_after_loss = config['trade_management'].get('cooloff_after_loss', 900)

        # Entry models
        self.entry_models = config['entry_models']

        # State
        self.active_positions = {}
        self.pending_orders = {}
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self.weekly_start_equity = 0.0
        self.last_daily_reset = None
        self.last_weekly_reset = None
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.session_trade_count = 0
        self.last_loss_time = None

        self._init_database()
        self._load_state()

    # Database operations

    def _init_database(self):
        """Initialize SQLite database."""
        self.db_conn = sqlite3.connect('goldbot_state.db', check_same_thread=False)
        with self.db_conn:
            self.db_conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    ticket INTEGER PRIMARY KEY,
                    direction TEXT,
                    entry_price REAL,
                    sl_price REAL,
                    tp1_price REAL,
                    tp2_price REAL,
                    lot_size REAL,
                    entry_model TEXT,
                    entry_time TEXT,
                    sl_distance REAL,
                    state TEXT,
                    partial_closed INTEGER DEFAULT 0,
                    breakeven_moved INTEGER DEFAULT 0,
                    trailing_active INTEGER DEFAULT 0,
                    data_json TEXT
                )
            ''')
            self.db_conn.execute('''
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            ''')
            self.db_conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    lot_size REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    pnl REAL,
                    r_multiple REAL,
                    entry_model TEXT,
                    exit_reason TEXT,
                    score INTEGER
                )
            ''')

    def _load_state(self):
        """Load state from database."""
        cursor = self.db_conn.cursor()

        for row in cursor.execute('SELECT data_json FROM positions WHERE state = ?', (OrderState.FILLED.value,)):
            pos = json.loads(row[0])
            pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])
            self.active_positions[pos['ticket']] = pos

        for row in cursor.execute('SELECT data_json FROM positions WHERE state = ?', (OrderState.PENDING.value,)):
            order = json.loads(row[0])
            order['order_time'] = datetime.fromisoformat(order['order_time'])
            self.pending_orders[order['ticket']] = order

        for key, value in cursor.execute('SELECT key, value FROM system_state'):
            val = json.loads(value)
            if key == 'peak_equity':
                self.peak_equity = val
            elif key == 'consecutive_losses':
                self.consecutive_losses = val

        logger.info(f"Loaded {len(self.active_positions)} positions, {len(self.pending_orders)} pending")

    def _save_position(self, pos):
        """Save position to database."""
        pos_json = pos.copy()
        for key in ['entry_time', 'order_time']:
            if key in pos_json and isinstance(pos_json[key], datetime):
                pos_json[key] = pos_json[key].isoformat()

        with self.db_conn:
            self.db_conn.execute('''
                INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                pos['ticket'],
                pos.get('direction'),
                pos.get('entry_price'),
                pos.get('sl_price'),
                pos.get('tp1_price'),
                pos.get('tp2_price'),
                pos.get('lot_size'),
                pos.get('entry_model'),
                pos_json.get('entry_time') or pos_json.get('order_time'),
                pos.get('sl_distance'),
                pos.get('state', OrderState.FILLED.value),
                int(pos.get('partial_closed', False)),
                int(pos.get('breakeven_moved', False)),
                int(pos.get('trailing_active', False)),
                json.dumps(pos_json)
            ))

    def _remove_position(self, ticket):
        """Remove position from database."""
        with self.db_conn:
            self.db_conn.execute('DELETE FROM positions WHERE ticket = ?', (ticket,))

    def _save_system_state(self, key, value):
        """Save system state."""
        with self.db_conn:
            self.db_conn.execute('''
                INSERT OR REPLACE INTO system_state VALUES (?, ?, ?)
            ''', (key, json.dumps(value), datetime.now(timezone.utc).isoformat()))

    def _save_trade(self, trade):
        """Save completed trade to history."""
        with self.db_conn:
            self.db_conn.execute('''
                INSERT INTO trade_history
                (ticket, direction, entry_price, exit_price, lot_size, entry_time,
                 exit_time, pnl, r_multiple, entry_model, exit_reason, score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                trade.get('ticket'),
                trade.get('direction'),
                trade.get('entry_price'),
                trade.get('exit_price'),
                trade.get('lot_size'),
                trade.get('entry_time').isoformat() if isinstance(trade.get('entry_time'), datetime) else trade.get('entry_time'),
                datetime.now(timezone.utc).isoformat(),
                trade.get('pnl'),
                trade.get('r_multiple'),
                trade.get('entry_model'),
                trade.get('exit_reason'),
                trade.get('score')
            ))

    # Position sizing

    def get_dynamic_risk(self, equity):
        """Get risk percentage based on account size tiers."""
        if not self.dynamic_risk_tiers:
            return self.risk_per_trade

        for tier in self.dynamic_risk_tiers:
            if equity <= tier['max_equity']:
                return tier['risk_pct']

        return self.risk_per_trade

    def calculate_position_size(self, equity, entry_price, sl_price):
        """Calculate lot size based on risk."""
        risk_pct = self.get_dynamic_risk(equity)
        risk_dollars = equity * risk_pct
        sl_distance = abs(entry_price - sl_price)

        if sl_distance == 0:
            logger.error("SL distance is zero")
            return 0.0

        lot = risk_dollars / (sl_distance * self.point_value)
        lot = round(lot / self.lot_step) * self.lot_step
        lot = max(self.min_lot, min(lot, self.max_lot))

        logger.info(f"Position: {lot:.2f} lots, risk {risk_pct*100:.1f}% (${risk_dollars:.2f}), SL ${sl_distance:.2f}")
        return lot

    # Entry level calculation

    def calculate_entry_levels(self, analysis_result, current_price):
        """Calculate entry levels from analysis without look-ahead bias."""
        direction = analysis_result['direction']
        signals = analysis_result['signals']
        setup_data = analysis_result['setup_data']

        if not direction:
            return None

        entry_model = self._determine_entry_model(signals)
        if not entry_model:
            return None

        if entry_model == 'MOMENTUM_BREAK':
            return self._calc_momentum_entry(direction, current_price, setup_data)
        elif entry_model == 'SHALLOW_RETEST':
            return self._calc_shallow_entry(direction, current_price, setup_data)
        elif entry_model == 'FVG_SNIPER':
            return self._calc_fvg_entry(direction, current_price, setup_data)

        return None

    def _determine_entry_model(self, signals):
        """Determine entry model from current signals."""
        if signals.get('sweep') and signals.get('impulse'):
            if self.entry_models['momentum_break']['enabled']:
                return 'MOMENTUM_BREAK'
            if self.entry_models['shallow_retest']['enabled']:
                return 'SHALLOW_RETEST'

        if signals.get('fvg') and self.entry_models['fvg_sniper']['enabled']:
            return 'FVG_SNIPER'

        if signals.get('sweep') and signals.get('impulse') and self.entry_models['shallow_retest']['enabled']:
            return 'SHALLOW_RETEST'

        return None

    def _calc_momentum_entry(self, direction, current_price, setup_data):
        """Calculate momentum break entry (market order)."""
        entry_price = current_price

        if direction == 'BEARISH':
            sl_price = setup_data.get('swing_high', current_price + 5.0) + self.sl_buffer
        else:
            sl_price = setup_data.get('swing_low', current_price - 5.0) - self.sl_buffer

        sl_distance = abs(entry_price - sl_price)
        if sl_distance < self.sl_min:
            sl_distance = self.sl_min
            sl_price = entry_price + sl_distance if direction == 'BEARISH' else entry_price - sl_distance

        if direction == 'BEARISH':
            tp1_price = entry_price - (sl_distance * self.tp1_mult)
            tp2_price = entry_price - (sl_distance * self.tp2_mult)
        else:
            tp1_price = entry_price + (sl_distance * self.tp1_mult)
            tp2_price = entry_price + (sl_distance * self.tp2_mult)

        return {
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'entry_model': 'MOMENTUM_BREAK',
            'order_type': 'MARKET',
            'sl_distance': sl_distance
        }

    def _calc_shallow_entry(self, direction, current_price, setup_data):
        """Calculate shallow retest entry (limit order)."""
        shallow_ratio = self.config['retracement']['shallow_ratio']

        if direction == 'BEARISH':
            swing_high = setup_data.get('swing_high', current_price + 5.0)
            entry_price = current_price + (swing_high - current_price) * shallow_ratio
            sl_price = swing_high + self.sl_buffer
        else:
            swing_low = setup_data.get('swing_low', current_price - 5.0)
            entry_price = current_price - (current_price - swing_low) * shallow_ratio
            sl_price = swing_low - self.sl_buffer

        sl_distance = abs(entry_price - sl_price)
        if sl_distance < self.sl_min:
            sl_distance = self.sl_min
            sl_price = entry_price + sl_distance if direction == 'BEARISH' else entry_price - sl_distance

        if direction == 'BEARISH':
            tp1_price = entry_price - (sl_distance * self.tp1_mult)
            tp2_price = entry_price - (sl_distance * self.tp2_mult)
        else:
            tp1_price = entry_price + (sl_distance * self.tp1_mult)
            tp2_price = entry_price + (sl_distance * self.tp2_mult)

        return {
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'entry_model': 'SHALLOW_RETEST',
            'order_type': 'LIMIT',
            'cancel_after_bars': self.entry_models['shallow_retest']['cancel_after_bars'],
            'sl_distance': sl_distance
        }

    def _calc_fvg_entry(self, direction, current_price, setup_data):
        """Calculate FVG sniper entry (limit order)."""
        fvg_ratio = self.entry_models['fvg_sniper']['fvg_entry_ratio']

        if direction == 'BEARISH':
            swing_high = setup_data.get('swing_high', current_price + 3.0)
            entry_price = current_price + (swing_high - current_price) * fvg_ratio
            sl_price = swing_high + self.sl_buffer
        else:
            swing_low = setup_data.get('swing_low', current_price - 3.0)
            entry_price = current_price - (current_price - swing_low) * fvg_ratio
            sl_price = swing_low - self.sl_buffer

        sl_distance = abs(entry_price - sl_price)
        if sl_distance < self.sl_min:
            sl_distance = self.sl_min
            sl_price = entry_price + sl_distance if direction == 'BEARISH' else entry_price - sl_distance

        if direction == 'BEARISH':
            tp1_price = entry_price - (sl_distance * self.tp1_mult)
            tp2_price = entry_price - (sl_distance * self.tp2_mult)
        else:
            tp1_price = entry_price + (sl_distance * self.tp1_mult)
            tp2_price = entry_price + (sl_distance * self.tp2_mult)

        return {
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'entry_model': 'FVG_SNIPER',
            'order_type': 'LIMIT',
            'cancel_after_bars': self.entry_models['fvg_sniper']['cancel_after_bars'],
            'sl_distance': sl_distance
        }

    # Risk checks

    def _check_risk_limits(self, equity):
        """Check all risk limits before trading."""
        # Drawdown check
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown >= self.max_drawdown:
                return False, f"Max drawdown {drawdown*100:.1f}%"

        # Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"{self.consecutive_losses} consecutive losses"

        # Daily loss
        today = datetime.now(timezone.utc).date()
        if self.last_daily_reset != today:
            self.daily_start_equity = equity
            self.last_daily_reset = today

        if self.daily_start_equity > 0:
            daily_loss = (self.daily_start_equity - equity) / self.daily_start_equity
            if daily_loss >= self.daily_loss_limit:
                return False, f"Daily loss {daily_loss*100:.1f}%"

        # Weekly loss
        week = datetime.now(timezone.utc).isocalendar()[1]
        if self.last_weekly_reset is None or self.last_weekly_reset != week:
            self.weekly_start_equity = equity
            self.last_weekly_reset = week

        if self.weekly_start_equity > 0:
            weekly_loss = (self.weekly_start_equity - equity) / self.weekly_start_equity
            if weekly_loss >= self.weekly_loss_limit:
                return False, f"Weekly loss {weekly_loss*100:.1f}%"

        # Throttling
        now = datetime.now(timezone.utc)
        if self.last_trade_time:
            if (now - self.last_trade_time).total_seconds() < self.min_trade_interval:
                return False, "Trade throttled"

        if self.session_trade_count >= self.max_trades_per_session:
            return False, "Session limit reached"

        if self.last_loss_time:
            if (now - self.last_loss_time).total_seconds() < self.cooloff_after_loss:
                return False, "Cooloff period"

        # Concurrent positions
        if len(self.active_positions) >= self.max_concurrent:
            return False, "Max concurrent positions"

        # Correlation (max 2 per direction)
        bull_count = sum(1 for p in self.active_positions.values() if p.get('direction') == 'BULLISH')
        bear_count = sum(1 for p in self.active_positions.values() if p.get('direction') == 'BEARISH')
        if bull_count >= 2 or bear_count >= 2:
            return False, "Correlation limit"

        return True, "Pass"

    # Order execution

    def execute_trade(self, direction, entry_levels, equity):
        """Execute trade with all risk checks."""
        if not entry_levels:
            return {'success': False, 'reason': 'No entry levels'}

        passed, reason = self._check_risk_limits(equity)
        if not passed:
            return {'success': False, 'reason': reason}

        lot_size = self.calculate_position_size(equity, entry_levels['entry_price'], entry_levels['sl_price'])
        if lot_size < self.min_lot:
            return {'success': False, 'reason': 'Lot size too small'}

        if entry_levels['order_type'] == 'MARKET':
            return self._execute_market_order(direction, entry_levels, lot_size)
        else:
            return self._execute_limit_order(direction, entry_levels, lot_size)

    def _execute_market_order(self, direction, levels, lot):
        """Execute market order."""
        order_type = mt5.ORDER_TYPE_BUY if direction == 'BULLISH' else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.ask if direction == 'BULLISH' else tick.bid

        if not price:
            return {'success': False, 'reason': 'No price'}

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": levels['sl_price'],
            "tp": levels['tp2_price'],
            "deviation": 10,
            "magic": 234000,
            "comment": f"GOLDBOT_{levels['entry_model']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {'success': False, 'reason': result.comment if result else 'Order failed'}

        # Check slippage
        slippage = abs(result.price - levels['entry_price'])
        max_slip = self.entry_models['momentum_break'].get('max_slippage', 0.05)
        if slippage > max_slip:
            logger.warning(f"Excessive slippage {slippage:.2f}, closing")
            self._close_position_by_ticket(result.order)
            return {'success': False, 'reason': f'Slippage {slippage:.2f}'}

        pos = {
            'ticket': result.order,
            'direction': direction,
            'entry_price': result.price,
            'sl_price': levels['sl_price'],
            'tp1_price': levels['tp1_price'],
            'tp2_price': levels['tp2_price'],
            'lot_size': lot,
            'entry_model': levels['entry_model'],
            'entry_time': datetime.now(timezone.utc),
            'sl_distance': levels['sl_distance'],
            'state': OrderState.FILLED.value,
            'partial_closed': False,
            'breakeven_moved': False,
            'trailing_active': False
        }

        self.active_positions[result.order] = pos
        self._save_position(pos)

        self.last_trade_time = datetime.now(timezone.utc)
        self.session_trade_count += 1

        logger.info(f"Order filled: #{result.order} @ {result.price:.2f}")
        return {'success': True, 'ticket': result.order, 'position_info': pos}

    def _execute_limit_order(self, direction, levels, lot):
        """Execute limit order."""
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == 'BULLISH' else mt5.ORDER_TYPE_SELL_LIMIT

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": levels['entry_price'],
            "sl": levels['sl_price'],
            "tp": levels['tp2_price'],
            "deviation": 10,
            "magic": 234000,
            "comment": f"GOLDBOT_{levels['entry_model']}",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)

        if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {'success': False, 'reason': result.comment if result else 'Limit failed'}

        pending = {
            'ticket': result.order,
            'direction': direction,
            'entry_price': levels['entry_price'],
            'sl_price': levels['sl_price'],
            'tp1_price': levels['tp1_price'],
            'tp2_price': levels['tp2_price'],
            'lot_size': lot,
            'entry_model': levels['entry_model'],
            'order_time': datetime.now(timezone.utc),
            'cancel_after_bars': levels.get('cancel_after_bars', 5),
            'sl_distance': levels['sl_distance'],
            'state': OrderState.PENDING.value
        }

        self.pending_orders[result.order] = pending
        self._save_position(pending)

        logger.info(f"Limit order: #{result.order} @ {levels['entry_price']:.2f}")
        return {'success': True, 'ticket': result.order, 'pending': True}

    # Pending order management

    def manage_pending_orders(self):
        """Manage pending orders lifecycle."""
        for ticket, order_info in list(self.pending_orders.items()):
            position = self._get_position_by_ticket(ticket)

            if position:
                logger.info(f"Pending #{ticket} filled @ {position.price_open:.2f}")
                order_info['entry_price'] = position.price_open
                order_info['entry_time'] = datetime.now(timezone.utc)
                order_info['state'] = OrderState.FILLED.value
                del order_info['order_time']
                del order_info['cancel_after_bars']

                self.active_positions[ticket] = order_info
                del self.pending_orders[ticket]
                self._save_position(order_info)

                self.last_trade_time = datetime.now(timezone.utc)
                self.session_trade_count += 1
                continue

            mt5_order = self._get_pending_order(ticket)
            if not mt5_order:
                del self.pending_orders[ticket]
                self._remove_position(ticket)
                continue

            bars_waited = (datetime.now(timezone.utc) - order_info['order_time']).total_seconds() / 300
            if bars_waited >= order_info.get('cancel_after_bars', 5):
                logger.info(f"Cancelling stale order #{ticket}")
                self._cancel_pending_order(ticket)
                del self.pending_orders[ticket]
                self._remove_position(ticket)

    def _get_pending_order(self, ticket):
        """Get pending order from MT5."""
        orders = mt5.orders_get(symbol=self.symbol)
        return next((o for o in orders if o.ticket == ticket), None) if orders else None

    def _cancel_pending_order(self, ticket):
        """Cancel pending order."""
        result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
        return result and result.retcode == mt5.TRADE_RETCODE_DONE

    # Position management

    def manage_positions(self, current_price, atr_current):
        """Manage all active positions."""
        if not self.active_positions:
            return

        to_remove = []

        for ticket, pos in list(self.active_positions.items()):
            mt5_pos = self._get_position_by_ticket(ticket)

            if not mt5_pos:
                to_remove.append(ticket)
                trade = pos.copy()
                trade['exit_reason'] = 'TP_SL'
                self._save_trade(trade)
                continue

            entry = pos['entry_price']
            sl_dist = pos['sl_distance']

            profit = (current_price - entry) if pos['direction'] == 'BULLISH' else (entry - current_price)
            r = profit / sl_dist if sl_dist > 0 else 0

            # Partial close at TP1
            if not pos['partial_closed'] and r >= self.tp1_mult:
                self._partial_close(ticket, pos, mt5_pos)

            # Breakeven
            if not pos['breakeven_moved'] and r >= self.breakeven_at_r:
                self._move_to_breakeven(ticket, pos, mt5_pos, entry)

            # Trailing
            if not pos['trailing_active'] and r >= self.trail_at_r:
                pos['trailing_active'] = True
                logger.info(f"Trailing activated: #{ticket}")

            if pos['trailing_active']:
                self._update_trailing_stop(ticket, pos, mt5_pos, current_price, atr_current)

            # Time-based exit
            if pos.get('entry_time'):
                bars_held = (datetime.now(timezone.utc) - pos['entry_time']).total_seconds() / 300
                if bars_held >= self.max_hold_bars:
                    logger.warning(f"Max hold reached: #{ticket}")
                    self._close_position_by_ticket(ticket)
                    to_remove.append(ticket)

        for ticket in to_remove:
            if ticket in self.active_positions:
                del self.active_positions[ticket]
                self._remove_position(ticket)

    def _partial_close(self, ticket, pos, mt5_pos):
        """Close partial position at TP1 (70% for scalping)."""
        volume = mt5_pos.volume * self.partial_pct
        volume = round(volume / self.lot_step) * self.lot_step

        if volume < self.min_lot:
            return

        order_type = mt5.ORDER_TYPE_SELL if pos['direction'] == 'BULLISH' else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.bid if pos['direction'] == 'BULLISH' else tick.ask

        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "GOLDBOT_TP1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            pos['partial_closed'] = True
            self._save_position(pos)
            logger.info(f"Partial close: #{ticket}")

    def _move_to_breakeven(self, ticket, pos, mt5_pos, entry_price):
        """Move SL to breakeven."""
        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": entry_price,
            "tp": mt5_pos.tp,
        })

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            pos['breakeven_moved'] = True
            self._save_position(pos)
            logger.info(f"Breakeven: #{ticket}")

    def _update_trailing_stop(self, ticket, pos, mt5_pos, current_price, atr):
        """Update trailing stop."""
        trail_dist = atr * self.trail_atr_mult

        if pos['direction'] == 'BULLISH':
            new_sl = current_price - trail_dist
            if new_sl > mt5_pos.sl:
                self._modify_sl(ticket, mt5_pos, new_sl)
        else:
            new_sl = current_price + trail_dist
            if new_sl < mt5_pos.sl:
                self._modify_sl(ticket, mt5_pos, new_sl)

    def _modify_sl(self, ticket, mt5_pos, new_sl):
        """Modify stop loss."""
        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": mt5_pos.tp,
        })

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Trailing SL: #{ticket} -> {new_sl:.2f}")

    # Helpers

    def _get_position_by_ticket(self, ticket):
        """Get MT5 position by ticket."""
        positions = mt5.positions_get(symbol=self.symbol)
        return next((p for p in positions if p.ticket == ticket), None) if positions else None

    def _close_position_by_ticket(self, ticket):
        """Close position completely."""
        pos = self._get_position_by_ticket(ticket)
        if not pos:
            return False

        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "GOLDBOT_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })

        return result and result.retcode == mt5.TRADE_RETCODE_DONE

    def update_consecutive_losses(self, pnl):
        """Update consecutive loss counter."""
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now(timezone.utc)
            self._save_system_state('consecutive_losses', self.consecutive_losses)
        else:
            self.consecutive_losses = 0
            self._save_system_state('consecutive_losses', 0)

    def reset_session_count(self):
        """Reset session trade counter."""
        self.session_trade_count = 0

    def close_database(self):
        """Close database connection."""
        if self.db_conn:
            self.db_conn.close()
