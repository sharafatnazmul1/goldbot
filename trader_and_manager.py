"""
trader_and_manager.py - CORRECTED VERSION
Execution, Position Sizing, Trade Management, State Persistence

Fixes applied:
- #2, #3: Proper entry execution without look-ahead bias
- #4: Complete pending order management
- #7: UTC timezone consistency
- #12: Order state machine
- #13: Correct slippage measurement
- #14: Remove TP1 conflict (manual only)
- #16: Correlation management
- #17: Complete exposure calculation
- #18, #19: Drawdown limits and throttling
- #24: Integrated state persistence (SQLite)
"""

import MetaTrader5 as mt5
import logging
import sqlite3
import json
from datetime import datetime, timedelta, timezone
from enum import Enum
import time

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Order lifecycle states (FIX #12)."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL_FILLED = "partial_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    CLOSED = "closed"


class TradeManager:
    """
    Handles all trade execution, position sizing, management, and state persistence.

    Integrated state management to avoid extra modules.
    """

    def __init__(self, config):
        """Initialize trade manager with config."""
        self.config = config
        self.symbol = config['symbol']
        self.point_value = config['point_value']

        # Risk settings
        self.risk_per_trade = config['risk']['risk_per_trade']
        self.max_position_risk = config['risk']['max_position_risk']
        self.max_concurrent_trades = config['risk']['max_concurrent_trades']
        self.daily_loss_limit = config['risk']['daily_loss_limit']
        self.weekly_loss_limit = config['risk'].get('weekly_loss_limit', 0.08)
        self.max_drawdown_limit = config['risk'].get('max_drawdown_limit', 0.15)
        self.consecutive_loss_limit = config['risk'].get('consecutive_loss_limit', 5)

        # Position sizing
        self.min_lot = config['position']['min_lot']
        self.max_lot = config['position']['max_lot']
        self.lot_step = config['position']['lot_step']

        # SL/TP settings
        self.stop_buffer = config['stop_loss']['buffer']
        self.sl_min = config['stop_loss']['minimum']
        self.tp1_mult = config['take_profit']['tp1_multiplier']
        self.tp2_mult = config['take_profit']['tp2_multiplier']
        self.partial_pct = config['take_profit']['partial_close_pct']

        # Trade management
        self.breakeven_at_r = config['trade_management']['breakeven_at_r']
        self.trail_at_r = config['trade_management']['trail_at_r']
        self.trail_atr_mult = config['trade_management']['trail_atr_mult']
        self.max_hold_bars = config['trade_management']['max_hold_bars']
        self.min_time_between_trades = config['trade_management'].get('min_time_between_trades', 300)
        self.max_trades_per_session = config['trade_management'].get('max_trades_per_session', 4)
        self.cooloff_after_loss = config['trade_management'].get('cooloff_after_loss', 900)

        # Entry models
        self.entry_models = config['entry_models']

        # Tracking
        self.active_positions = {}
        self.pending_orders = {}

        # Performance tracking (FIX #18)
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self.weekly_start_equity = 0.0
        self.last_daily_reset = None
        self.last_weekly_reset = None
        self.consecutive_losses = 0

        # Throttling (FIX #19)
        self.last_trade_time = None
        self.session_trade_count = 0
        self.last_loss_time = None
        self.current_session = None

        # Integrated state persistence (FIX #24)
        self._init_database()
        self._load_state()

        logger.info("TradeManager initialized with state persistence")

    # ========================================================================
    # STATE PERSISTENCE (Integrated - FIX #24)
    # ========================================================================

    def _init_database(self):
        """Initialize SQLite database for state persistence."""
        self.db_conn = sqlite3.connect('goldbot_state.db', check_same_thread=False)
        cursor = self.db_conn.cursor()

        # Positions table
        cursor.execute('''
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

        # System state table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        ''')

        # Trade history
        cursor.execute('''
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

        self.db_conn.commit()

    def _load_state(self):
        """Load state from database on startup."""
        cursor = self.db_conn.cursor()

        # Load active positions
        cursor.execute('SELECT data_json FROM positions WHERE state = ?', (OrderState.FILLED.value,))
        for row in cursor.fetchall():
            pos_data = json.loads(row[0])
            pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
            self.active_positions[pos_data['ticket']] = pos_data

        # Load pending orders
        cursor.execute('SELECT data_json FROM positions WHERE state = ?', (OrderState.PENDING.value,))
        for row in cursor.fetchall():
            order_data = json.loads(row[0])
            order_data['order_time'] = datetime.fromisoformat(order_data['order_time'])
            self.pending_orders[order_data['ticket']] = order_data

        # Load system state
        cursor.execute('SELECT key, value FROM system_state')
        for row in cursor.fetchall():
            key, value_json = row
            value = json.loads(value_json)

            if key == 'peak_equity':
                self.peak_equity = value
            elif key == 'consecutive_losses':
                self.consecutive_losses = value

        logger.info(f"Loaded {len(self.active_positions)} positions, {len(self.pending_orders)} pending orders from DB")

    def _save_position(self, pos_info):
        """Save position to database."""
        cursor = self.db_conn.cursor()

        pos_json = pos_info.copy()
        if 'entry_time' in pos_json and isinstance(pos_json['entry_time'], datetime):
            pos_json['entry_time'] = pos_json['entry_time'].isoformat()
        if 'order_time' in pos_json and isinstance(pos_json['order_time'], datetime):
            pos_json['order_time'] = pos_json['order_time'].isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            pos_info['ticket'],
            pos_info.get('direction'),
            pos_info.get('entry_price'),
            pos_info.get('sl_price'),
            pos_info.get('tp1_price'),
            pos_info.get('tp2_price'),
            pos_info.get('lot_size'),
            pos_info.get('entry_model'),
            pos_json.get('entry_time') or pos_json.get('order_time'),
            pos_info.get('sl_distance'),
            pos_info.get('state', OrderState.FILLED.value),
            int(pos_info.get('partial_closed', False)),
            int(pos_info.get('breakeven_moved', False)),
            int(pos_info.get('trailing_active', False)),
            json.dumps(pos_json)
        ))
        self.db_conn.commit()

    def _remove_position(self, ticket):
        """Remove position from database."""
        cursor = self.db_conn.cursor()
        cursor.execute('DELETE FROM positions WHERE ticket = ?', (ticket,))
        self.db_conn.commit()

    def _save_system_state(self, key, value):
        """Save system state value."""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO system_state VALUES (?, ?, ?)
        ''', (key, json.dumps(value), datetime.now(timezone.utc).isoformat()))
        self.db_conn.commit()

    def _save_trade(self, trade_info):
        """Save completed trade to history."""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO trade_history
            (ticket, direction, entry_price, exit_price, lot_size, entry_time,
             exit_time, pnl, r_multiple, entry_model, exit_reason, score)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            trade_info.get('ticket'),
            trade_info.get('direction'),
            trade_info.get('entry_price'),
            trade_info.get('exit_price'),
            trade_info.get('lot_size'),
            trade_info.get('entry_time').isoformat() if isinstance(trade_info.get('entry_time'), datetime) else trade_info.get('entry_time'),
            datetime.now(timezone.utc).isoformat(),
            trade_info.get('pnl'),
            trade_info.get('r_multiple'),
            trade_info.get('entry_model'),
            trade_info.get('exit_reason'),
            trade_info.get('score')
        ))
        self.db_conn.commit()

    # ========================================================================
    # POSITION SIZING (FIX #1 applied via config)
    # ========================================================================

    def calculate_position_size(self, equity, entry_price, sl_price):
        """
        Calculate position size with correct point_value (100.0 for XAUUSD).

        FIX #1: point_value now correctly set to 100.0 in config.
        """
        risk_dollars = equity * self.risk_per_trade
        sl_distance = abs(entry_price - sl_price)

        if sl_distance == 0:
            logger.error("SL distance is zero")
            return 0.0

        lot = risk_dollars / (sl_distance * self.point_value)

        # Round to lot step
        lot = round(lot / self.lot_step) * self.lot_step

        # Enforce limits
        lot = max(self.min_lot, min(lot, self.max_lot))

        logger.info(f"Position size: {lot:.2f} lots (Risk ${risk_dollars:.2f}, SL ${sl_distance:.2f})")
        return lot

    # ========================================================================
    # ENTRY LEVELS CALCULATION (FIX #2, #3)
    # ========================================================================

    def calculate_entry_levels(self, analysis_result, current_price):
        """
        Calculate entry levels WITHOUT look-ahead bias (FIX #2, #3).

        Uses current market price instead of historical bar prices.
        """
        direction = analysis_result['direction']
        signals = analysis_result['signals']
        setup_data = analysis_result['setup_data']

        if not direction:
            return None

        # Determine entry model based on signals (no future bar checking)
        entry_model = self._determine_entry_model_no_lookahead(signals)

        if not entry_model:
            logger.warning("No valid entry model")
            return None

        logger.info(f"Using entry model: {entry_model}")

        # Calculate levels based on model
        if entry_model == 'MOMENTUM_BREAK':
            return self._calc_momentum_entry(direction, current_price, setup_data)
        elif entry_model == 'SHALLOW_RETEST':
            return self._calc_shallow_entry(direction, current_price, setup_data, analysis_result)
        elif entry_model == 'FVG_SNIPER':
            return self._calc_fvg_entry(direction, current_price, setup_data)

        return None

    def _determine_entry_model_no_lookahead(self, signals):
        """
        Determine entry model WITHOUT future bar access (FIX #2).

        Original checked idx+1 (future). Now uses current signals only.
        """
        # If sweep + impulse detected, prefer momentum or shallow retest
        if signals.get('sweep') and signals.get('impulse'):
            # Momentum break: use stop order just beyond displacement
            if self.entry_models['momentum_break']['enabled']:
                return 'MOMENTUM_BREAK'

            # Or shallow retest if momentum disabled
            if self.entry_models['shallow_retest']['enabled']:
                return 'SHALLOW_RETEST'

        # If FVG exists
        if signals.get('fvg') and self.entry_models['fvg_sniper']['enabled']:
            return 'FVG_SNIPER'

        # Default to shallow if sweep+impulse
        if signals.get('sweep') and signals.get('impulse'):
            if self.entry_models['shallow_retest']['enabled']:
                return 'SHALLOW_RETEST'

        return None

    def _calc_momentum_entry(self, direction, current_price, setup_data):
        """
        Momentum break entry (FIX #3).

        Use current market price for market order, NOT historical bar price.
        """
        # FIX #3: Use CURRENT price, not historical D_low/D_high
        entry_price = current_price

        if direction == 'BEARISH':
            sl_price = setup_data.get('swing_high', current_price + 5.0) + self.stop_buffer
        else:  # BULLISH
            sl_price = setup_data.get('swing_low', current_price - 5.0) - self.stop_buffer

        # Calculate SL distance
        sl_distance = abs(entry_price - sl_price)
        if sl_distance < self.sl_min:
            sl_distance = self.sl_min
            sl_price = entry_price + sl_distance if direction == 'BEARISH' else entry_price - sl_distance

        # Calculate TPs
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

    def _calc_shallow_entry(self, direction, current_price, setup_data, analysis_result):
        """Shallow retest entry - limit order at retrace level."""
        # Place limit order at shallow retrace from current price
        shallow_ratio = self.config['retracement']['shallow_ratio']

        # Estimate displacement zone from swing levels
        if direction == 'BEARISH':
            swing_high = setup_data.get('swing_high', current_price + 5.0)
            entry_price = current_price + (swing_high - current_price) * shallow_ratio
            sl_price = swing_high + self.stop_buffer
        else:  # BULLISH
            swing_low = setup_data.get('swing_low', current_price - 5.0)
            entry_price = current_price - (current_price - swing_low) * shallow_ratio
            sl_price = swing_low - self.stop_buffer

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
        """FVG sniper entry - limit order in gap zone."""
        # Simplified: use offset from current price
        fvg_ratio = self.entry_models['fvg_sniper']['fvg_entry_ratio']

        if direction == 'BEARISH':
            swing_high = setup_data.get('swing_high', current_price + 3.0)
            entry_price = current_price + (swing_high - current_price) * fvg_ratio
            sl_price = swing_high + self.stop_buffer
        else:
            swing_low = setup_data.get('swing_low', current_price - 3.0)
            entry_price = current_price - (current_price - swing_low) * fvg_ratio
            sl_price = swing_low - self.stop_buffer

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

    # ========================================================================
    # RISK CHECKS (FIX #16, #17, #18, #19)
    # ========================================================================

    def _update_peak_equity(self, equity):
        """Track peak equity for drawdown calc (FIX #18)."""
        if equity > self.peak_equity:
            self.peak_equity = equity
            self._save_system_state('peak_equity', equity)

    def _check_drawdown_limits(self, equity):
        """Check drawdown circuit breakers (FIX #18)."""
        self._update_peak_equity(equity)

        # Max drawdown check
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown >= self.max_drawdown_limit:
                logger.error(f"🚨 MAX DRAWDOWN BREAKER: {drawdown*100:.1f}% >= {self.max_drawdown_limit*100:.1f}%")
                return False

        # Consecutive losses check
        if self.consecutive_losses >= self.consecutive_loss_limit:
            logger.error(f"🚨 CONSECUTIVE LOSS BREAKER: {self.consecutive_losses} losses")
            return False

        return True

    def _check_daily_loss_limit(self, equity):
        """Check daily loss limit (FIX #7 - UTC timezone)."""
        today = datetime.now(timezone.utc).date()

        if self.last_daily_reset != today:
            self.daily_start_equity = equity
            self.last_daily_reset = today
            logger.info(f"Daily reset: Starting equity ${equity:.2f}")

        if self.daily_start_equity > 0:
            daily_loss = (self.daily_start_equity - equity) / self.daily_start_equity
            if daily_loss >= self.daily_loss_limit:
                logger.error(f"🚨 DAILY LOSS LIMIT: {daily_loss*100:.1f}%")
                return False

        return True

    def _check_weekly_loss_limit(self, equity):
        """Check weekly loss limit (FIX #18)."""
        now = datetime.now(timezone.utc)
        week_num = now.isocalendar()[1]

        if self.last_weekly_reset is None or self.last_weekly_reset != week_num:
            self.weekly_start_equity = equity
            self.last_weekly_reset = week_num
            logger.info(f"Weekly reset: Starting equity ${equity:.2f}")

        if self.weekly_start_equity > 0:
            weekly_loss = (self.weekly_start_equity - equity) / self.weekly_start_equity
            if weekly_loss >= self.weekly_loss_limit:
                logger.error(f"🚨 WEEKLY LOSS LIMIT: {weekly_loss*100:.1f}%")
                return False

        return True

    def _check_trade_throttling(self):
        """Check trade frequency limits (FIX #19)."""
        now = datetime.now(timezone.utc)

        # Min time between trades
        if self.last_trade_time:
            seconds_since = (now - self.last_trade_time).total_seconds()
            if seconds_since < self.min_time_between_trades:
                logger.info(f"Throttle: Only {seconds_since:.0f}s since last trade")
                return False

        # Session trade limit (reset per session)
        if self.session_trade_count >= self.max_trades_per_session:
            logger.info(f"Throttle: {self.session_trade_count} trades this session")
            return False

        # Cooloff after loss
        if self.last_loss_time:
            seconds_since_loss = (now - self.last_loss_time).total_seconds()
            if seconds_since_loss < self.cooloff_after_loss:
                logger.info(f"Cooloff: {seconds_since_loss:.0f}s since loss")
                return False

        return True

    def _check_correlation_limits(self):
        """Prevent multiple same-direction positions (FIX #16)."""
        # For single instrument (XAUUSD), limit to 1 per direction
        bull_count = sum(1 for p in self.active_positions.values() if p.get('direction') == 'BULLISH')
        bear_count = sum(1 for p in self.active_positions.values() if p.get('direction') == 'BEARISH')

        return bull_count < 2 and bear_count < 2

    def _check_concurrent_positions(self):
        """Check max concurrent positions."""
        active_count = len(self.active_positions)
        return active_count < self.max_concurrent_trades

    def get_current_exposure(self, equity):
        """
        Calculate complete exposure including unrealized P&L (FIX #17).
        """
        total_risk = 0.0
        unrealized_pnl = 0.0

        # Risk from all positions (active + pending)
        for pos_info in list(self.active_positions.values()) + list(self.pending_orders.values()):
            lot_size = pos_info.get('lot_size', 0)
            sl_distance = pos_info.get('sl_distance', 0)
            risk_dollars = lot_size * sl_distance * self.point_value
            total_risk += risk_dollars

        # Unrealized P&L from active positions
        for ticket, pos_info in self.active_positions.items():
            position = self._get_position_by_ticket(ticket)
            if position:
                unrealized_pnl += position.profit

        # Effective equity
        effective_equity = equity + unrealized_pnl
        exposure_pct = (total_risk / effective_equity) * 100 if effective_equity > 0 else 0

        return exposure_pct, unrealized_pnl

    # ========================================================================
    # ORDER EXECUTION (FIX #3, #12, #13, #14)
    # ========================================================================

    def execute_trade(self, direction, entry_levels, equity):
        """Execute trade with all risk checks."""
        if not entry_levels:
            return {'success': False, 'reason': 'No entry levels'}

        # Check all risk limits
        if not self._check_drawdown_limits(equity):
            return {'success': False, 'reason': 'Drawdown limit reached'}

        if not self._check_daily_loss_limit(equity):
            return {'success': False, 'reason': 'Daily loss limit'}

        if not self._check_weekly_loss_limit(equity):
            return {'success': False, 'reason': 'Weekly loss limit'}

        if not self._check_trade_throttling():
            return {'success': False, 'reason': 'Trade throttled'}

        if not self._check_correlation_limits():
            return {'success': False, 'reason': 'Correlation limit'}

        if not self._check_concurrent_positions():
            return {'success': False, 'reason': 'Max concurrent positions'}

        # Calculate lot size
        lot_size = self.calculate_position_size(equity, entry_levels['entry_price'], entry_levels['sl_price'])

        if lot_size < self.min_lot:
            return {'success': False, 'reason': 'Lot size too small'}

        # Execute
        if entry_levels['order_type'] == 'MARKET':
            return self._execute_market_order(direction, entry_levels, lot_size)
        else:
            return self._execute_limit_order(direction, entry_levels, lot_size)

    def _execute_market_order(self, direction, entry_levels, lot_size):
        """
        Execute market order with correct slippage check (FIX #13).
        Set TP2 only, not TP1 (FIX #14).
        """
        logger.info(f"Executing MARKET {direction}: {lot_size} lots")

        order_type = mt5.ORDER_TYPE_BUY if direction == 'BULLISH' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if direction == 'BULLISH' else mt5.symbol_info_tick(self.symbol).bid

        if price is None:
            return {'success': False, 'reason': 'Failed to get price'}

        # FIX #14: Set TP2 on order, TP1 handled manually
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": entry_levels['sl_price'],
            "tp": entry_levels['tp2_price'],  # FIX #14: TP2 only
            "deviation": 10,
            "magic": 234000,
            "comment": f"GOLDBOT_{entry_levels['entry_model']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment if result else 'None'}")
            return {'success': False, 'reason': 'Order rejected'}

        # FIX #13: Check slippage against INTENDED entry, not market price
        intended_entry = entry_levels['entry_price']
        actual_price = result.price
        slippage = abs(actual_price - intended_entry)

        max_slippage = self.entry_models['momentum_break'].get('max_slippage', 0.05)
        if slippage > max_slippage:
            logger.warning(f"Slippage {slippage:.2f} > {max_slippage}, closing")
            self._close_position_by_ticket(result.order)
            return {'success': False, 'reason': f'Excessive slippage: {slippage:.2f}'}

        # Track position with state machine (FIX #12)
        position_info = {
            'ticket': result.order,
            'direction': direction,
            'entry_price': actual_price,
            'sl_price': entry_levels['sl_price'],
            'tp1_price': entry_levels['tp1_price'],
            'tp2_price': entry_levels['tp2_price'],
            'lot_size': lot_size,
            'entry_model': entry_levels['entry_model'],
            'entry_time': datetime.now(timezone.utc),
            'sl_distance': entry_levels['sl_distance'],
            'state': OrderState.FILLED.value,
            'partial_closed': False,
            'breakeven_moved': False,
            'trailing_active': False
        }

        self.active_positions[result.order] = position_info
        self._save_position(position_info)

        # Update throttling
        self.last_trade_time = datetime.now(timezone.utc)
        self.session_trade_count += 1

        logger.info(f"✓ Trade executed: Ticket {result.order}, Entry {actual_price:.2f}")
        return {'success': True, 'ticket': result.order, 'position_info': position_info}

    def _execute_limit_order(self, direction, entry_levels, lot_size):
        """Execute limit order and track as pending (FIX #4, #12)."""
        logger.info(f"Placing LIMIT {direction}: {lot_size} lots at {entry_levels['entry_price']:.2f}")

        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == 'BULLISH' else mt5.ORDER_TYPE_SELL_LIMIT

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_levels['entry_price'],
            "sl": entry_levels['sl_price'],
            "tp": entry_levels['tp2_price'],  # FIX #14
            "deviation": 10,
            "magic": 234000,
            "comment": f"GOLDBOT_{entry_levels['entry_model']}",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)

        if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Limit order failed: {result.comment if result else 'None'}")
            return {'success': False, 'reason': 'Limit order rejected'}

        # Track as pending (FIX #4, #12)
        pending_info = {
            'ticket': result.order,
            'direction': direction,
            'entry_price': entry_levels['entry_price'],
            'sl_price': entry_levels['sl_price'],
            'tp1_price': entry_levels['tp1_price'],
            'tp2_price': entry_levels['tp2_price'],
            'lot_size': lot_size,
            'entry_model': entry_levels['entry_model'],
            'order_time': datetime.now(timezone.utc),
            'cancel_after_bars': entry_levels.get('cancel_after_bars', 5),
            'sl_distance': entry_levels['sl_distance'],
            'state': OrderState.PENDING.value
        }

        self.pending_orders[result.order] = pending_info
        self._save_position(pending_info)

        logger.info(f"✓ Limit order placed: Ticket {result.order}")
        return {'success': True, 'ticket': result.order, 'pending': True}

    # ========================================================================
    # PENDING ORDER MANAGEMENT (FIX #4)
    # ========================================================================

    def manage_pending_orders(self):
        """
        Manage pending orders lifecycle (FIX #4).

        Check if filled, cancel if stale.
        """
        for ticket, order_info in list(self.pending_orders.items()):
            # Check if filled
            position = self._get_position_by_ticket(ticket)
            if position:
                # Order filled! Move to active
                logger.info(f"Pending order {ticket} filled at {position.price_open:.2f}")

                order_info['entry_price'] = position.price_open
                order_info['entry_time'] = datetime.now(timezone.utc)
                order_info['state'] = OrderState.FILLED.value
                del order_info['order_time']
                del order_info['cancel_after_bars']

                self.active_positions[ticket] = order_info
                del self.pending_orders[ticket]
                self._save_position(order_info)

                # Update throttling
                self.last_trade_time = datetime.now(timezone.utc)
                self.session_trade_count += 1
                continue

            # Check if still pending in MT5
            order = self._get_pending_order(ticket)
            if not order:
                # Cancelled or rejected
                logger.info(f"Pending order {ticket} no longer exists")
                del self.pending_orders[ticket]
                self._remove_position(ticket)
                continue

            # Check timeout
            bars_waited = (datetime.now(timezone.utc) - order_info['order_time']).total_seconds() / 300
            if bars_waited >= order_info.get('cancel_after_bars', 5):
                logger.info(f"Cancelling stale pending order {ticket} after {bars_waited:.1f} bars")
                self._cancel_pending_order(ticket)
                del self.pending_orders[ticket]
                self._remove_position(ticket)

    def _get_pending_order(self, ticket):
        """Get pending order from MT5."""
        orders = mt5.orders_get(symbol=self.symbol)
        if orders:
            for order in orders:
                if order.ticket == ticket:
                    return order
        return None

    def _cancel_pending_order(self, ticket):
        """Cancel pending order."""
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✓ Pending order {ticket} cancelled")
            return True
        return False

    # ========================================================================
    # TRADE MANAGEMENT
    # ========================================================================

    def manage_positions(self, current_price, atr_current):
        """Manage all active positions."""
        if not self.active_positions:
            return

        positions_to_remove = []

        for ticket, pos_info in list(self.active_positions.items()):
            position = self._get_position_by_ticket(ticket)
            if not position:
                logger.info(f"Position {ticket} closed (TP/SL hit)")
                positions_to_remove.append(ticket)

                # Record trade
                trade_info = pos_info.copy()
                trade_info['exit_reason'] = 'TP_SL'
                self._save_trade(trade_info)
                continue

            # Calculate R
            entry_price = pos_info['entry_price']
            sl_distance = pos_info['sl_distance']

            if pos_info['direction'] == 'BULLISH':
                current_profit = current_price - entry_price
            else:
                current_profit = entry_price - current_price

            r_value = current_profit / sl_distance if sl_distance > 0 else 0

            logger.debug(f"Position {ticket}: R={r_value:.2f}, P/L=${current_profit:.2f}")

            # Partial close at TP1 (manual, FIX #14)
            if not pos_info['partial_closed'] and r_value >= self.tp1_mult:
                self._partial_close(ticket, pos_info, position)

            # Breakeven
            if not pos_info['breakeven_moved'] and r_value >= self.breakeven_at_r:
                self._move_to_breakeven(ticket, pos_info, position, entry_price)

            # Trailing
            if not pos_info['trailing_active'] and r_value >= self.trail_at_r:
                pos_info['trailing_active'] = True
                logger.info(f"Trailing activated for {ticket}")

            if pos_info['trailing_active']:
                self._update_trailing_stop(ticket, pos_info, position, current_price, atr_current)

            # Time-based exit
            entry_time = pos_info.get('entry_time')
            if entry_time:
                bars_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 300
                if bars_held >= self.max_hold_bars:
                    logger.warning(f"Max hold time reached for {ticket}, closing")
                    self._close_position_by_ticket(ticket)
                    positions_to_remove.append(ticket)

        # Remove closed
        for ticket in positions_to_remove:
            if ticket in self.active_positions:
                del self.active_positions[ticket]
                self._remove_position(ticket)

    def _partial_close(self, ticket, pos_info, position):
        """Close 50% at TP1."""
        close_volume = position.volume * self.partial_pct
        close_volume = round(close_volume / self.lot_step) * self.lot_step

        if close_volume < self.min_lot:
            logger.warning(f"Partial volume {close_volume} too small")
            return

        logger.info(f"Closing 50% of {ticket} at TP1")

        order_type = mt5.ORDER_TYPE_SELL if pos_info['direction'] == 'BULLISH' else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol).bid if pos_info['direction'] == 'BULLISH' else mt5.symbol_info_tick(self.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": close_volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "GOLDBOT_TP1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            pos_info['partial_closed'] = True
            self._save_position(pos_info)
            logger.info(f"✓ Partial close successful")

    def _move_to_breakeven(self, ticket, pos_info, position, entry_price):
        """Move SL to breakeven."""
        logger.info(f"Moving {ticket} to breakeven")

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": entry_price,
            "tp": position.tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            pos_info['breakeven_moved'] = True
            self._save_position(pos_info)
            logger.info(f"✓ Breakeven set")

    def _update_trailing_stop(self, ticket, pos_info, position, current_price, atr_current):
        """Update trailing stop."""
        trail_distance = atr_current * self.trail_atr_mult

        if pos_info['direction'] == 'BULLISH':
            new_sl = current_price - trail_distance
            if new_sl > position.sl:
                self._modify_sl(ticket, position, new_sl)
        else:
            new_sl = current_price + trail_distance
            if new_sl < position.sl:
                self._modify_sl(ticket, position, new_sl)

    def _modify_sl(self, ticket, position, new_sl):
        """Modify stop loss."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": position.tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✓ Trailing SL updated: {new_sl:.2f}")

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _get_position_by_ticket(self, ticket):
        """Get position from MT5."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                if pos.ticket == ticket:
                    return pos
        return None

    def _close_position_by_ticket(self, ticket):
        """Close position completely."""
        position = self._get_position_by_ticket(ticket)
        if not position:
            return False

        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "GOLDBOT_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        return result and result.retcode == mt5.TRADE_RETCODE_DONE

    def update_consecutive_losses(self, pnl):
        """Track consecutive losses for circuit breaker (FIX #18)."""
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now(timezone.utc)
            logger.info(f"Consecutive losses: {self.consecutive_losses}")
            self._save_system_state('consecutive_losses', self.consecutive_losses)
        else:
            if self.consecutive_losses > 0:
                logger.info("Win streak started, resetting consecutive losses")
            self.consecutive_losses = 0
            self._save_system_state('consecutive_losses', 0)

    def reset_session_count(self):
        """Reset session trade counter when new session starts."""
        self.session_trade_count = 0
        logger.info("Session trade count reset")

    def close_database(self):
        """Close database connection."""
        if self.db_conn:
            self.db_conn.close()
