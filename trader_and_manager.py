"""
trader_and_manager.py - Execution, Position Sizing, Trade Management, and Failsafes

This module handles:
- Position sizing calculations
- Order execution (market and limit orders)
- Trade management (partial TPs, breakeven, trailing stop)
- Risk management and failsafes
- Position monitoring and exit logic
"""

import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class TradeManager:
    """Handles all trade execution, position sizing, and management."""

    def __init__(self, config):
        """
        Initialize the trade manager with configuration.

        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.symbol = config['symbol']
        self.point_value = config['point_value']

        # Risk settings
        self.risk_per_trade = config['risk']['risk_per_trade']
        self.max_position_risk = config['risk']['max_position_risk']
        self.max_concurrent_trades = config['risk']['max_concurrent_trades']
        self.daily_loss_limit = config['risk']['daily_loss_limit']

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

        # Entry models
        self.entry_models = config['entry_models']

        # Tracking
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.last_reset_date = None

        logger.info("TradeManager initialized")

    # ========================================================================
    # POSITION SIZING
    # ========================================================================

    def calculate_position_size(self, equity, entry_price, sl_price):
        """
        Calculate position size based on risk management rules.

        Formula:
        risk_dollars = equity * risk_per_trade
        sl_distance = abs(entry_price - sl_price)
        lot = risk_dollars / sl_distance / point_value

        Args:
            equity: Current account equity
            entry_price: Entry price
            sl_price: Stop loss price

        Returns:
            Float: Position size in lots (rounded to lot_step)
        """
        risk_dollars = equity * self.risk_per_trade
        sl_distance = abs(entry_price - sl_price)

        if sl_distance == 0:
            logger.error("SL distance is zero, cannot calculate position size")
            return 0.0

        lot = risk_dollars / (sl_distance * self.point_value)

        # Round to lot step
        lot = round(lot / self.lot_step) * self.lot_step

        # Enforce min/max
        lot = max(self.min_lot, min(lot, self.max_lot))

        logger.info(f"Position size calculated: {lot:.2f} lots (Risk: ${risk_dollars:.2f}, SL distance: ${sl_distance:.2f})")
        return lot

    # ========================================================================
    # ENTRY PRICE AND SL/TP CALCULATION
    # ========================================================================

    def calculate_entry_levels(self, analysis_result, df_m5):
        """
        Calculate entry, SL, and TP levels based on entry model.

        Args:
            analysis_result: Dictionary from analyst containing signals and setup data
            df_m5: M5 DataFrame with OHLC data

        Returns:
            Dictionary with entry_price, sl_price, tp1_price, tp2_price, entry_model, order_type
        """
        direction = analysis_result['direction']
        signals = analysis_result['signals']
        setup_data = analysis_result['setup_data']

        if not direction:
            return None

        idx = len(df_m5) - 1
        current_bar = df_m5.iloc[idx]

        # Determine which entry model to use
        entry_model = self._determine_entry_model(signals, df_m5, idx)

        if not entry_model:
            logger.warning("No valid entry model found")
            return None

        logger.info(f"Using entry model: {entry_model}")

        # Calculate based on entry model
        if entry_model == 'MOMENTUM_BREAK':
            return self._calculate_momentum_break_entry(direction, df_m5, idx, setup_data)

        elif entry_model == 'SHALLOW_RETEST':
            return self._calculate_shallow_retest_entry(direction, df_m5, idx, setup_data)

        elif entry_model == 'FVG_SNIPER':
            return self._calculate_fvg_sniper_entry(direction, df_m5, idx, setup_data)

        return None

    def _determine_entry_model(self, signals, df_m5, idx):
        """
        Determine which entry model to use based on signals.

        Priority:
        1. Momentum Break (if next bar breaks displacement)
        2. Shallow Retest (if in retrace range)
        3. FVG Sniper (if FVG exists)

        Args:
            signals: Detected signals
            df_m5: M5 DataFrame
            idx: Current index

        Returns:
            String: Entry model name or None
        """
        # Check if models are enabled
        if signals.get('sweep') and signals.get('impulse'):
            # Check for momentum break
            if self.entry_models['momentum_break']['enabled']:
                # If next bar breaks displacement low/high, use momentum
                if idx + 1 < len(df_m5):
                    next_bar = df_m5.iloc[idx + 1]
                    current_bar = df_m5.iloc[idx]

                    if signals['direction'] == 'BEARISH':
                        D_low = min(current_bar['open'], current_bar['close'])
                        if next_bar['low'] < D_low:
                            return 'MOMENTUM_BREAK'
                    else:  # BULLISH
                        D_high = max(current_bar['open'], current_bar['close'])
                        if next_bar['high'] > D_high:
                            return 'MOMENTUM_BREAK'

            # Check for shallow retest
            if self.entry_models['shallow_retest']['enabled']:
                # Calculate retrace (placeholder - needs real-time price)
                return 'SHALLOW_RETEST'

        # Check for FVG
        if signals.get('fvg') and self.entry_models['fvg_sniper']['enabled']:
            return 'FVG_SNIPER'

        # Default to shallow retest if sweep + impulse
        if signals.get('sweep') and signals.get('impulse'):
            if self.entry_models['shallow_retest']['enabled']:
                return 'SHALLOW_RETEST'

        return None

    def _calculate_momentum_break_entry(self, direction, df_m5, idx, setup_data):
        """
        Calculate momentum break entry (A).

        Entry at break of displacement low/high.
        """
        current_bar = df_m5.iloc[idx]

        if direction == 'BEARISH':
            D_low = min(current_bar['open'], current_bar['close'])
            D_high = max(current_bar['open'], current_bar['close'])

            entry_price = D_low
            sl_price = setup_data.get('swing_high', D_high) + self.stop_buffer

        else:  # BULLISH
            D_high = max(current_bar['open'], current_bar['close'])
            D_low = min(current_bar['open'], current_bar['close'])

            entry_price = D_high
            sl_price = setup_data.get('swing_low', D_low) - self.stop_buffer

        # Calculate SL distance and enforce minimum
        sl_distance = abs(entry_price - sl_price)
        if sl_distance < self.sl_min:
            sl_distance = self.sl_min
            if direction == 'BEARISH':
                sl_price = entry_price + sl_distance
            else:
                sl_price = entry_price - sl_distance

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

    def _calculate_shallow_retest_entry(self, direction, df_m5, idx, setup_data):
        """
        Calculate shallow retest entry (B).

        Entry at 30% retrace of displacement.
        """
        current_bar = df_m5.iloc[idx]
        shallow_ratio = self.config['retracement']['shallow_ratio']

        if direction == 'BEARISH':
            D_high = max(current_bar['open'], current_bar['close'])
            D_low = min(current_bar['open'], current_bar['close'])

            entry_price = D_low + shallow_ratio * (D_high - D_low)
            sl_price = setup_data.get('swing_high', D_high) + self.stop_buffer

        else:  # BULLISH
            D_high = max(current_bar['open'], current_bar['close'])
            D_low = min(current_bar['open'], current_bar['close'])

            entry_price = D_high - shallow_ratio * (D_high - D_low)
            sl_price = setup_data.get('swing_low', D_low) - self.stop_buffer

        # Calculate SL distance and enforce minimum
        sl_distance = abs(entry_price - sl_price)
        if sl_distance < self.sl_min:
            sl_distance = self.sl_min
            if direction == 'BEARISH':
                sl_price = entry_price + sl_distance
            else:
                sl_price = entry_price - sl_distance

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
            'entry_model': 'SHALLOW_RETEST',
            'order_type': 'LIMIT',
            'cancel_after_bars': self.entry_models['shallow_retest']['cancel_after_bars'],
            'sl_distance': sl_distance
        }

    def _calculate_fvg_sniper_entry(self, direction, df_m5, idx, setup_data):
        """
        Calculate FVG sniper entry (C).

        Entry at midpoint of FVG zone.
        """
        if idx < 2:
            return None

        current_bar = df_m5.iloc[idx]
        fvg_ratio = self.entry_models['fvg_sniper']['fvg_entry_ratio']

        if direction == 'BEARISH':
            # FVG zone: between low[i-2] and high[i]
            zone_high = df_m5.iloc[idx - 2]['low']
            zone_low = df_m5.iloc[idx]['high']

            if zone_low >= zone_high:
                logger.warning("Invalid bearish FVG zone")
                return None

            entry_price = zone_low + fvg_ratio * (zone_high - zone_low)
            sl_price = setup_data.get('swing_high', current_bar['high']) + self.stop_buffer

        else:  # BULLISH
            # FVG zone: between high[i-2] and low[i]
            zone_low = df_m5.iloc[idx - 2]['high']
            zone_high = df_m5.iloc[idx]['low']

            if zone_low >= zone_high:
                logger.warning("Invalid bullish FVG zone")
                return None

            entry_price = zone_low + fvg_ratio * (zone_high - zone_low)
            sl_price = setup_data.get('swing_low', current_bar['low']) - self.stop_buffer

        # Calculate SL distance and enforce minimum
        sl_distance = abs(entry_price - sl_price)
        if sl_distance < self.sl_min:
            sl_distance = self.sl_min
            if direction == 'BEARISH':
                sl_price = entry_price + sl_distance
            else:
                sl_price = entry_price - sl_distance

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
            'entry_model': 'FVG_SNIPER',
            'order_type': 'LIMIT',
            'cancel_after_bars': self.entry_models['fvg_sniper']['cancel_after_bars'],
            'sl_distance': sl_distance
        }

    # ========================================================================
    # ORDER EXECUTION
    # ========================================================================

    def execute_trade(self, direction, entry_levels, equity):
        """
        Execute trade with proper risk management.

        Args:
            direction: 'BULLISH' or 'BEARISH'
            entry_levels: Dictionary with entry prices and levels
            equity: Current account equity

        Returns:
            Dictionary with trade result
        """
        if not entry_levels:
            logger.error("No entry levels provided")
            return {'success': False, 'reason': 'No entry levels'}

        # Check daily loss limit
        if not self._check_daily_loss_limit(equity):
            logger.warning("Daily loss limit reached, blocking trade")
            return {'success': False, 'reason': 'Daily loss limit reached'}

        # Check concurrent positions
        if not self._check_concurrent_positions():
            logger.warning("Max concurrent positions reached, blocking trade")
            return {'success': False, 'reason': 'Max concurrent positions'}

        # Calculate position size
        lot_size = self.calculate_position_size(
            equity,
            entry_levels['entry_price'],
            entry_levels['sl_price']
        )

        if lot_size < self.min_lot:
            logger.error(f"Calculated lot size {lot_size} below minimum {self.min_lot}")
            return {'success': False, 'reason': 'Lot size too small'}

        # Execute based on order type
        if entry_levels['order_type'] == 'MARKET':
            return self._execute_market_order(direction, entry_levels, lot_size)
        else:  # LIMIT
            return self._execute_limit_order(direction, entry_levels, lot_size)

    def _execute_market_order(self, direction, entry_levels, lot_size):
        """Execute market order with IOC/FOK."""
        logger.info(f"Executing MARKET {direction} order: {lot_size} lots")

        # Prepare order request
        order_type = mt5.ORDER_TYPE_BUY if direction == 'BULLISH' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if direction == 'BULLISH' else mt5.symbol_info_tick(self.symbol).bid

        if price is None:
            logger.error("Failed to get current price")
            return {'success': False, 'reason': 'Failed to get price'}

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": entry_levels['sl_price'],
            "tp": entry_levels['tp1_price'],  # Set TP1 initially
            "deviation": 10,
            "magic": 234000,
            "comment": f"GOLDBOT_{entry_levels['entry_model']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)

        if result is None:
            logger.error("order_send failed, no result returned")
            return {'success': False, 'reason': 'order_send returned None'}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return {'success': False, 'reason': f"Order failed: {result.comment}"}

        # Check slippage
        max_slippage = self.entry_models['momentum_break']['max_slippage']
        actual_price = result.price
        slippage = abs(actual_price - price)

        if slippage > max_slippage:
            logger.warning(f"Slippage too high: {slippage:.2f} > {max_slippage}")
            # Try to close position
            self._close_position_by_ticket(result.order)
            return {'success': False, 'reason': f'Excessive slippage: {slippage:.2f}'}

        # Track position
        position_info = {
            'ticket': result.order,
            'direction': direction,
            'entry_price': actual_price,
            'sl_price': entry_levels['sl_price'],
            'tp1_price': entry_levels['tp1_price'],
            'tp2_price': entry_levels['tp2_price'],
            'lot_size': lot_size,
            'entry_model': entry_levels['entry_model'],
            'entry_time': datetime.now(),
            'sl_distance': entry_levels['sl_distance'],
            'partial_closed': False,
            'breakeven_moved': False,
            'trailing_active': False
        }

        self.active_positions[result.order] = position_info

        logger.info(f"✓ Trade executed successfully: Ticket {result.order}, Entry {actual_price:.2f}")
        return {'success': True, 'ticket': result.order, 'position_info': position_info}

    def _execute_limit_order(self, direction, entry_levels, lot_size):
        """Execute limit order."""
        logger.info(f"Placing LIMIT {direction} order: {lot_size} lots at {entry_levels['entry_price']:.2f}")

        # Prepare order request
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == 'BULLISH' else mt5.ORDER_TYPE_SELL_LIMIT

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_levels['entry_price'],
            "sl": entry_levels['sl_price'],
            "tp": entry_levels['tp1_price'],
            "deviation": 10,
            "magic": 234000,
            "comment": f"GOLDBOT_{entry_levels['entry_model']}",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        # Send order
        result = mt5.order_send(request)

        if result is None:
            logger.error("order_send failed for limit order")
            return {'success': False, 'reason': 'order_send returned None'}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Limit order failed: {result.retcode} - {result.comment}")
            return {'success': False, 'reason': f"Limit order failed: {result.comment}"}

        # Track pending order
        pending_info = {
            'ticket': result.order,
            'direction': direction,
            'entry_price': entry_levels['entry_price'],
            'sl_price': entry_levels['sl_price'],
            'tp1_price': entry_levels['tp1_price'],
            'tp2_price': entry_levels['tp2_price'],
            'lot_size': lot_size,
            'entry_model': entry_levels['entry_model'],
            'order_time': datetime.now(),
            'cancel_after_bars': entry_levels.get('cancel_after_bars', 5),
            'sl_distance': entry_levels['sl_distance']
        }

        self.active_positions[result.order] = pending_info

        logger.info(f"✓ Limit order placed: Ticket {result.order}")
        return {'success': True, 'ticket': result.order, 'pending': True, 'position_info': pending_info}

    # ========================================================================
    # TRADE MANAGEMENT
    # ========================================================================

    def manage_positions(self, current_price, atr_current):
        """
        Manage all active positions.

        - Partial close at TP1
        - Move to breakeven
        - Trailing stop
        - Time-based exit

        Args:
            current_price: Current market price
            atr_current: Current ATR value
        """
        if not self.active_positions:
            return

        positions_to_remove = []

        for ticket, pos_info in self.active_positions.items():
            # Skip pending orders (they are managed separately)
            if 'order_time' in pos_info:
                continue

            # Check if position still exists
            position = self._get_position_by_ticket(ticket)
            if not position:
                logger.info(f"Position {ticket} closed (TP/SL hit or manual close)")
                positions_to_remove.append(ticket)
                continue

            # Calculate R (risk units)
            entry_price = pos_info['entry_price']
            sl_distance = pos_info['sl_distance']
            current_sl = position.sl

            if pos_info['direction'] == 'BULLISH':
                current_profit = current_price - entry_price
            else:  # BEARISH
                current_profit = entry_price - current_price

            r_value = current_profit / sl_distance if sl_distance > 0 else 0

            logger.debug(f"Position {ticket}: R = {r_value:.2f}, Profit = ${current_profit:.2f}")

            # 1. Partial close at TP1
            if not pos_info['partial_closed'] and r_value >= self.tp1_mult:
                self._partial_close(ticket, pos_info, position)

            # 2. Move to breakeven
            if not pos_info['breakeven_moved'] and r_value >= self.breakeven_at_r:
                self._move_to_breakeven(ticket, pos_info, position, entry_price)

            # 3. Activate trailing stop
            if not pos_info['trailing_active'] and r_value >= self.trail_at_r:
                pos_info['trailing_active'] = True
                logger.info(f"Position {ticket}: Trailing stop activated at {r_value:.2f}R")

            # 4. Update trailing stop
            if pos_info['trailing_active']:
                self._update_trailing_stop(ticket, pos_info, position, current_price, atr_current)

            # 5. Time-based exit check
            entry_time = pos_info.get('entry_time')
            if entry_time:
                bars_held = (datetime.now() - entry_time).total_seconds() / 300  # M5 bars
                if bars_held >= self.max_hold_bars:
                    logger.warning(f"Position {ticket} held for {bars_held:.0f} bars, closing")
                    self._close_position_by_ticket(ticket)
                    positions_to_remove.append(ticket)

        # Remove closed positions
        for ticket in positions_to_remove:
            if ticket in self.active_positions:
                del self.active_positions[ticket]

    def _partial_close(self, ticket, pos_info, position):
        """Close partial position at TP1."""
        close_volume = position.volume * self.partial_pct
        close_volume = round(close_volume / self.lot_step) * self.lot_step

        if close_volume < self.min_lot:
            logger.warning(f"Partial close volume {close_volume} too small, skipping")
            return

        logger.info(f"Closing {self.partial_pct*100:.0f}% of position {ticket} at TP1")

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
            "comment": "GOLDBOT_PARTIAL_TP1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            pos_info['partial_closed'] = True
            logger.info(f"✓ Partial close successful for position {ticket}")
        else:
            logger.error(f"Partial close failed: {result.comment if result else 'No result'}")

    def _move_to_breakeven(self, ticket, pos_info, position, entry_price):
        """Move stop loss to breakeven."""
        logger.info(f"Moving position {ticket} to breakeven")

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
            logger.info(f"✓ Breakeven set for position {ticket}")
        else:
            logger.error(f"Breakeven move failed: {result.comment if result else 'No result'}")

    def _update_trailing_stop(self, ticket, pos_info, position, current_price, atr_current):
        """Update trailing stop based on ATR."""
        trail_distance = atr_current * self.trail_atr_mult

        if pos_info['direction'] == 'BULLISH':
            new_sl = current_price - trail_distance
            # Only move SL up
            if new_sl > position.sl:
                self._modify_sl(ticket, position, new_sl)
        else:  # BEARISH
            new_sl = current_price + trail_distance
            # Only move SL down
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
            logger.info(f"✓ Trailing SL updated for {ticket}: {new_sl:.2f}")
        else:
            logger.debug(f"SL modification failed: {result.comment if result else 'No result'}")

    # ========================================================================
    # POSITION HELPERS
    # ========================================================================

    def _get_position_by_ticket(self, ticket):
        """Get position by ticket number."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                if pos.ticket == ticket:
                    return pos
        return None

    def _close_position_by_ticket(self, ticket):
        """Close position by ticket."""
        position = self._get_position_by_ticket(ticket)
        if not position:
            logger.warning(f"Position {ticket} not found")
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

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✓ Position {ticket} closed")
            return True
        else:
            logger.error(f"Failed to close position {ticket}: {result.comment if result else 'No result'}")
            return False

    # ========================================================================
    # RISK CHECKS
    # ========================================================================

    def _check_daily_loss_limit(self, equity):
        """Check if daily loss limit is reached."""
        today = datetime.now().date()

        # Reset daily tracking if new day
        if self.last_reset_date != today:
            self.daily_start_equity = equity
            self.daily_pnl = 0.0
            self.last_reset_date = today
            logger.info(f"Daily tracking reset: Starting equity ${equity:.2f}")

        # Calculate daily loss
        daily_loss = (self.daily_start_equity - equity) / self.daily_start_equity

        if daily_loss >= self.daily_loss_limit:
            logger.error(f"CIRCUIT BREAKER: Daily loss {daily_loss*100:.2f}% >= {self.daily_loss_limit*100:.2f}%")
            return False

        return True

    def _check_concurrent_positions(self):
        """Check if max concurrent positions is reached."""
        active_count = len([p for p in self.active_positions.values() if 'entry_time' in p])

        if active_count >= self.max_concurrent_trades:
            logger.warning(f"Max concurrent positions reached: {active_count}/{self.max_concurrent_trades}")
            return False

        return True

    def get_current_exposure(self, equity):
        """Calculate current position exposure as % of equity."""
        total_risk = 0.0

        for pos_info in self.active_positions.values():
            if 'entry_time' in pos_info:  # Active position
                lot_size = pos_info['lot_size']
                sl_distance = pos_info['sl_distance']
                risk_dollars = lot_size * sl_distance * self.point_value
                total_risk += risk_dollars

        exposure_pct = (total_risk / equity) * 100 if equity > 0 else 0
        return exposure_pct
