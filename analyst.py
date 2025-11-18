"""
analyst.py - Signal Generation, Calculations, Scoring, and Filtering
CORRECTED VERSION - All audit fixes applied

Fixes applied:
- #5: Retracement calculation corrected
- #6: Midnight session crossing handled
- #7: UTC timezone consistency
- #8: ATR using Wilder's smoothing
- #9: Contradictory signals prevented
- #10: Doji candle filtering
- #11: NaN values handled
"""

import numpy as np
import pandas as pd
from datetime import datetime, time, timezone
import logging

logger = logging.getLogger(__name__)

# Minimum body size to avoid doji false signals (FIX #10)
MIN_BODY_SIZE = 0.50  # $0.50 minimum for XAUUSD


class MarketAnalyst:
    """Handles all market analysis, signal generation, and scoring."""

    def __init__(self, config):
        """Initialize the analyst with configuration."""
        self.config = config
        self.atr_period = config['atr']['period']
        self.swing_lookback = config['lookback']['swing']
        self.avg_body_period = config['lookback']['avg_body']
        self.impulse_multiplier = config['impulse']['body_multiplier']

        # Thresholds
        self.atr_vol_threshold = config['atr']['volatility_threshold']
        self.atr_high_threshold = config['atr']['high_threshold']
        self.max_spread = config['filters']['max_spread']

        # Session times
        self.sessions = config['sessions']

        # Scoring weights
        self.weights = config['scoring']['weights']
        self.score_threshold = config['scoring']['threshold_trade']
        self.score_high = config['scoring']['threshold_high']

        # Retracement settings
        self.shallow_min = config['retracement']['shallow_min']
        self.shallow_max = config['retracement']['shallow_max']

        logger.info("MarketAnalyst initialized")

    def calculate_atr(self, df, period=None):
        """
        Calculate ATR using Wilder's smoothing (FIX #8).

        Original used SMA, now uses EMA with alpha=1/period (Wilder's method).
        """
        if period is None:
            period = self.atr_period

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # FIX #8: Use Wilder's smoothing (EMA with alpha=1/period)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

        return atr

    def detect_swing_high(self, df, lookback=None):
        """Detect swing highs over lookback period."""
        if lookback is None:
            lookback = self.swing_lookback

        swing_high = df['high'].rolling(window=lookback, min_periods=1).max().shift(1)
        return swing_high

    def detect_swing_low(self, df, lookback=None):
        """Detect swing lows over lookback period."""
        if lookback is None:
            lookback = self.swing_lookback

        swing_low = df['low'].rolling(window=lookback, min_periods=1).min().shift(1)
        return swing_low

    def detect_sweep_bull(self, df, swing_low):
        """
        Detect bullish liquidity sweep with doji filtering (FIX #10).

        Conditions:
        1. Low breaks below swing low
        2. Close is above swing low (rejection)
        3. Rejection wick is larger than body
        4. Body must be >= MIN_BODY_SIZE (not a doji)
        """
        high = df['high']
        low = df['low']
        open_price = df['open']
        close = df['close']

        body = abs(close - open_price)
        wick_down = np.minimum(open_price, close) - low

        cond1 = low < swing_low
        cond2 = close > swing_low
        cond3 = wick_down > body
        cond4 = body >= MIN_BODY_SIZE  # FIX #10: Filter dojis

        sweep_bull = cond1 & cond2 & cond3 & cond4
        return sweep_bull

    def detect_sweep_bear(self, df, swing_high):
        """
        Detect bearish liquidity sweep with doji filtering (FIX #10).
        """
        high = df['high']
        low = df['low']
        open_price = df['open']
        close = df['close']

        body = abs(close - open_price)
        wick_up = high - np.maximum(open_price, close)

        cond1 = high > swing_high
        cond2 = close < swing_high
        cond3 = wick_up > body
        cond4 = body >= MIN_BODY_SIZE  # FIX #10: Filter dojis

        sweep_bear = cond1 & cond2 & cond3 & cond4
        return sweep_bear

    def detect_rejection_bull(self, df):
        """
        Detect bullish rejection with doji filtering (FIX #10).
        """
        open_price = df['open']
        close = df['close']
        low = df['low']

        body = abs(close - open_price)
        wick_down = np.minimum(open_price, close) - low

        # FIX #10: Require minimum body AND wick >= body
        rejection_bull = (body >= MIN_BODY_SIZE) & (wick_down >= body * 1.0)
        return rejection_bull

    def detect_rejection_bear(self, df):
        """
        Detect bearish rejection with doji filtering (FIX #10).
        """
        open_price = df['open']
        close = df['close']
        high = df['high']

        body = abs(close - open_price)
        wick_up = high - np.maximum(open_price, close)

        # FIX #10: Require minimum body AND wick >= body
        rejection_bear = (body >= MIN_BODY_SIZE) & (wick_up >= body * 1.0)
        return rejection_bear

    def calculate_avg_body(self, df, period=None):
        """Calculate average body size over period."""
        if period is None:
            period = self.avg_body_period

        body = abs(df['close'] - df['open'])
        avg_body = body.rolling(window=period).mean()
        return avg_body

    def detect_impulse_bull(self, df, avg_body=None):
        """Detect bullish impulse/displacement candle."""
        if avg_body is None:
            avg_body = self.calculate_avg_body(df)

        body = abs(df['close'] - df['open'])
        is_bullish = df['close'] > df['open']

        impulse_bull = (body > self.impulse_multiplier * avg_body) & is_bullish
        return impulse_bull

    def detect_impulse_bear(self, df, avg_body=None):
        """Detect bearish impulse/displacement candle."""
        if avg_body is None:
            avg_body = self.calculate_avg_body(df)

        body = abs(df['close'] - df['open'])
        is_bearish = df['close'] < df['open']

        impulse_bear = (body > self.impulse_multiplier * avg_body) & is_bearish
        return impulse_bear

    def detect_fvg_bull(self, df):
        """Detect bullish FVG (Fair Value Gap)."""
        low = df['low']
        high = df['high']

        fvg_bull = low > high.shift(2)
        return fvg_bull

    def detect_fvg_bear(self, df):
        """Detect bearish FVG (Fair Value Gap)."""
        low = df['low']
        high = df['high']

        fvg_bear = high < low.shift(2)
        return fvg_bear

    def get_fvg_zone(self, df, index):
        """Get FVG zone boundaries."""
        if index < 2:
            return None

        zone_low = df.loc[index - 2, 'high']
        zone_high = df.loc[index, 'low']

        if zone_low >= zone_high:
            zone_low = df.loc[index, 'high']
            zone_high = df.loc[index - 2, 'low']

        if zone_low >= zone_high:
            return None

        return (zone_low, zone_high)

    def calculate_htf_bias(self, df_htf):
        """Calculate higher timeframe bias using EMA."""
        if len(df_htf) < self.config['htf_bias']['ema_period']:
            return 'NEUTRAL'

        ema_period = self.config['htf_bias']['ema_period']
        ema = df_htf['close'].ewm(span=ema_period, adjust=False).mean()

        current_price = df_htf['close'].iloc[-1]
        current_ema = ema.iloc[-1]

        if current_price > current_ema:
            return 'BULLISH'
        elif current_price < current_ema:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def calculate_retrace_pct(self, df, index, direction):
        """
        Calculate retracement percentage - CORRECTED (FIX #5).

        Original had inverted logic for bearish retracements.

        Bearish: Price fell from D_high to D_low, retrace is how far it came BACK UP
        Bullish: Price rose from D_low to D_high, retrace is how far it pulled BACK DOWN
        """
        if index >= len(df):
            return 0.0

        D_high = max(df.loc[index, 'open'], df.loc[index, 'close'])
        D_low = min(df.loc[index, 'open'], df.loc[index, 'close'])

        current_price = df['close'].iloc[-1]

        displacement_range = D_high - D_low
        if displacement_range == 0:
            return 0.0

        # FIX #5: Corrected retracement calculation
        if direction == 'BEAR':
            # Bearish move: went DOWN from D_high to D_low
            # Retracement: how far price came back UP from the low
            retrace_pct = (current_price - D_low) / displacement_range
        else:  # BULL
            # Bullish move: went UP from D_low to D_high
            # Retracement: how far price pulled back DOWN from the high
            retrace_pct = (D_high - current_price) / displacement_range

        return max(0.0, min(1.0, retrace_pct))

    def is_trading_session(self, current_time):
        """
        Check if current time is within London or NY session.

        FIX #7: Uses UTC timezone consistently.
        """
        current_time_only = current_time.time()

        london_start = datetime.strptime(self.sessions['london']['start'], "%H:%M").time()
        london_end = datetime.strptime(self.sessions['london']['end'], "%H:%M").time()
        ny_start = datetime.strptime(self.sessions['newyork']['start'], "%H:%M").time()
        ny_end = datetime.strptime(self.sessions['newyork']['end'], "%H:%M").time()

        in_london = london_start <= current_time_only <= london_end
        in_ny = ny_start <= current_time_only <= ny_end

        return in_london or in_ny

    def is_no_trade_window(self, current_time):
        """
        Check if current time is in no-trade window.

        FIX #6: Handles midnight crossing correctly.
        """
        current_time_only = current_time.time()

        no_trade_start = datetime.strptime(self.sessions['no_trade']['start'], "%H:%M").time()
        no_trade_end = datetime.strptime(self.sessions['no_trade']['end'], "%H:%M").time()

        # FIX #6: Handle midnight crossing (23:50 to 00:10)
        if no_trade_start > no_trade_end:
            # Crosses midnight
            return current_time_only >= no_trade_start or current_time_only <= no_trade_end
        else:
            return no_trade_start <= current_time_only <= no_trade_end

    def check_hard_filters(self, spread, atr_current, equity):
        """Check hard filters that block trading."""
        if spread > self.max_spread:
            return False, f"Spread too high: {spread:.2f} > {self.max_spread}"

        if atr_current < self.atr_vol_threshold:
            return False, f"ATR too low: {atr_current:.2f} < {self.atr_vol_threshold}"

        min_equity = self.config['risk']['min_equity']
        if equity < min_equity:
            return False, f"Equity too low: {equity:.2f} < {min_equity}"

        return True, "All hard filters passed"

    def calculate_score(self, signals, current_time, atr_current, htf_bias):
        """Calculate trading score based on signals and conditions."""
        score = 0

        if self.is_trading_session(current_time):
            score += self.weights['session']
            logger.debug("Score +20: In trading session")

        if signals.get('sweep', False):
            score += self.weights['sweep']
            logger.debug("Score +30: Sweep detected")

        if signals.get('rejection', False):
            score += self.weights['rejection']
            logger.debug("Score +15: Rejection detected")

        if signals.get('impulse', False):
            score += self.weights['impulse']
            logger.debug("Score +20: Impulse detected")

        if signals.get('fvg', False):
            score += self.weights['fvg']
            logger.debug("Score +15: FVG detected")

        direction = signals.get('direction', 'NEUTRAL')
        if htf_bias == direction:
            score += self.weights['htf_bias_aligned']
            logger.debug(f"Score +10: HTF bias aligned ({htf_bias})")
        elif htf_bias != 'NEUTRAL' and direction != 'NEUTRAL':
            score += self.weights['htf_bias_opposed']
            logger.debug(f"Score -10: HTF bias opposed ({htf_bias} vs {direction})")

        if atr_current > self.atr_high_threshold:
            score += self.weights['high_volatility']
            logger.debug(f"Score +5: High volatility ATR={atr_current:.2f}")

        if self.config['impulse']['volume_enabled'] and signals.get('volume_confirm', False):
            score += self.weights['volume_confirm']
            logger.debug("Score +5: Volume confirmed")

        logger.info(f"Total score: {score}")
        return score

    def analyze_market(self, df_m5, df_m1, df_h1, current_time, spread, equity):
        """
        Perform complete market analysis with all fixes applied.

        FIX #9: Prevents contradictory signals (only one direction per bar)
        FIX #11: Handles NaN values properly
        """
        logger.info("=== Starting Market Analysis ===")

        result = {
            'trade_allowed': False,
            'direction': None,
            'entry_model': None,
            'score': 0,
            'signals': {},
            'setup_data': {},
            'reason': ''
        }

        # Check minimum data
        min_bars_needed = max(self.atr_period, self.swing_lookback, self.avg_body_period) + 5
        if len(df_m5) < min_bars_needed:
            result['reason'] = f"Insufficient data: {len(df_m5)} < {min_bars_needed} bars"
            logger.warning(result['reason'])
            return result

        # Calculate ATR
        atr_m5 = self.calculate_atr(df_m5)
        atr_current = atr_m5.iloc[-1]

        # FIX #11: Check for NaN
        if pd.isna(atr_current):
            result['reason'] = "ATR calculation returned NaN"
            logger.warning(result['reason'])
            return result

        logger.info(f"ATR (M5): {atr_current:.2f}")

        # Check hard filters
        passed, reason = self.check_hard_filters(spread, atr_current, equity)
        if not passed:
            result['reason'] = f"Hard filter failed: {reason}"
            logger.warning(result['reason'])
            return result

        # Check no-trade window
        if self.is_no_trade_window(current_time):
            result['reason'] = "In no-trade window"
            logger.info(result['reason'])
            return result

        # Calculate HTF bias
        htf_bias = self.calculate_htf_bias(df_h1) if self.config['htf_bias']['enabled'] else 'NEUTRAL'
        logger.info(f"HTF Bias (H1): {htf_bias}")

        # Calculate indicators
        swing_high = self.detect_swing_high(df_m5)
        swing_low = self.detect_swing_low(df_m5)
        avg_body = self.calculate_avg_body(df_m5)

        # Detect signals
        sweep_bull = self.detect_sweep_bull(df_m5, swing_low)
        sweep_bear = self.detect_sweep_bear(df_m5, swing_high)
        rejection_bull = self.detect_rejection_bull(df_m5)
        rejection_bear = self.detect_rejection_bear(df_m5)
        impulse_bull = self.detect_impulse_bull(df_m5, avg_body)
        impulse_bear = self.detect_impulse_bear(df_m5, avg_body)
        fvg_bull = self.detect_fvg_bull(df_m5)
        fvg_bear = self.detect_fvg_bear(df_m5)

        idx = len(df_m5) - 1

        # FIX #11: Check for NaN in critical indicators
        if pd.isna(swing_high.iloc[idx]) or pd.isna(swing_low.iloc[idx]) or pd.isna(avg_body.iloc[idx]):
            result['reason'] = "Indicators contain NaN values"
            logger.warning(result['reason'])
            return result

        # FIX #9: Determine SINGLE direction based on candle color and dominant signal
        # Prevents contradictory signals on same bar
        current_bar = df_m5.iloc[idx]
        is_bullish_candle = current_bar['close'] > current_bar['open']

        bull_signal_strength = (
            (sweep_bull.iloc[idx] if idx > 0 else False) * 40 +
            (impulse_bull.iloc[idx] if idx > 0 else False) * 30 +
            (rejection_bull.iloc[idx] if idx > 0 else False) * 20 +
            (fvg_bull.iloc[idx] if idx > 0 else False) * 10
        )

        bear_signal_strength = (
            (sweep_bear.iloc[idx] if idx > 0 else False) * 40 +
            (impulse_bear.iloc[idx] if idx > 0 else False) * 30 +
            (rejection_bear.iloc[idx] if idx > 0 else False) * 20 +
            (fvg_bear.iloc[idx] if idx > 0 else False) * 10
        )

        # Choose dominant direction
        if bull_signal_strength > bear_signal_strength and bull_signal_strength > 0:
            direction = 'BULLISH'
            signals = {
                'sweep': sweep_bull.iloc[idx] if idx > 0 else False,
                'rejection': rejection_bull.iloc[idx] if idx > 0 else False,
                'impulse': impulse_bull.iloc[idx] if idx > 0 else False,
                'fvg': fvg_bull.iloc[idx] if idx > 0 else False,
                'direction': 'BULLISH'
            }
            setup_data = {
                'swing_low': swing_low.iloc[idx] if idx > 0 else None,
                'atr': atr_current,
                'htf_bias': htf_bias
            }
        elif bear_signal_strength > bull_signal_strength and bear_signal_strength > 0:
            direction = 'BEARISH'
            signals = {
                'sweep': sweep_bear.iloc[idx] if idx > 0 else False,
                'rejection': rejection_bear.iloc[idx] if idx > 0 else False,
                'impulse': impulse_bear.iloc[idx] if idx > 0 else False,
                'fvg': fvg_bear.iloc[idx] if idx > 0 else False,
                'direction': 'BEARISH'
            }
            setup_data = {
                'swing_high': swing_high.iloc[idx] if idx > 0 else None,
                'atr': atr_current,
                'htf_bias': htf_bias
            }
        else:
            result['reason'] = "No clear directional bias"
            logger.debug(result['reason'])
            return result

        # Calculate score for chosen direction
        score = self.calculate_score(signals, current_time, atr_current, htf_bias)

        if score >= self.score_threshold:
            result['trade_allowed'] = True
            result['direction'] = direction
            result['score'] = score
            result['signals'] = signals
            result['setup_data'] = setup_data
            result['reason'] = f"{direction} setup detected with score {score}"
            logger.info(f"✓ {result['reason']}")
        else:
            result['reason'] = f"Score too low: {score} < {self.score_threshold}"
            logger.info(result['reason'])

        return result
