"""
analyst.py - Signal Generation, Calculations, Scoring, and Filtering

This module contains all analytical functions for the XAUUSD breakout trading system:
- ATR calculation
- Swing high/low detection
- Liquidity sweep detection
- Rejection wick analysis
- Impulse/displacement confirmation
- FVG (Fair Value Gap) detection
- HTF bias calculation
- Scoring engine
- Session filtering
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


class MarketAnalyst:
    """Handles all market analysis, signal generation, and scoring."""

    def __init__(self, config):
        """
        Initialize the analyst with configuration.

        Args:
            config: Dictionary containing all configuration parameters
        """
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

    # ========================================================================
    # ATR CALCULATION
    # ========================================================================

    def calculate_atr(self, df, period=None):
        """
        Calculate Average True Range.

        Args:
            df: DataFrame with OHLC data
            period: ATR period (default from config)

        Returns:
            Series with ATR values
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
        atr = tr.rolling(window=period).mean()

        return atr

    # ========================================================================
    # SWING HIGH/LOW DETECTION
    # ========================================================================

    def detect_swing_high(self, df, lookback=None):
        """
        Detect swing highs over lookback period.

        Args:
            df: DataFrame with high prices
            lookback: Number of bars to look back

        Returns:
            Series with swing high values
        """
        if lookback is None:
            lookback = self.swing_lookback

        swing_high = df['high'].rolling(window=lookback, min_periods=1).max().shift(1)
        return swing_high

    def detect_swing_low(self, df, lookback=None):
        """
        Detect swing lows over lookback period.

        Args:
            df: DataFrame with low prices
            lookback: Number of bars to look back

        Returns:
            Series with swing low values
        """
        if lookback is None:
            lookback = self.swing_lookback

        swing_low = df['low'].rolling(window=lookback, min_periods=1).min().shift(1)
        return swing_low

    # ========================================================================
    # LIQUIDITY SWEEP DETECTION
    # ========================================================================

    def detect_sweep_bull(self, df, swing_low):
        """
        Detect bullish liquidity sweep (sweep below swing low with rejection).

        Conditions:
        1. Low breaks below swing low
        2. Close is above swing low (rejection)
        3. Rejection wick is larger than body

        Args:
            df: DataFrame with OHLC data
            swing_low: Series with swing low values

        Returns:
            Series with boolean values
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

        sweep_bull = cond1 & cond2 & cond3
        return sweep_bull

    def detect_sweep_bear(self, df, swing_high):
        """
        Detect bearish liquidity sweep (sweep above swing high with rejection).

        Conditions:
        1. High breaks above swing high
        2. Close is below swing high (rejection)
        3. Rejection wick is larger than body

        Args:
            df: DataFrame with OHLC data
            swing_high: Series with swing high values

        Returns:
            Series with boolean values
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

        sweep_bear = cond1 & cond2 & cond3
        return sweep_bear

    # ========================================================================
    # REJECTION WICK ANALYSIS
    # ========================================================================

    def detect_rejection_bull(self, df):
        """
        Detect bullish rejection (strong lower wick >= body size).

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with boolean values
        """
        open_price = df['open']
        close = df['close']
        low = df['low']

        body = abs(close - open_price)
        wick_down = np.minimum(open_price, close) - low

        rejection_bull = wick_down >= body * 1.0
        return rejection_bull

    def detect_rejection_bear(self, df):
        """
        Detect bearish rejection (strong upper wick >= body size).

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with boolean values
        """
        open_price = df['open']
        close = df['close']
        high = df['high']

        body = abs(close - open_price)
        wick_up = high - np.maximum(open_price, close)

        rejection_bear = wick_up >= body * 1.0
        return rejection_bear

    # ========================================================================
    # IMPULSE / DISPLACEMENT CONFIRMATION
    # ========================================================================

    def calculate_avg_body(self, df, period=None):
        """
        Calculate average body size over period.

        Args:
            df: DataFrame with OHLC data
            period: Lookback period

        Returns:
            Series with average body values
        """
        if period is None:
            period = self.avg_body_period

        body = abs(df['close'] - df['open'])
        avg_body = body.rolling(window=period).mean()
        return avg_body

    def detect_impulse_bull(self, df, avg_body=None):
        """
        Detect bullish impulse/displacement candle.

        Conditions:
        - Body size > 1.5 * average body
        - Candle is bullish (close > open)

        Args:
            df: DataFrame with OHLC data
            avg_body: Pre-calculated average body (optional)

        Returns:
            Series with boolean values
        """
        if avg_body is None:
            avg_body = self.calculate_avg_body(df)

        body = abs(df['close'] - df['open'])
        is_bullish = df['close'] > df['open']

        impulse_bull = (body > self.impulse_multiplier * avg_body) & is_bullish
        return impulse_bull

    def detect_impulse_bear(self, df, avg_body=None):
        """
        Detect bearish impulse/displacement candle.

        Conditions:
        - Body size > 1.5 * average body
        - Candle is bearish (close < open)

        Args:
            df: DataFrame with OHLC data
            avg_body: Pre-calculated average body (optional)

        Returns:
            Series with boolean values
        """
        if avg_body is None:
            avg_body = self.calculate_avg_body(df)

        body = abs(df['close'] - df['open'])
        is_bearish = df['close'] < df['open']

        impulse_bear = (body > self.impulse_multiplier * avg_body) & is_bearish
        return impulse_bear

    # ========================================================================
    # FVG (FAIR VALUE GAP) DETECTION
    # ========================================================================

    def detect_fvg_bull(self, df):
        """
        Detect bullish FVG (Fair Value Gap).

        Condition: low[i] > high[i-2] (gap between candles)

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with boolean values
        """
        low = df['low']
        high = df['high']

        fvg_bull = low > high.shift(2)
        return fvg_bull

    def detect_fvg_bear(self, df):
        """
        Detect bearish FVG (Fair Value Gap).

        Condition: high[i] < low[i-2] (gap between candles)

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with boolean values
        """
        low = df['low']
        high = df['high']

        fvg_bear = high < low.shift(2)
        return fvg_bear

    def get_fvg_zone(self, df, index):
        """
        Get FVG zone boundaries.

        Args:
            df: DataFrame with OHLC data
            index: Index where FVG was detected

        Returns:
            Tuple (zone_low, zone_high) or None
        """
        if index < 2:
            return None

        # For bullish FVG: zone is between high[i-2] and low[i]
        # For bearish FVG: zone is between low[i-2] and high[i]

        zone_low = df.loc[index - 2, 'high']  # For bullish
        zone_high = df.loc[index, 'low']

        if zone_low >= zone_high:
            # Try bearish FVG
            zone_low = df.loc[index, 'high']
            zone_high = df.loc[index - 2, 'low']

        if zone_low >= zone_high:
            return None

        return (zone_low, zone_high)

    # ========================================================================
    # HTF BIAS CALCULATION
    # ========================================================================

    def calculate_htf_bias(self, df_htf):
        """
        Calculate higher timeframe bias using EMA.

        Args:
            df_htf: DataFrame with H1 OHLC data

        Returns:
            String: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
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

    # ========================================================================
    # RETRACEMENT CALCULATION
    # ========================================================================

    def calculate_retrace_pct(self, df, index, direction):
        """
        Calculate retracement percentage after displacement.

        Args:
            df: DataFrame with OHLC data
            index: Index of displacement candle
            direction: 'BEAR' or 'BULL'

        Returns:
            Float: Retracement percentage (0.0 to 1.0)
        """
        if index >= len(df):
            return 0.0

        D_high = max(df.loc[index, 'open'], df.loc[index, 'close'])
        D_low = min(df.loc[index, 'open'], df.loc[index, 'close'])

        current_price = df['close'].iloc[-1]

        displacement_range = D_high - D_low
        if displacement_range == 0:
            return 0.0

        if direction == 'BEAR':
            retrace_pct = (D_high - current_price) / displacement_range
        else:  # BULL
            retrace_pct = (current_price - D_low) / displacement_range

        return max(0.0, min(1.0, retrace_pct))

    # ========================================================================
    # SESSION FILTERING
    # ========================================================================

    def is_trading_session(self, current_time):
        """
        Check if current time is within London or NY session.

        Args:
            current_time: datetime object (UTC)

        Returns:
            Boolean
        """
        current_time_only = current_time.time()

        # Parse session times
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

        Args:
            current_time: datetime object (UTC)

        Returns:
            Boolean
        """
        current_time_only = current_time.time()

        no_trade_start = datetime.strptime(self.sessions['no_trade']['start'], "%H:%M").time()
        no_trade_end = datetime.strptime(self.sessions['no_trade']['end'], "%H:%M").time()

        return no_trade_start <= current_time_only <= no_trade_end

    # ========================================================================
    # HARD FILTERS (KILLERS)
    # ========================================================================

    def check_hard_filters(self, spread, atr_current, equity):
        """
        Check hard filters that block trading.

        Args:
            spread: Current spread
            atr_current: Current ATR value
            equity: Account equity

        Returns:
            Tuple (passed: bool, reason: str)
        """
        if spread > self.max_spread:
            return False, f"Spread too high: {spread:.2f} > {self.max_spread}"

        if atr_current < self.atr_vol_threshold:
            return False, f"ATR too low: {atr_current:.2f} < {self.atr_vol_threshold}"

        min_equity = self.config['risk']['min_equity']
        if equity < min_equity:
            return False, f"Equity too low: {equity:.2f} < {min_equity}"

        return True, "All hard filters passed"

    # ========================================================================
    # SCORING ENGINE
    # ========================================================================

    def calculate_score(self, signals, current_time, atr_current, htf_bias):
        """
        Calculate trading score based on signals and conditions.

        Args:
            signals: Dictionary containing detected signals
            current_time: Current datetime (UTC)
            atr_current: Current ATR value
            htf_bias: HTF bias ('BULLISH', 'BEARISH', 'NEUTRAL')

        Returns:
            Integer score
        """
        score = 0

        # Session filter (+20)
        if self.is_trading_session(current_time):
            score += self.weights['session']
            logger.debug("Score +20: In trading session")

        # Sweep detection (+30)
        if signals.get('sweep', False):
            score += self.weights['sweep']
            logger.debug("Score +30: Sweep detected")

        # Rejection (+15)
        if signals.get('rejection', False):
            score += self.weights['rejection']
            logger.debug("Score +15: Rejection detected")

        # Impulse (+20)
        if signals.get('impulse', False):
            score += self.weights['impulse']
            logger.debug("Score +20: Impulse detected")

        # FVG exists (+15)
        if signals.get('fvg', False):
            score += self.weights['fvg']
            logger.debug("Score +15: FVG detected")

        # HTF bias alignment (+10 / -10)
        direction = signals.get('direction', 'NEUTRAL')
        if htf_bias == direction:
            score += self.weights['htf_bias_aligned']
            logger.debug(f"Score +10: HTF bias aligned ({htf_bias})")
        elif htf_bias != 'NEUTRAL' and direction != 'NEUTRAL':
            score += self.weights['htf_bias_opposed']
            logger.debug(f"Score -10: HTF bias opposed ({htf_bias} vs {direction})")

        # High volatility (+5)
        if atr_current > self.atr_high_threshold:
            score += self.weights['high_volatility']
            logger.debug(f"Score +5: High volatility ATR={atr_current:.2f}")

        # Volume confirmation (if enabled)
        if self.config['impulse']['volume_enabled'] and signals.get('volume_confirm', False):
            score += self.weights['volume_confirm']
            logger.debug("Score +5: Volume confirmed")

        logger.info(f"Total score: {score}")
        return score

    # ========================================================================
    # COMPLETE ANALYSIS
    # ========================================================================

    def analyze_market(self, df_m5, df_m1, df_h1, current_time, spread, equity):
        """
        Perform complete market analysis.

        Args:
            df_m5: M5 DataFrame with OHLC data
            df_m1: M1 DataFrame with OHLC data (for confirmation)
            df_h1: H1 DataFrame with OHLC data (for bias)
            current_time: Current datetime (UTC)
            spread: Current spread
            equity: Account equity

        Returns:
            Dictionary with analysis results
        """
        logger.info("=== Starting Market Analysis ===")

        # Initialize result
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
        if len(df_m5) < max(self.atr_period, self.swing_lookback, self.avg_body_period):
            result['reason'] = "Insufficient data for analysis"
            logger.warning(result['reason'])
            return result

        # Calculate ATR
        atr_m5 = self.calculate_atr(df_m5)
        atr_current = atr_m5.iloc[-1]
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

        # Get latest signals
        idx = len(df_m5) - 1

        # Determine direction and collect signals
        signals_bull = {
            'sweep': sweep_bull.iloc[idx] if idx > 0 else False,
            'rejection': rejection_bull.iloc[idx] if idx > 0 else False,
            'impulse': impulse_bull.iloc[idx] if idx > 0 else False,
            'fvg': fvg_bull.iloc[idx] if idx > 0 else False,
            'direction': 'BULLISH'
        }

        signals_bear = {
            'sweep': sweep_bear.iloc[idx] if idx > 0 else False,
            'rejection': rejection_bear.iloc[idx] if idx > 0 else False,
            'impulse': impulse_bear.iloc[idx] if idx > 0 else False,
            'fvg': fvg_bear.iloc[idx] if idx > 0 else False,
            'direction': 'BEARISH'
        }

        # Calculate scores for both directions
        score_bull = self.calculate_score(signals_bull, current_time, atr_current, htf_bias)
        score_bear = self.calculate_score(signals_bear, current_time, atr_current, htf_bias)

        logger.info(f"Bullish score: {score_bull}, Bearish score: {score_bear}")

        # Choose highest scoring direction
        if score_bull >= self.score_threshold and score_bull > score_bear:
            result['direction'] = 'BULLISH'
            result['score'] = score_bull
            result['signals'] = signals_bull
            result['setup_data'] = {
                'swing_low': swing_low.iloc[idx] if idx > 0 else None,
                'atr': atr_current,
                'htf_bias': htf_bias
            }
        elif score_bear >= self.score_threshold and score_bear >= score_bull:
            result['direction'] = 'BEARISH'
            result['score'] = score_bear
            result['signals'] = signals_bear
            result['setup_data'] = {
                'swing_high': swing_high.iloc[idx] if idx > 0 else None,
                'atr': atr_current,
                'htf_bias': htf_bias
            }
        else:
            result['reason'] = f"Score too low (Bull: {score_bull}, Bear: {score_bear}, Threshold: {self.score_threshold})"
            logger.info(result['reason'])
            return result

        # If we have a valid direction, allow trade
        if result['direction']:
            result['trade_allowed'] = True
            result['reason'] = f"{result['direction']} setup detected with score {result['score']}"
            logger.info(f"✓ {result['reason']}")

        return result
