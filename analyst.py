"""Market analysis, signal detection, and scoring for XAUUSD breakout strategy."""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

MIN_BODY_SIZE = 0.50  # Minimum body size in dollars to avoid doji false signals


class MarketAnalyst:
    def __init__(self, config):
        self.config = config
        self.atr_period = config['atr']['period']
        self.swing_lookback = config['lookback']['swing']
        self.avg_body_period = config['lookback']['avg_body']
        self.impulse_mult = config['impulse']['body_multiplier']
        self.atr_threshold = config['atr']['volatility_threshold']
        self.atr_high = config['atr']['high_threshold']
        self.max_spread = config['filters']['max_spread']
        self.sessions = config['sessions']
        self.session_buffers = config.get('session_buffers', {'enabled': False})
        self.weights = config['scoring']['weights']
        self.score_threshold = config['scoring']['threshold_trade']
        self.shallow_min = config['retracement']['shallow_min']
        self.shallow_max = config['retracement']['shallow_max']

    def calculate_atr(self, df, period=None):
        """Calculate ATR using Wilder's smoothing (EMA with alpha=1/period)."""
        period = period or self.atr_period
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def detect_swing_high(self, df, lookback=None):
        """Detect swing highs over lookback period."""
        lookback = lookback or self.swing_lookback
        return df['high'].rolling(lookback, min_periods=1).max().shift(1)

    def detect_swing_low(self, df, lookback=None):
        """Detect swing lows over lookback period."""
        lookback = lookback or self.swing_lookback
        return df['low'].rolling(lookback, min_periods=1).min().shift(1)

    def detect_sweep_bull(self, df, swing_low):
        """Detect bullish liquidity sweep: low breaks swing_low, close recovers above."""
        body = abs(df['close'] - df['open'])
        wick_down = np.minimum(df['open'], df['close']) - df['low']
        return (df['low'] < swing_low) & (df['close'] > swing_low) & \
               (wick_down > body) & (body >= MIN_BODY_SIZE)

    def detect_sweep_bear(self, df, swing_high):
        """Detect bearish liquidity sweep: high breaks swing_high, close recovers below."""
        body = abs(df['close'] - df['open'])
        wick_up = df['high'] - np.maximum(df['open'], df['close'])
        return (df['high'] > swing_high) & (df['close'] < swing_high) & \
               (wick_up > body) & (body >= MIN_BODY_SIZE)

    def detect_rejection_bull(self, df):
        """Detect bullish rejection wick."""
        body = abs(df['close'] - df['open'])
        wick_down = np.minimum(df['open'], df['close']) - df['low']
        return (body >= MIN_BODY_SIZE) & (wick_down >= body)

    def detect_rejection_bear(self, df):
        """Detect bearish rejection wick."""
        body = abs(df['close'] - df['open'])
        wick_up = df['high'] - np.maximum(df['open'], df['close'])
        return (body >= MIN_BODY_SIZE) & (wick_up >= body)

    def calculate_avg_body(self, df, period=None):
        """Calculate average body size over period."""
        period = period or self.avg_body_period
        return abs(df['close'] - df['open']).rolling(period).mean()

    def detect_impulse_bull(self, df, avg_body=None):
        """Detect bullish impulse candle (body > multiplier * avg_body)."""
        avg_body = avg_body if avg_body is not None else self.calculate_avg_body(df)
        body = abs(df['close'] - df['open'])
        return (body > self.impulse_mult * avg_body) & (df['close'] > df['open'])

    def detect_impulse_bear(self, df, avg_body=None):
        """Detect bearish impulse candle."""
        avg_body = avg_body if avg_body is not None else self.calculate_avg_body(df)
        body = abs(df['close'] - df['open'])
        return (body > self.impulse_mult * avg_body) & (df['close'] < df['open'])

    def detect_fvg_bull(self, df):
        """Detect bullish FVG: low[i] > high[i-2]."""
        return df['low'] > df['high'].shift(2)

    def detect_fvg_bear(self, df):
        """Detect bearish FVG: high[i] < low[i-2]."""
        return df['high'] < df['low'].shift(2)

    def calculate_htf_bias(self, df_htf):
        """Calculate higher timeframe bias using EMA."""
        if len(df_htf) < self.config['htf_bias']['ema_period']:
            return 'NEUTRAL'
        ema = df_htf['close'].ewm(span=self.config['htf_bias']['ema_period'], adjust=False).mean()
        price = df_htf['close'].iloc[-1]
        return 'BULLISH' if price > ema.iloc[-1] else 'BEARISH' if price < ema.iloc[-1] else 'NEUTRAL'

    def is_trading_session(self, current_time):
        """Check if within London or NY session (UTC)."""
        t = current_time.time()
        london = (datetime.strptime(self.sessions['london']['start'], "%H:%M").time(),
                  datetime.strptime(self.sessions['london']['end'], "%H:%M").time())
        ny = (datetime.strptime(self.sessions['newyork']['start'], "%H:%M").time(),
              datetime.strptime(self.sessions['newyork']['end'], "%H:%M").time())
        return (london[0] <= t <= london[1]) or (ny[0] <= t <= ny[1])

    def is_no_trade_window(self, current_time):
        """Check if in no-trade window or session buffer zone."""
        t = current_time.time()

        # Check original no-trade window (23:50 - 00:10)
        start = datetime.strptime(self.sessions['no_trade']['start'], "%H:%M").time()
        end = datetime.strptime(self.sessions['no_trade']['end'], "%H:%M").time()
        in_no_trade = (t >= start or t <= end) if start > end else (start <= t <= end)

        if in_no_trade:
            return True

        # Check session buffer zones (if enabled)
        if self.session_buffers.get('enabled', False):
            buffer_hit, reason = self._is_in_buffer_zone(current_time)
            if buffer_hit:
                logger.info(f"No-trade: {reason}")
                return True

        return False

    def _is_in_buffer_zone(self, current_time):
        """Check if current time is within any session transition buffer zone."""
        from datetime import timedelta

        t = current_time.time()
        current_minutes = t.hour * 60 + t.minute

        # Check each configured buffer zone
        buffer_zones = [
            ('london_open', self.session_buffers.get('london_open')),
            ('london_close', self.session_buffers.get('london_close')),
            ('newyork_open', self.session_buffers.get('newyork_open')),
            ('newyork_close', self.session_buffers.get('newyork_close'))
        ]

        for zone_name, zone_config in buffer_zones:
            if not zone_config:
                continue

            # Parse transition time
            transition_time = datetime.strptime(zone_config['time'], "%H:%M").time()
            transition_minutes = transition_time.hour * 60 + transition_time.minute

            # Get buffer in minutes
            buffer_mins = zone_config.get('buffer_minutes', 0)

            # Calculate buffer window
            buffer_start = transition_minutes - buffer_mins
            buffer_end = transition_minutes + buffer_mins

            # Check if current time is within buffer (handle negative values for day wrap)
            if buffer_start < 0:
                # Wraps to previous day (e.g., 06:30 - 30 = -30 = 23:30 previous day)
                buffer_start += 1440  # Add 24 hours in minutes
                if current_minutes >= buffer_start or current_minutes <= buffer_end:
                    return True, f"{zone_name} buffer zone ({zone_config.get('reason', 'transition volatility')})"
            elif buffer_end >= 1440:
                # Wraps to next day
                buffer_end -= 1440
                if current_minutes >= buffer_start or current_minutes <= buffer_end:
                    return True, f"{zone_name} buffer zone ({zone_config.get('reason', 'transition volatility')})"
            else:
                # Normal case - no day wrap
                if buffer_start <= current_minutes <= buffer_end:
                    return True, f"{zone_name} buffer zone ({zone_config.get('reason', 'transition volatility')})"

        return False, ""

    def check_hard_filters(self, spread, atr_current, equity):
        """Check blocking filters."""
        if spread > self.max_spread:
            return False, f"Spread {spread:.2f} > {self.max_spread}"
        if atr_current < self.atr_threshold:
            return False, f"ATR {atr_current:.2f} < {self.atr_threshold}"
        if equity < self.config['risk']['min_equity']:
            return False, f"Equity {equity:.2f} too low"
        return True, "Pass"

    def calculate_score(self, signals, current_time, atr_current, htf_bias):
        """Calculate setup score from signals and conditions."""
        score = 0
        if self.is_trading_session(current_time):
            score += self.weights['session']
        if signals.get('sweep'):
            score += self.weights['sweep']
        if signals.get('rejection'):
            score += self.weights['rejection']
        if signals.get('impulse'):
            score += self.weights['impulse']
        if signals.get('fvg'):
            score += self.weights['fvg']
        if htf_bias == signals.get('direction'):
            score += self.weights['htf_bias_aligned']
        elif htf_bias != 'NEUTRAL' and signals.get('direction') != 'NEUTRAL':
            score += self.weights['htf_bias_opposed']
        if atr_current > self.atr_high:
            score += self.weights['high_volatility']
        return score

    def analyze_market(self, df_m5, df_m1, df_h1, current_time, spread, equity):
        """Perform complete market analysis and return trade signal."""
        result = {
            'trade_allowed': False,
            'direction': None,
            'entry_model': None,
            'score': 0,
            'signals': {},
            'setup_data': {},
            'reason': ''
        }

        min_bars = max(self.atr_period, self.swing_lookback, self.avg_body_period) + 5
        if len(df_m5) < min_bars:
            result['reason'] = f"Insufficient data: {len(df_m5)}/{min_bars} bars"
            return result

        atr_m5 = self.calculate_atr(df_m5)
        atr_current = atr_m5.iloc[-1]

        if pd.isna(atr_current):
            result['reason'] = "ATR is NaN"
            return result

        passed, reason = self.check_hard_filters(spread, atr_current, equity)
        if not passed:
            result['reason'] = reason
            return result

        if self.is_no_trade_window(current_time):
            result['reason'] = "No-trade window"
            return result

        htf_bias = self.calculate_htf_bias(df_h1) if self.config['htf_bias']['enabled'] else 'NEUTRAL'

        swing_high = self.detect_swing_high(df_m5)
        swing_low = self.detect_swing_low(df_m5)
        avg_body = self.calculate_avg_body(df_m5)

        sweep_bull = self.detect_sweep_bull(df_m5, swing_low)
        sweep_bear = self.detect_sweep_bear(df_m5, swing_high)
        rejection_bull = self.detect_rejection_bull(df_m5)
        rejection_bear = self.detect_rejection_bear(df_m5)
        impulse_bull = self.detect_impulse_bull(df_m5, avg_body)
        impulse_bear = self.detect_impulse_bear(df_m5, avg_body)
        fvg_bull = self.detect_fvg_bull(df_m5)
        fvg_bear = self.detect_fvg_bear(df_m5)

        idx = len(df_m5) - 1

        if pd.isna(swing_high.iloc[idx]) or pd.isna(swing_low.iloc[idx]):
            result['reason'] = "Indicators NaN"
            return result

        # Determine dominant direction
        bull_strength = sum([
            sweep_bull.iloc[idx] * 40,
            impulse_bull.iloc[idx] * 30,
            rejection_bull.iloc[idx] * 20,
            fvg_bull.iloc[idx] * 10
        ])

        bear_strength = sum([
            sweep_bear.iloc[idx] * 40,
            impulse_bear.iloc[idx] * 30,
            rejection_bear.iloc[idx] * 20,
            fvg_bear.iloc[idx] * 10
        ])

        if bull_strength > bear_strength and bull_strength > 0:
            direction = 'BULLISH'
            signals = {
                'sweep': sweep_bull.iloc[idx],
                'rejection': rejection_bull.iloc[idx],
                'impulse': impulse_bull.iloc[idx],
                'fvg': fvg_bull.iloc[idx],
                'direction': 'BULLISH'
            }
            setup_data = {
                'swing_low': swing_low.iloc[idx],
                'atr': atr_current,
                'htf_bias': htf_bias
            }
        elif bear_strength > bull_strength and bear_strength > 0:
            direction = 'BEARISH'
            signals = {
                'sweep': sweep_bear.iloc[idx],
                'rejection': rejection_bear.iloc[idx],
                'impulse': impulse_bear.iloc[idx],
                'fvg': fvg_bear.iloc[idx],
                'direction': 'BEARISH'
            }
            setup_data = {
                'swing_high': swing_high.iloc[idx],
                'atr': atr_current,
                'htf_bias': htf_bias
            }
        else:
            result['reason'] = "No directional bias"
            return result

        score = self.calculate_score(signals, current_time, atr_current, htf_bias)

        if score >= self.score_threshold:
            result['trade_allowed'] = True
            result['direction'] = direction
            result['score'] = score
            result['signals'] = signals
            result['setup_data'] = setup_data
            result['reason'] = f"{direction} setup, score={score}"
            logger.info(f"Trade signal: {result['reason']}")
        else:
            result['reason'] = f"Score {score} < {self.score_threshold}"

        return result
