# GOLDBOT - XAUUSD Breakout Trading System

Enterprise-grade algorithmic trading bot for XAUUSD (Gold) breakout strategies using MetaTrader5.

## Features

- **Three Entry Models**: Momentum Break, Shallow Retest, and FVG Sniper
- **Advanced Signal Detection**: Liquidity sweeps, rejection wicks, displacement, and FVG detection
- **Intelligent Scoring System**: Multi-factor scoring with hard and soft filters
- **Robust Risk Management**: Position sizing, daily loss limits, and circuit breakers
- **Smart Trade Management**: Partial TPs, breakeven, and dynamic trailing stops
- **Session Filtering**: London and NY session windows
- **HTF Bias Confirmation**: H1 EMA trend alignment
- **Telegram Notifications**: Real-time trade alerts and error reporting
- **24/7 Operation**: Continuous market monitoring
- **Backtest Mode**: Historical strategy testing

## Project Structure

```
goldbot/
├── main.py                  # Entry point with live/backtest modes
├── analyst.py               # Signal generation and scoring engine
├── trader_and_manager.py    # Execution and trade management
├── bot_config.yaml          # Configuration file
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- MetaTrader5 terminal installed
- Active MT5 trading account (Exness recommended)
- Telegram Bot Token (optional, for notifications)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sharafatnazmul1/goldbot.git
   cd goldbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the bot**:
   Edit `bot_config.yaml` and update:
   - MT5 credentials (login, password, server)
   - Telegram bot token and chat ID
   - Risk parameters
   - Trading preferences

## Configuration

### Essential Settings

Edit `bot_config.yaml`:

```yaml
# MT5 Connection
mt5:
  login: YOUR_ACCOUNT_NUMBER
  password: "YOUR_PASSWORD"
  server: "Exness-MT5Real"

# Telegram
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"

# Risk Management
risk:
  risk_per_trade: 0.005      # 0.5% per trade
  max_position_risk: 2.0     # 2% total exposure
  daily_loss_limit: 0.03     # 3% daily stop
```

### Getting Telegram Credentials

1. Create a bot with [@BotFather](https://t.me/botfather)
2. Get your bot token
3. Get your chat ID from [@userinfobot](https://t.me/userinfobot)

## Usage

### Live Trading

```bash
python main.py --live
```

### Backtesting

```bash
python main.py --backtest
```

### Custom Config File

```bash
python main.py --live --config my_config.yaml
```

## Strategy Overview

### Entry Models

#### A) Momentum Break Entry (MBO/E)
- Captures fast moves with no retracement
- Enters on break of displacement low/high
- Execution: Market order (IOC)

#### B) Shallow Retest Entry (SRE)
- Captures 10-40% retracements
- Entry at 30% retrace of displacement
- Execution: Limit order

#### C) FVG Sniper Entry (SFE)
- Targets Fair Value Gap fills
- Entry at FVG midpoint
- Execution: Limit order

### Signal Components

1. **Liquidity Sweep**: Price wicks through swing high/low
2. **Rejection**: Strong wick showing rejection
3. **Impulse**: Body > 1.5x average body size
4. **FVG**: 2-bar imbalance gap
5. **HTF Bias**: H1 EMA20 trend direction

### Scoring System

Signals are scored based on:
- Session timing: +20
- Sweep detected: +30
- Rejection: +15
- Impulse: +20
- FVG present: +15
- HTF bias aligned: +10
- High volatility: +5

Minimum score: 60 (configurable)

### Risk Management

- **Position Sizing**: Dynamic based on equity and SL distance
- **Stop Loss**: Sweep wick + buffer (min $2.50)
- **Take Profit**: TP1 = 1.5R, TP2 = 3.0R
- **Partial Close**: 50% at TP1
- **Breakeven**: SL to entry at 1.0R
- **Trailing**: Activates at 1.2R using 0.6 × ATR

### Filters

**Hard Filters** (blockers):
- Spread > $0.18
- ATR < $0.80
- Equity below minimum
- Daily loss limit reached

**Session Filter**:
- London: 07:00 - 10:30 UTC
- New York: 12:30 - 15:30 UTC
- No-trade: 23:50 - 00:10 UTC

## Trade Management

1. **Entry**: Based on selected entry model
2. **TP1 Hit**: Close 50% of position
3. **1.0R Profit**: Move SL to breakeven
4. **1.2R Profit**: Activate trailing stop
5. **Trailing**: Update SL using 0.6 × ATR
6. **TP2**: Close remaining position

## Logging

All activity is logged to `goldbot.log` with timestamps:
- Market analysis
- Trade executions
- Risk checks
- Errors and warnings

Log level configurable in `bot_config.yaml`:
- DEBUG: Detailed analysis
- INFO: General operations (default)
- WARNING: Issues and alerts
- ERROR: Critical problems

## Safety Features

- **Idempotency**: Prevents duplicate orders
- **SL/TP Verification**: Confirms placement after order
- **Connection Monitoring**: Auto-reconnect on disconnect
- **Circuit Breaker**: Stops trading on daily loss limit
- **Max Concurrent Positions**: Limits simultaneous trades
- **Slippage Control**: Cancels orders with excessive slippage

## Performance Monitoring

Monitor performance via:
- Telegram notifications
- Log file analysis
- MT5 account history
- Backtest reports (when implemented)

## Troubleshooting

### MT5 Connection Failed

- Verify MT5 terminal is running
- Check credentials in config
- Ensure XAUUSD symbol is available
- Check firewall/antivirus settings

### No Trades Executing

- Check hard filters (spread, ATR)
- Verify session times (UTC)
- Check scoring threshold
- Review logs for rejection reasons

### Telegram Not Working

- Verify bot token and chat ID
- Test with [@BotFather](https://t.me/botfather)
- Check internet connection
- Set `enabled: false` to disable

## Disclaimer

**IMPORTANT**: This trading bot is for educational and personal use only.

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Never risk more than you can afford to lose
- Test thoroughly in demo account before live trading
- The authors are not responsible for any financial losses

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: https://github.com/sharafatnazmul1/goldbot/issues
- Review logs for error details
- Check configuration settings

## Version

v1.0.0 - Initial Release

Built following professional algo trading standards with modular architecture, comprehensive risk management, and production-grade error handling.
