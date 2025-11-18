"""
state_manager.py - Persistent State Management (FIX #24)

Handles:
- Position state persistence across restarts
- Trade history logging
- Performance metrics tracking
- Recovery from crashes
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class StateManager:
    """Manages persistent state using SQLite."""

    def __init__(self, db_path='goldbot_state.db'):
        """
        Initialize state manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
        logger.info(f"StateManager initialized with database: {db_path}")

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Active positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_positions (
                ticket INTEGER PRIMARY KEY,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl_price REAL NOT NULL,
                tp1_price REAL NOT NULL,
                tp2_price REAL NOT NULL,
                lot_size REAL NOT NULL,
                entry_model TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                sl_distance REAL NOT NULL,
                partial_closed INTEGER DEFAULT 0,
                breakeven_moved INTEGER DEFAULT 0,
                trailing_active INTEGER DEFAULT 0,
                state_json TEXT
            )
        ''')

        # Trade history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                lot_size REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                pnl REAL,
                r_multiple REAL,
                entry_model TEXT,
                exit_reason TEXT,
                score INTEGER,
                metadata_json TEXT
            )
        ''')

        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                balance REAL NOT NULL,
                peak_equity REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                daily_pnl REAL,
                weekly_pnl REAL,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL
            )
        ''')

        # System state table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        self.conn.commit()
        logger.info("Database tables initialized")

    def save_position(self, position_info):
        """Save or update position state."""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO active_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position_info['ticket'],
            position_info['direction'],
            position_info['entry_price'],
            position_info['sl_price'],
            position_info['tp1_price'],
            position_info['tp2_price'],
            position_info['lot_size'],
            position_info['entry_model'],
            position_info['entry_time'].isoformat(),
            position_info['sl_distance'],
            int(position_info.get('partial_closed', False)),
            int(position_info.get('breakeven_moved', False)),
            int(position_info.get('trailing_active', False)),
            json.dumps(position_info)
        ))

        self.conn.commit()
        logger.debug(f"Position {position_info['ticket']} saved to database")

    def load_positions(self):
        """Load all active positions from database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT state_json FROM active_positions')

        positions = {}
        for row in cursor.fetchall():
            pos_info = json.loads(row[0])
            pos_info['entry_time'] = datetime.fromisoformat(pos_info['entry_time'])
            positions[pos_info['ticket']] = pos_info

        logger.info(f"Loaded {len(positions)} active positions from database")
        return positions

    def remove_position(self, ticket):
        """Remove position from active positions."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM active_positions WHERE ticket = ?', (ticket,))
        self.conn.commit()
        logger.debug(f"Position {ticket} removed from database")

    def save_trade(self, trade_info):
        """Save completed trade to history."""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO trade_history
            (ticket, direction, entry_price, exit_price, lot_size, entry_time, exit_time,
             pnl, r_multiple, entry_model, exit_reason, score, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_info['ticket'],
            trade_info['direction'],
            trade_info['entry_price'],
            trade_info.get('exit_price'),
            trade_info['lot_size'],
            trade_info['entry_time'].isoformat() if isinstance(trade_info['entry_time'], datetime) else trade_info['entry_time'],
            trade_info.get('exit_time').isoformat() if trade_info.get('exit_time') else None,
            trade_info.get('pnl'),
            trade_info.get('r_multiple'),
            trade_info.get('entry_model'),
            trade_info.get('exit_reason'),
            trade_info.get('score'),
            json.dumps(trade_info)
        ))

        self.conn.commit()
        logger.info(f"Trade {trade_info['ticket']} saved to history")

    def get_trade_history(self, limit=100):
        """Get recent trade history."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT metadata_json FROM trade_history
            ORDER BY exit_time DESC LIMIT ?
        ''', (limit,))

        trades = []
        for row in cursor.fetchall():
            trades.append(json.loads(row[0]))

        return trades

    def save_system_state(self, key, value):
        """Save system state value."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO system_state VALUES (?, ?, ?)
        ''', (key, json.dumps(value), datetime.utcnow().isoformat()))
        self.conn.commit()

    def load_system_state(self, key, default=None):
        """Load system state value."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT value FROM system_state WHERE key = ?', (key,))
        row = cursor.fetchone()

        if row:
            return json.loads(row[0])
        return default

    def save_performance_snapshot(self, metrics):
        """Save performance metrics snapshot."""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO performance_metrics
            (timestamp, equity, balance, peak_equity, drawdown_pct, daily_pnl, weekly_pnl,
             total_trades, win_rate, profit_factor, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            metrics.get('equity', 0),
            metrics.get('balance', 0),
            metrics.get('peak_equity', 0),
            metrics.get('drawdown_pct', 0),
            metrics.get('daily_pnl', 0),
            metrics.get('weekly_pnl', 0),
            metrics.get('total_trades', 0),
            metrics.get('win_rate', 0),
            metrics.get('profit_factor', 0),
            metrics.get('sharpe_ratio', 0)
        ))

        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
