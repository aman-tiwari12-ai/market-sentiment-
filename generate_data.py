"""
Generate realistic synthetic datasets matching:
1. Bitcoin Fear & Greed Index (daily)
2. Hyperliquid historical trader data

Note: Replace data/fear_greed.csv and data/trades.csv with the real datasets
      when available — all downstream code is identical.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ── 1. Fear & Greed Index ───────────────────────────────────────────────────
start = datetime(2024, 1, 1)
end   = datetime(2025, 3, 31)
dates = pd.date_range(start, end, freq='D')
n_days = len(dates)

# Simulate realistic fear/greed cycles (mean-reverting with regime shifts)
fg_values = []
val = 45.0
for i in range(n_days):
    shock = np.random.normal(0, 8)
    val = np.clip(val * 0.92 + 50 * 0.08 + shock, 1, 99)
    fg_values.append(round(val))

def classify_fg(v):
    if v <= 24:   return 'Extreme Fear'
    elif v <= 44: return 'Fear'
    elif v <= 54: return 'Neutral'
    elif v <= 74: return 'Greed'
    else:         return 'Extreme Greed'

fg_df = pd.DataFrame({
    'date':           dates,
    'fg_value':       fg_values,
    'classification': [classify_fg(v) for v in fg_values]
})
fg_df.to_csv('data/fear_greed.csv', index=False)
print(f"Fear/Greed: {len(fg_df)} rows, {fg_df['classification'].value_counts().to_dict()}")

# ── 2. Hyperliquid Trader Data ──────────────────────────────────────────────
N_ACCOUNTS  = 120
N_TRADES    = 85_000
SYMBOLS     = ['BTC', 'ETH', 'SOL', 'ARB', 'DOGE', 'AVAX', 'BNB', 'OP', 'WIF', 'PEPE']

accounts = [f"0x{np.random.randint(0x10000000, 0xFFFFFFFF):08X}" for _ in range(N_ACCOUNTS)]

# Assign trader archetypes
archetype_probs = np.random.dirichlet([2, 2, 1.5, 1], N_ACCOUNTS)
archetypes = np.argmax(archetype_probs, axis=1)  # 0=scalper,1=swing,2=whale,3=degen

archetype_params = {
    0: dict(lev_mu=8,  lev_sd=4,  size_mu=3000,  size_sd=1500, win_p=0.52, freq=4.0),   # scalper
    1: dict(lev_mu=5,  lev_sd=3,  size_mu=8000,  size_sd=4000, win_p=0.55, freq=1.2),   # swing
    2: dict(lev_mu=3,  lev_sd=2,  size_mu=50000, size_sd=20000,win_p=0.58, freq=0.8),   # whale
    3: dict(lev_mu=20, lev_sd=10, size_mu=5000,  size_sd=3000, win_p=0.42, freq=3.0),   # degen
}

rows = []
trade_dates = pd.date_range(start, end, freq='D')
fg_lookup = dict(zip(fg_df['date'], fg_df['classification']))

for _ in range(N_TRADES):
    acc_idx   = np.random.choice(N_ACCOUNTS)
    acc       = accounts[acc_idx]
    arch      = archetypes[acc_idx]
    p         = archetype_params[arch]
    trade_date= np.random.choice(trade_dates)
    sentiment = fg_lookup.get(trade_date, 'Neutral')

    # Sentiment adjusts behavior
    fear_day  = sentiment in ('Fear', 'Extreme Fear')
    greed_day = sentiment in ('Greed', 'Extreme Greed')

    leverage  = max(1, np.random.normal(
        p['lev_mu'] * (0.80 if fear_day else 1.15 if greed_day else 1.0),
        p['lev_sd']
    ))
    size      = max(10, np.random.normal(
        p['size_mu'] * (0.85 if fear_day else 1.10 if greed_day else 1.0),
        p['size_sd']
    ))
    # Long bias shifts with sentiment
    long_bias = 0.62 if greed_day else 0.42 if fear_day else 0.52
    side      = 'BUY' if np.random.random() < long_bias else 'SELL'
    symbol    = np.random.choice(SYMBOLS, p=[.35,.20,.12,.07,.06,.05,.05,.04,.03,.03])

    # PnL: winners/losers shaped by archetype win rate + sentiment
    win_prob  = p['win_p'] * (0.92 if fear_day else 1.05 if greed_day else 1.0)
    win_prob  = np.clip(win_prob, 0.1, 0.9)
    is_win    = np.random.random() < win_prob
    pnl_base  = size * leverage * 0.01
    pnl       = abs(np.random.exponential(pnl_base)) if is_win else -abs(np.random.exponential(pnl_base * 1.3))

    exec_price = np.random.uniform(20000, 70000) if symbol == 'BTC' else np.random.uniform(1000, 4000)
    ts = pd.Timestamp(trade_date) + pd.Timedelta(seconds=np.random.randint(0, 86400))

    rows.append({
        'account':       acc,
        'symbol':        symbol + '-USD',
        'execution_price': round(exec_price, 2),
        'size':          round(size, 4),
        'side':          side,
        'time':          ts,
        'start_position': round(np.random.uniform(-size, size), 4),
        'event':         np.random.choice(['FILL', 'LIQUIDATION', 'FILL', 'FILL'], p=[.85,.05,.05,.05]),
        'closedPnL':     round(pnl, 4),
        'leverage':      round(leverage, 2),
        'archetype':     ['scalper','swing','whale','degen'][arch],  # for validation only
    })

trades_df = pd.DataFrame(rows)
trades_df.to_csv('data/trades.csv', index=False)
print(f"Trades: {len(trades_df)} rows, {trades_df['account'].nunique()} accounts")
print(f"Columns: {list(trades_df.columns)}")
