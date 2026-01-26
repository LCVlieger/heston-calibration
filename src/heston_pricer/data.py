import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from .calibration import MarketOption

# fetch real-time options using yahoo finance. 

def fetch_options(ticker_symbol: str, max_per_bucket: int = 6) -> Tuple[List[MarketOption], float]:
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        # Try to get real-time price first, fallback to previous close
        S0 = ticker.fast_info.get('last_price', None)
        if S0 is None:
            hist = ticker.history(period="1d")
            if hist.empty: raise ValueError("No price data")
            S0 = hist['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching spot price: {e}")
        return [], 0.0

    buckets = {
        "Short":  {'min': 0.10, 'max': 0.40, 'count': 0},
        "Medium": {'min': 0.40, 'max': 1.00, 'count': 0},
        "Long":   {'min': 1.00, 'max': 2.50, 'count': 0}
    }
    
    # Expanded moneyness for better surface coverage
    target_moneyness = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    market_options = []
    
    print(f"Fetching option chain for {ticker_symbol} (Spot: {S0:.2f})...")
    expirations = ticker.options
    if not expirations: return [], 0.0

    for exp_str in expirations:
        if len(market_options) > 60: break

        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (exp_date - datetime.now()).days / 365.25
        except: continue
        
        # Skip extremely short expirations (< 2 weeks) to avoid microstructure noise
        if T < 0.04: continue

        target_bucket = next((name for name, b in buckets.items() 
                              if b['min'] <= T <= b['max'] and b['count'] < max_per_bucket), None)
        if not target_bucket: continue

        try:
            calls = ticker.option_chain(exp_str).calls
        except: continue
        if calls.empty: continue

        # Filter roughly around spot
        calls = calls[(calls['strike'] > S0 * 0.70) & (calls['strike'] < S0 * 1.40)]
        
        selected_indices = set()
        for m in target_moneyness:
            target_strike = S0 * m
            # Find closest strike
            calls['dist'] = (calls['strike'] - target_strike).abs()
            if calls.empty: continue
            
            # Get the indices of the closest strikes
            # We take the top 1 closest to ensure we get data if the absolute closest is illiquid
            closest_candidates = calls.nsmallest(1, 'dist')
            for idx in closest_candidates.index:
                selected_indices.add(idx)

        for idx in selected_indices:
            if buckets[target_bucket]['count'] >= max_per_bucket: break
            row = calls.loc[idx]
            
            # --- PRICING LOGIC ---
            bid = row.get('bid', 0.0)
            ask = row.get('ask', 0.0)
            last = row['lastPrice']
            
            market_price = 0.0
            
            # Logic: Prefer Mid > Last. 
            # If off-hours (bid/ask=0), force Last.
            if bid > 0 and ask > 0 and bid < ask:
                market_price = (bid + ask) / 2.0
            else:
                market_price = last
            
            # --- VALIDITY CHECKS ---
            if market_price <= 0.05: continue 
            
            # Intrinsic Value Check (Arbitrage)
            # If using Last Price, this check might fail due to Spot movement. 
            # We relax it slightly for off-hours debugging (0.95 factor).
            intrinsic = max(S0 - row['strike'], 0)
            if market_price < (intrinsic * 0.95): continue 
            
            market_options.append(MarketOption(
                strike=float(row['strike']),
                maturity=float(T),
                market_price=float(market_price),
                option_type="CALL"
            ))
            buckets[target_bucket]['count'] += 1

    print(f"Selected {len(market_options)} instruments across maturities.")
    return market_options, S0