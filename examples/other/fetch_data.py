import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL"

def fetch_options(ticker_symbol: str = "KO", max_per_bucket: int = 20) -> Tuple[List[MarketOption], float]:
    ticker = yf.Ticker(ticker_symbol)
    
    # --- 1. GET ACCURATE LIVE SPOT ---
    try:
        # fast_info is usually real-time for US stocks during market hours
        S0 = ticker.fast_info.get('last_price', None)
        
        if S0 is None:
            hist = ticker.history(period="1d")
            if hist.empty: raise ValueError("No price data found")
            S0 = hist['Close'].iloc[-1]
            
        print(f"[Data] Live Reference Spot ({ticker_symbol}): {S0:.2f}")
    except Exception as e:
        print(f"[Error] Failed to fetch spot: {e}")
        return [], 0.0

    # --- 2. SETUP SCANNING (THE FIX) ---
    market_options = []
    
    # Adjusted 'Short' min from 0.02 to 0.10.
    # T = 0.10 is approx 36 days. This filters out the noisy, jump-heavy weekly options.
    buckets = {
        "Short":  {'min': 0.10, 'max': 0.25, 'count': 0}, 
        "Medium": {'min': 0.25, 'max': 0.75, 'count': 0},
        "Long":   {'min': 0.75, 'max': 2.00, 'count': 0}
    }
    
    target_moneyness = [0.95, 1.00, 1.05, 1.10]
    
    print(f"[Data] Scanning Call Chains for {ticker_symbol}...")

    expirations = ticker.options
    if not expirations: return [], 0.0

    for exp_str in expirations:
        if len(market_options) > 60: break

        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            # Calculate T in years
            T = (exp_date - datetime.now()).days / 365.25
        except: continue
        
        # FILTER: Strict check against bucket limits
        target_bucket = next((name for name, b in buckets.items() 
                              if b['min'] <= T <= b['max'] and b['count'] < max_per_bucket), None)
        
        # If T is < 0.10, this will return None and we SKIP the chain entirely.
        if not target_bucket: continue

        try:
            calls = ticker.option_chain(exp_str).calls
        except: continue
        if calls.empty: continue

        # Filter roughly around ATM to save processing
        calls = calls[(calls['strike'] > S0 * 0.75) & (calls['strike'] < S0 * 1.35)]
        
        # --- SELECTION LOGIC ---
        selected_indices = set()
        for m in target_moneyness:
            target_strike = S0 * m
            calls['dist'] = (calls['strike'] - target_strike).abs()
            
            if calls.empty: continue
            
            best_idx = calls['dist'].idxmin()
            # Ensure we are actually close to the target moneyness (within 2.5%)
            if calls.loc[best_idx, 'dist'] / S0 < 0.025:
                selected_indices.add(best_idx)

        for idx in selected_indices:
            if buckets[target_bucket]['count'] >= max_per_bucket: break
            
            row = calls.loc[idx]
            price = row['lastPrice']
            strike = row['strike']
            
            # --- CRITICAL INTEGRITY CHECK ---
            # 1. Price limit (Avoid $0.01 options that break optimization weights)
            if price < 0.05: continue
            
            # 2. Arbitrage Check 
            # If Market Price < Intrinsic, the data is broken/stale.
            intrinsic = max(S0 - strike, 0.0)
            if price < (intrinsic - 0.5):
                continue

            market_options.append(MarketOption(
                strike=float(strike),
                maturity=float(T),
                market_price=float(price),
                option_type="CALL"
            ))
            buckets[target_bucket]['count'] += 1

    return market_options, S0

if __name__ == "__main__":
    # Test with NVDA to see if it skips the 1-week options
    opts, spot = fetch_options("NVDA")
    
    # Sort by maturity to verify
    opts.sort(key=lambda x: x.maturity)
    
    print(f"Fetched {len(opts)} options. Spot: {spot:.2f}")
    if opts:
        print(f"Min Maturity: {opts[0].maturity:.4f} (Should be > 0.10)")
        print(f"Max Maturity: {opts[-1].maturity:.4f}")