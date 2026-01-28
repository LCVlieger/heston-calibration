import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from .calibration import MarketOption 

def fetch_options(ticker_symbol: str, target_size: int = 150) -> Tuple[List[MarketOption], float]:
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Fetch Spot
    try:
        S0 = ticker.fast_info.get('last_price', None)
        if S0 is None:
            hist = ticker.history(period="1d")
            S0 = hist['Close'].iloc[-1]
    except:
        return [], 0.0

    print(f"--- Fetching Surface for {ticker_symbol} (Spot: {S0:.2f}) ---")
    
    expirations = ticker.options
    if not expirations: return [], 0.0

    today = datetime.now()
    
    # 2. STABILIZED MATURITY SELECTION
    MIN_T_YEARS = 21 / 365.25  # Lowered slightly to 21d to ensure data flow
    
    short_dates = []   
    med_dates = []     
    long_dates = []    

    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (d - today).days / 365.25
            
            if T < MIN_T_YEARS: continue 
            
            if T < 0.5: short_dates.append(exp_str)
            elif T < 1.5: med_dates.append(exp_str)
            else: long_dates.append(exp_str)
        except: continue

    selected_dates = []
    
    def pick_evenly(lst, n):
        if len(lst) <= n: return lst
        indices = np.linspace(0, len(lst)-1, n, dtype=int)
        return [lst[i] for i in indices]

    selected_dates.extend(pick_evenly(short_dates, 3))
    selected_dates.extend(pick_evenly(med_dates, 4))
    selected_dates.extend(long_dates) 
    
    selected_dates = sorted(list(set(selected_dates)))
    print(f"Scanning {len(selected_dates)} maturities...")

    # 3. CONSTRAINED MONEYNESS SEARCH
    # Expanded slightly to ensure we catch wings if ATM is missing in fallback
    target_moneyness = [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.35, 1.45]
    
    market_options = []

    for exp_str in selected_dates:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (exp_date - today).days / 365.25
            
            # Get RAW chain first
            raw_chain = ticker.option_chain(exp_str).calls
            if raw_chain.empty: continue

            # --- LOGIC BRANCHING ---
            
            # PRIMARY PATH: Your Strict Logic
            mask_strict = (raw_chain['openInterest'] > 50) & (raw_chain['bid'] > 0.05)
            if T > 1.5:
                mask_strict = (raw_chain['openInterest'] > 0) & (raw_chain['bid'] > 0.05)
            
            chain = raw_chain[mask_strict].copy()
            use_fallback = False

            # ELSE: Fallback if strict yielded nothing
            if chain.empty:
                # Relaxed Mask: Just need a valid price
                mask_relaxed = (raw_chain['lastPrice'] > 0) | (raw_chain['bid'] > 0)
                chain = raw_chain[mask_relaxed].copy()
                use_fallback = True # Flag to trigger relaxed logic inside loop
            
            if chain.empty: continue

            for m in target_moneyness:
                target_k = S0 * m
                
                chain['dist'] = (chain['strike'] - target_k).abs()
                candidates = chain.nsmallest(1, 'dist')
                
                if candidates.empty: continue
                
                row = candidates.iloc[0]
                
                # [FILTER] Proximity Check
                # Strict: 7.5% | Fallback: 15%
                limit_dist = S0 * 0.15 if use_fallback else S0 * 0.075
                if row['dist'] > limit_dist: continue

                # Pricing Logic
                bid, ask, last = row.get('bid', 0), row.get('ask', 0), row['lastPrice']
                
                # [FILTER] Spread Integrity
                mid = (bid + ask) / 2.0
                spread = ask - bid
                
                # Only apply strict spread check if NOT in fallback mode
                if not use_fallback:
                    if mid > 0 and (spread / mid) > 0.4: continue

                price = mid if (bid > 0 and ask > 0) else last
                
                # [FILTER] Arbitrage Check
                intrinsic = max(S0 - row['strike'], 0)
                
                if price <= intrinsic:
                    if not use_fallback:
                        continue # Strict: Skip
                    else:
                        price = intrinsic + 0.05 # Fallback: Repair

                # Avoid duplicates
                is_dupe = any(o.strike == row['strike'] and o.maturity == T for o in market_options)
                if not is_dupe:
                    market_options.append(MarketOption(
                        strike=float(row['strike']),
                        maturity=float(T),
                        market_price=float(price),
                        option_type="CALL"
                    ))
        except: continue

    # 4. Final Polish
    market_options.sort(key=lambda x: (x.maturity, x.strike))
    
    if len(market_options) > target_size:
        step = len(market_options) // target_size
        market_options = market_options[::step]

    print(f"Selected {len(market_options)} instruments. Range: T=[{market_options[0].maturity:.2f}, {market_options[-1].maturity:.2f}].")
    return market_options, S0