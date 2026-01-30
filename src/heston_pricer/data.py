import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL"  # Now stores "CALL" or "PUT"

def fetch_options(ticker_symbol: str, target_size: int = 100) -> Tuple[List[MarketOption], float]:
    """
    PRO FETCHER:
    - Fetches OTM Puts (Left Wing) and OTM Calls (Right Wing).
    - Stratifies across all available maturities (Time).
    - Zero Intrinsic Value in the dataset.
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Get Spot Price
    try:
        S0 = ticker.fast_info.get('last_price', None)
        if S0 is None:
            hist = ticker.history(period="1d")
            S0 = hist['Close'].iloc[-1]
    except:
        return [], 0.0

    print(f"--- Pro Calibration Set: {ticker_symbol} (Spot: {S0:.2f}) ---")
    
    expirations = ticker.options
    if not expirations: return [], 0.0
    today = datetime.now()
    
    all_candidates = []
    # Use Domain Restriction immediately (T > 0.46) to fix Xi instability
    MIN_T, MAX_T = 0.46, 2.5 
    
    print("Scanning option chains (Puts & Calls)...")
    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (d - today).days / 365.25
            if not (MIN_T <= T <= MAX_T): continue

            # Fetch BOTH chains
            chain = ticker.option_chain(exp_str)
            calls = chain.calls
            puts = chain.puts
            
            # --- SELECTION LOGIC ---
            # 1. OTM Puts (Strikes < Spot) -> Capture Downside Skew
            candidates_puts = puts[puts['strike'] < S0].copy()
            candidates_puts['type'] = 'PUT'
            
            # 2. OTM Calls (Strikes >= Spot) -> Capture Upside/Smile
            candidates_calls = calls[calls['strike'] >= S0].copy()
            candidates_calls['type'] = 'CALL'
            
            # Combine and Filter
            combined = pd.concat([candidates_puts, candidates_calls])
            
            # Quality Mask
            mask = (combined['bid'] > 0.05) & (combined['openInterest'] > 0)
            valid = combined[mask].copy()
            
            for _, row in valid.iterrows():
                mid = (row['bid'] + row['ask']) / 2.0
                spread = row['ask'] - row['bid']
                
                # Moneyness Filter (0.6 to 1.5 covers the relevant smile)
                moneyness = row['strike'] / S0
                if not (0.6 <= moneyness <= 1.5): continue

                all_candidates.append({
                    'strike': row['strike'],
                    'maturity': T,
                    'market_price': mid,
                    'spread': spread,
                    'moneyness': moneyness,
                    'type': row['type']
                })
        except: continue

    if not all_candidates: return [], S0
    df = pd.DataFrame(all_candidates)

    # 3. STRATIFIED SELECTION
    unique_maturities = sorted(df['maturity'].unique())
    n_maturities = len(unique_maturities)
    if n_maturities == 0: return [], S0

    target_per_date = max(8, target_size // n_maturities)
    
    final_selection = []
    print(f"Stratifying across {n_maturities} maturities...")
    
    for mat in unique_maturities:
        mat_slice = df[df['maturity'] == mat]
        
        # Split by Type instead of arbitrary moneyness
        # This ensures we get both Left Wing (Puts) and Right Wing (Calls)
        puts_slice = mat_slice[mat_slice['type'] == 'PUT'].sort_values('spread')
        calls_slice = mat_slice[mat_slice['type'] == 'CALL'].sort_values('spread')
        
        n_side = target_per_date // 2
        
        best_puts = puts_slice.head(n_side)
        best_calls = calls_slice.head(n_side + 2) # Slightly more calls usually available
        
        final_selection.extend(best_puts.to_dict('records'))
        final_selection.extend(best_calls.to_dict('records'))

    # 4. FINAL POLISH
    final_df = pd.DataFrame(final_selection)
    
    # Random sample if over target (preserves distribution better than spread sort)
    if len(final_df) > target_size:
        final_df = final_df.sample(n=target_size, random_state=42)
    
    market_options = [
        MarketOption(r['strike'], r['maturity'], r['market_price'], r['type'])
        for _, r in final_df.iterrows()
    ]
    
    market_options.sort(key=lambda x: (x.maturity, x.strike))
    print(f"Selected {len(market_options)} OTM options (Puts & Calls).")
    return market_options, S0