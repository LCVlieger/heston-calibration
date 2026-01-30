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
    option_type: str = "CALL" 

def fetch_options(ticker_symbol: str, target_size: int = 100) -> Tuple[List[MarketOption], float]:
    """
    PRO FETCHER:
    - Uses ASK price to avoid zero-bid crashes during after-hours/illiquid periods.
    - Stratifies across T > 0.46.
    - Filters out 'ghost' options with 0.00 price.
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
    # Domain Restriction (T > 0.46) to fix NVDA Xi instability
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
            
            candidates_puts = puts[puts['strike'] < S0].copy()
            candidates_puts['type'] = 'PUT'
            
            candidates_calls = calls[calls['strike'] >= S0].copy()
            candidates_calls['type'] = 'CALL'
            
            combined = pd.concat([candidates_puts, candidates_calls])
            
            # --- VALIDATION LOOP ---
            for _, row in combined.iterrows():
                # FIX: Use ASK price instead of Mid or Bid
                # When markets are closed, Bid often drops to 0.00, but Ask remains.
                market_p = row['ask']
                
                # FALLBACK: If Ask is also 0 (data error), try lastPrice, else skip
                if market_p <= 0.01: 
                    market_p = row['lastPrice']
                
                # FINAL SAFETY: If it's still < 0.01, it will crash the optimizer. Skip it.
                if market_p < 0.01: continue

                spread = row['ask'] - row['bid']
                
                # Moneyness Filter (0.75 to 1.25 covers the relevant smile)
                moneyness = row['strike'] / S0
                if not (0.75 <= moneyness <= 1.25): continue 

                all_candidates.append({
                    'strike': row['strike'],
                    'maturity': T,
                    'market_price': market_p, # Now using ASK
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
        
        puts_slice = mat_slice[mat_slice['type'] == 'PUT'].sort_values('spread')
        calls_slice = mat_slice[mat_slice['type'] == 'CALL'].sort_values('spread')
        
        n_side = target_per_date // 2
        
        best_puts = puts_slice.head(n_side)
        best_calls = calls_slice.head(n_side + 2)
        
        final_selection.extend(best_puts.to_dict('records'))
        final_selection.extend(best_calls.to_dict('records'))

    final_df = pd.DataFrame(final_selection)
    
    if len(final_df) > target_size:
        final_df = final_df.sample(n=target_size, random_state=42)
    
    market_options = [
        MarketOption(r['strike'], r['maturity'], r['market_price'], r['type'])
        for _, r in final_df.iterrows()
    ]
    
    market_options.sort(key=lambda x: (x.maturity, x.strike))
    print(f"Selected {len(market_options)} OTM options (Puts & Calls).")
    return market_options, S0