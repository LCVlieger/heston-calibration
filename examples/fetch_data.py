import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from heston_pricer.calibration import MarketOption

def fetch_spx_options(min_open_interest: int = 100) -> Tuple[List[MarketOption], float]:
    """
    Fetches S&P 500 (^SPX) option chains using Bucket Scanning.
    Ensures we get a Term Structure (Short, Medium, Long) rather than just next week's options.
    """
    ticker_symbol = "^SPX"
    print(f"--- 1. Connecting to Yahoo Finance ({ticker_symbol}) ---")
    
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        hist = ticker.history(period="1d")
        S0 = hist['Close'].iloc[-1]
        print(f"    Spot Price (S0): {S0:.2f}")
    except Exception:
        raise ValueError("Could not fetch spot price. Check Internet/Ticker.")

    expirations = ticker.options
    print(f"    Found {len(expirations)} total expiration dates.")
    
    market_options = []
    
    # --- BUCKET DEFINITIONS ---
    # We want to find at least one valid maturity in each bucket.
    buckets = {
        "Short (14d-3m)":  {'min': 0.04, 'max': 0.25, 'filled': False},
        "Medium (3m-9m)":  {'min': 0.25, 'max': 0.75, 'filled': False},
        "Long (9m-1.5y)":  {'min': 0.75, 'max': 1.50, 'filled': False}
    }
    
    print("\n--- 2. Scanning Option Chains (Bucket Strategy) ---")
    
    # Iterate through ALL expirations (or a large slice) to find matches for our buckets
    # We increase the slice to 50 to ensure we reach the yearly options.
    for exp_date_str in expirations[:60]:
        
        # Check if we are done
        if all(b['filled'] for b in buckets.values()):
            print("    >> All buckets filled. Stopping scan.")
            break

        # Calculate T
        exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
        T = (exp_date - datetime.now()).days / 365.25
        
        # Determine which bucket this T falls into
        target_bucket = None
        for name, limits in buckets.items():
            if limits['min'] <= T <= limits['max'] and not limits['filled']:
                target_bucket = name
                break
        
        # If this date doesn't fit a needed bucket, skip it (unless we want extra density)
        if target_bucket is None:
            continue
            
        print(f"    -> Checking Expiry: {exp_date_str} (T={T:.3f}y) for [{target_bucket}]")
        
        try:
            chain = ticker.option_chain(exp_date_str)
            calls = chain.calls
        except Exception as e:
            print(f"       [!] Error fetching chain: {e}")
            continue

        # FILTER: Moneyness & Liquidity
        # Moneyness: 0.85 < K/S0 < 1.15 (Focus on the smile around the money)
        mask = (
            (calls['strike'] > S0 * 0.85) & 
            (calls['strike'] < S0 * 1.15) & 
            (calls['openInterest'] > min_open_interest)
        )
        
        filtered_calls = calls[mask]
        
        if filtered_calls.empty:
            print(f"       [!] No liquid options found (OI > {min_open_interest}).")
            continue
            
        # Data Extraction
        count = 0
        # We limit to ~10 options per maturity to prevent the optimizer from 
        # being overwhelmed by one specific date.
        for _, row in filtered_calls.iloc[::2].iterrows(): # Take every 2nd option to spread strikes
            bid, ask = row['bid'], row['ask']
            price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else row['lastPrice']
            
            if price < 0.05: continue
            
            market_options.append(MarketOption(
                strike=row['strike'],
                maturity=T,
                market_price=price,
                option_type="CALL"
            ))
            count += 1
            
        if count > 0:
            print(f"       Loaded {count} options. >> Bucket [{target_bucket}] FILLED.")
            buckets[target_bucket]['filled'] = True
            
    print(f"\n--- 3. Summary ---")
    print(f"    Total Liquid Options Collected: {len(market_options)}")
    for name, status in buckets.items():
        print(f"    {name}: {'Found' if status['filled'] else '❌ Missing'}")
    
    return market_options, S0

if __name__ == "__main__":
    fetch_spx_options()