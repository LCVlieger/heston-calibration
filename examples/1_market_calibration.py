import time
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Tuple, Dict

from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC, MarketOption
from heston_pricer.analytics import HestonAnalyticalPricer

# ==========================================
# 1. HELPER: DATA FETCHING (Consolidated)
# ==========================================
def fetch_options(ticker: str, max_per_bucket: int = 5) -> Tuple[List[MarketOption], float]:
    """
    Fetches live option chain from Yahoo Finance, filters for liquidity/moneyness,
    and returns a clean list of MarketOption objects.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get Spot Price (try fast history first)
        hist = stock.history(period="1d")
        if hist.empty:
            print(f"   [Data] Error: Could not fetch spot price for {ticker}")
            return [], 0.0
        current_price = hist['Close'].iloc[-1]
    except Exception as e:
        print(f"   [Data] Connection Error: {e}")
        return [], 0.0

    expirations = stock.options
    if not expirations:
        print(f"   [Data] Error: No option chains found for {ticker}")
        return [], 0.0

    # Select expirations: Skip very near term (<14 days), take next 3 months
    target_expirations = []
    for exp in expirations:
        days = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        if 14 < days < 300:
            target_expirations.append(exp)
        if len(target_expirations) >= 3: break
    
    all_options = []
    print(f"   [Data] Processing chains: {target_expirations}")
    
    for exp_date in target_expirations:
        try:
            chain = stock.option_chain(exp_date).calls
            
            # 1. Liquidity Filter
            chain = chain[
                (chain['volume'] > 10) & 
                (chain['openInterest'] > 50) & 
                (chain['bid'] > 0.50)
            ].copy()
            
            # 2. Moneyness Filter (0.85 < K/S < 1.15)
            chain = chain[
                (chain['strike'] > current_price * 0.85) & 
                (chain['strike'] < current_price * 1.15)
            ]

            # 3. Sampling (Avoid overloading the MC calibrator)
            if len(chain) > max_per_bucket:
                # Take equidistant samples (e.g., every Nth item)
                indices = np.linspace(0, len(chain)-1, max_per_bucket, dtype=int)
                chain = chain.iloc[indices]

            # 4. Convert to MarketOption
            T = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()).days / 365.0
            
            for _, row in chain.iterrows():
                mid_price = (row['bid'] + row['ask']) / 2
                all_options.append(MarketOption(
                    strike=row['strike'],
                    maturity=T,
                    market_price=mid_price,
                    option_type="CALL"
                ))
        except Exception:
            continue

    return all_options, current_price

# ==========================================
# 2. HELPER: SAVE RESULTS
# ==========================================
def save_results_to_disk(ticker, S0, r, q, res_ana, res_mc, initial_guess, options, filename_prefix="calibration"):
    """
    Saves calibration metadata (JSON) and a detailed pricing comparison (CSV).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{filename_prefix}_{ticker}_{timestamp}"
    
    # A. JSON Metadata
    metadata = {
        "timestamp": timestamp,
        "ticker": ticker,
        "market": {"S0": S0, "r": r, "q": q},
        "initial_guess": initial_guess,
        "analytical_results": res_ana,
        "monte_carlo_results": res_mc
    }
    
    json_path = f"{base_name}_meta.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n[Saved] Metadata: {json_path}")

    # B. CSV Pricing Table
    data_rows = []
    for opt in options:
        # Reprice using both parameter sets
        p_ana = HestonAnalyticalPricer.price_european_call(
            S0, opt.strike, opt.maturity, r, q,
            res_ana['kappa'], res_ana['theta'], res_ana['xi'], res_ana['rho'], res_ana['v0']
        )
        p_mc = HestonAnalyticalPricer.price_european_call(
            S0, opt.strike, opt.maturity, r, q,
            res_mc['kappa'], res_mc['theta'], res_mc['xi'], res_mc['rho'], res_mc['v0']
        )
        
        data_rows.append({
            "Maturity": round(opt.maturity, 4),
            "Strike": opt.strike,
            "Market_Price": opt.market_price,
            "Model_Ana": round(p_ana, 3),
            "Model_MC": round(p_mc, 3),
            "Diff_Ana": round(p_ana - opt.market_price, 3),
            "Diff_MC": round(p_mc - opt.market_price, 3)
        })
        
    df = pd.DataFrame(data_rows)
    csv_path = f"{base_name}_prices.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Saved] Prices:   {csv_path}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    print("=== FULL DUAL CALIBRATION: ANALYTICAL VS MONTE CARLO ===\n")

    # 1. Configuration
    ticker = "NVDA"
    r = 0.045      # Risk-Free Rate (4.5%)
    q = 0.003      # Dividend Yield (0.3%)
    
    # 2. Fetch Data
    print(f"Fetching live data for {ticker}...")
    market_options, S0 = fetch_options(ticker, max_per_bucket=8)

    if not market_options:
        print("No valid options found. Check ticker or market hours.")
        return

    # Sort for cleaner display
    market_options.sort(key=lambda x: (x.maturity, x.strike))
    
    print(f"\n[Environment] Spot: {S0:.2f} | Rate: {r:.1%} | Div: {q:.1%} | Instruments: {len(market_options)}")

    # 3. Initialize Calibrators
    calibrator_ana = HestonCalibrator(S0, r, q)
    # Note: 20k paths / 100 steps is a balance between speed and precision for calibration
    calibrator_mc = HestonCalibratorMC(S0, r, q, n_paths=50000, n_steps=400)
    
    # 4. Define Initial Guess [kappa, theta, xi, rho, v0]
    # Using a generic start ensures we test the optimizer's convergence power
    initial_guess = [1.5, 0.06, 0.5, -0.6, 0.06]
    print(f"Initial Guess: {initial_guess}")

    # ---------------------------------------------------------
    # 5. Run Analytical Calibration (Fourier)
    # ---------------------------------------------------------
    print(f"\n[1/2] Analytical Calibration (L-BFGS-B + Fourier)...")
    t0 = time.time()
    try:
        res_ana = calibrator_ana.calibrate(market_options, init_guess=initial_guess)
        print(f"   -> Finished in {time.time() - t0:.2f}s")
        print(f"   -> Success: {res_ana['success']} | RMSE-IV: {res_ana.get('rmse_iv', 0)*100:.2f}%")
    except Exception as e:
        print(f"   -> Analytical Calibration Failed: {e}")
        return

    # ---------------------------------------------------------
    # 6. Run Monte Carlo Calibration (Simulation)
    # ---------------------------------------------------------
    print(f"\n[2/2] Monte Carlo Calibration (L-BFGS-B + Simulation)...")
    t1 = time.time()
    try:
        # We use the SAME initial guess to benchmark independent convergence
        res_mc = calibrator_mc.calibrate(market_options, init_guess=initial_guess)
        print(f"   -> Finished in {time.time() - t1:.2f}s")
        print(f"   -> Success: {res_mc['success']} | RMSE-IV: {res_mc.get('rmse_iv', 0)*100:.2f}%")
    except Exception as e:
        print(f"   -> Monte Carlo Calibration Failed: {e}")
        res_mc = res_ana # Fallback

    # ---------------------------------------------------------
    # 7. Comparison & Output
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(f"{'PARAMETER CONVERGENCE':^75}")
    print("="*75)
    print(f"{'Param':<10} | {'Analytical':<15} | {'Monte Carlo':<15} | {'Diff':<10}")
    print("-" * 75)
    
    params = ['kappa', 'theta', 'xi', 'rho', 'v0']
    for p in params:
        v1 = res_ana.get(p, 0.0)
        v2 = res_mc.get(p, 0.0)
        print(f"{p:<10} | {v1:<15.4f} | {v2:<15.4f} | {abs(v1-v2):<10.4f}")
    print("-" * 75)

    print("\n[Price Fit Check (Sample)]")
    print(f"{'Mat':<6} {'Strike':<8} {'Market':<8} {'Ana':<8} {'MC':<8} {'Diff(MC)'}")
    print("-" * 75)
    
    # Show every Nth option to fit on screen
    step = max(1, len(market_options) // 10)
    for i, opt in enumerate(market_options):
        if i % step == 0:
            p_ana = HestonAnalyticalPricer.price_european_call(
                S0, opt.strike, opt.maturity, r, q, **{k: res_ana[k] for k in params}
            )
            p_mc = HestonAnalyticalPricer.price_european_call(
                S0, opt.strike, opt.maturity, r, q, **{k: res_mc[k] for k in params}
            )
            print(f"{opt.maturity:<6.2f} {opt.strike:<8.0f} {opt.market_price:<8.2f} {p_ana:<8.2f} {p_mc:<8.2f} {p_mc - opt.market_price:+.2f}")

    # 8. Save
    save_results_to_disk(ticker, S0, r, q, res_ana, res_mc, initial_guess, market_options)

if __name__ == "__main__":
    main()