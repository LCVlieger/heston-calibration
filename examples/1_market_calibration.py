import time
import json
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy.ndimage import gaussian_filter

# Local package imports
try:
    from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC, implied_volatility, SimpleYieldCurve
    from heston_pricer.analytics import HestonAnalyticalPricer
    from heston_pricer.data import fetch_options
    from heston_pricer.instruments import EuropeanOption, OptionType
except ImportError:
    raise ImportError("heston_pricer package not found. Ensure PYTHONPATH is set correctly.")

""" 
1_market_calibration.py
-----------------------
Calibrates Heston parameters to live market data (NVDA/SPX) using both
Analytical and Monte Carlo methods with Yield Curve support.
"""

def save_results(ticker, S0, r_curve, q, res_ana, res_mc, options, init_guess):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_{ticker}_{timestamp}"
    
    # --- 1. Save Metadata (Now includes MC and Initial Guess) ---
    with open(f"{base_name}_meta.json", "w") as f: 
        json.dump({
            "market": {
                "S0": S0, 
                "r": r_curve.to_dict(), 
                "q": q
            }, 
            "initial_guess": {
                "kappa": init_guess[0],
                "theta": init_guess[1],
                "xi": init_guess[2],
                "rho": init_guess[3],
                "v0": init_guess[4]
            },
            "analytical": res_ana, 
            "monte_carlo_results": res_mc 
        }, f, indent=4)

    # --- 2. VALIDATION TABLE ---
    get_params = lambda res: [res.get(k, 0) for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    rows = []
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments with Yield Curve...")

    for opt in options:
        is_put = (opt.option_type == "PUT")
        
        # Get maturity-specific rate
        r_T = r_curve.get_rate(opt.maturity)
        
        # Helper to price based on parameters
        def price_with_params(params):
            if is_put:
                return HestonAnalyticalPricer.price_european_put(
                    S0, opt.strike, opt.maturity, r_T, q, *params
                )
            else:
                return HestonAnalyticalPricer.price_european_call(
                    S0, opt.strike, opt.maturity, r_T, q, *params
                )

        # 1. Analytical Params Price
        p_ana = price_with_params(get_params(res_ana))
        
        # 2. Monte Carlo Params Price 
        # (Using analytical pricer to verify the theoretical fit of MC parameters)
        p_mc = price_with_params(get_params(res_mc))
            
        # 3. Market IV
        iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r_T, q, opt.option_type)

        rows.append({
            "Type": opt.option_type,
            "T": opt.maturity, 
            "K": opt.strike, 
            "Mkt": opt.market_price, 
            
            "Ana": round(p_ana, 2), 
            "Err_A": round(p_ana - opt.market_price, 2),
            
            "MC": round(p_mc, 2),
            "Err_MC": round(p_mc - opt.market_price, 2),
            
            "IV_Mkt": iv_mkt,
            "r_used": round(r_T, 4)
        })

    df = pd.DataFrame(rows)
    
    # Print comparison of errors
    print(df[["Type", "T", "K", "Mkt", "Ana", "Err_A", "MC", "Err_MC"]].to_string(index=False))
    
    # Save full CSV
    df.to_csv(f"{base_name}_prices.csv", index=False)
    print(f"\n-> Saved results to {base_name}_prices.csv")

def plot_surface(S0, r, q, params, ticker, filename, market_options=None):
    pass

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def clear_numba_cache():
    for root, dirs, files in os.walk("src"):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

def extract_implied_dividends(options, S0, r_curve):
    """
    Extracts implied dividend yield q(T) using Put-Call Parity on ATM options.
    Formula: q = r - (1/T) * ln((C - P + K*exp(-rT)) / S0)
    """
    from collections import defaultdict
    
    # 1. Group by Maturity
    maturities = defaultdict(list)
    for opt in options:
        maturities[opt.maturity].append(opt)
        
    q_tenors = []
    q_rates = []
    
    print("\n[Implied Dividend Extraction]")
    print(f"{'Maturity':<10} {'Rate(r)':<10} {'Implied(q)':<12} {'Pairs':<5}")
    print("-" * 40)

    for T in sorted(maturities.keys()):
        opts = maturities[T]
        r = r_curve.get_rate(T)
        
        # 2. Find valid Put-Call pairs (Same Strike)
        # We focus on Near-The-Money (0.95 < K/S < 1.05) for best accuracy
        calls = {o.strike: o.market_price for o in opts if o.option_type == "CALL"}
        puts = {o.strike: o.market_price for o in opts if o.option_type == "PUT"}
        
        implied_qs = []
        
        common_strikes = set(calls.keys()).intersection(set(puts.keys()))
        
        for K in common_strikes:
            # Filter for ATM only to reduce noise
            if 0.95 <= K/S0 <= 1.05:
                C = calls[K]
                P = puts[K]
                
                # Put-Call Parity: S*e^-qT - K*e^-rT = C - P
                # S*e^-qT = C - P + K*e^-rT
                # -qT = ln( (C - P + K*e^-rT) / S )
                lhs = (C - P + K * np.exp(-r * T)) / S0
                
                if lhs > 0:
                    q_est = -np.log(lhs) / T
                    implied_qs.append(q_est)

        # 3. Average the estimates for this tenor
        if implied_qs:
            avg_q = np.mean(implied_qs)
            q_tenors.append(T)
            q_rates.append(avg_q)
            print(f"{T:<10.4f} {r:<10.4f} {avg_q:<12.4%} {len(implied_qs):<5}")
        else:
            # Fallback if no pairs found (rare)
            print(f"{T:<10.4f} {r:<10.4f} {'N/A':<12} 0")

    # 4. Construct Curve (fill ends if needed)
    return SimpleYieldCurve(q_tenors, q_rates)


def main():
    clear_numba_cache()
    os.makedirs("results", exist_ok=True)
    
    ticker = "^SPX" #"^SPX" #"NVDA" 
    
    options, S0 = fetch_options(ticker)
    if not options:
        log(f"No liquidity for {ticker}")
        return

    options.sort(key=lambda x: (x.maturity, x.strike))
    log(f"Target: {ticker} (S0={S0:.2f}) | N={len(options)}")
    
    avg_mkt_price = np.mean([o.market_price for o in options]) if options else 1.0
    q = 0.011 #0.0002 #0.011 #0.0002
    
    # Yield Curve Data
    tenors = [0.08, 0.25, 0.5, 1.0, 2.0, 3.0]
    rates = [0.0376, 0.0368, 0.0363, 0.0352, 0.0356, 0.0366]

    my_curve = SimpleYieldCurve(tenors, rates)
    
    # Initialize Calibrators
    cal_ana = HestonCalibrator(S0, r_curve=my_curve, q=q)
    
    # --- UPDATE: DYNAMIC MC SETTINGS ---
    # 1. Calculate max maturity to determine steps (Days)
    max_maturity = options[-1].maturity if options else 1.0
    # 2. Use 252 trading days per year for steps (1 step = 1 day)
    n_daily_steps = int(max_maturity * 252) 
    # Ensure at least 50 steps for very short options to maintain accuracy
    n_steps_mc = max(n_daily_steps, 50)
    
    log(f"Monte Carlo Config: 30,000 Paths | {n_steps_mc} Steps (Daily Resolution)")

    cal_mc = HestonCalibratorMC(
        S0, 
        r_curve=my_curve, 
        q=q, 
        n_paths=30_000,   # Requested: 50k Paths
        n_steps=n_steps_mc # Requested: Steps = Days
    )
    
    # UPDATED INITIAL GUESS
    # kappa, theta, xi, rho, v0
    init_guess = [2.0, 0.025, 0.1, -0.5, 0.015] #[2.0, 0.1, 0.1, -0.3, 0.02]

    # --- 1. Analytical Calibration ---
    t0 = time.time()    
    res_ana = cal_ana.calibrate(options, init_guess)
    
    rmse_p_ana = np.sqrt(res_ana['fun'] / len(options))
    log(f"Analytical: rmse={rmse_p_ana:.4f} ({rmse_p_ana/avg_mkt_price:.2%}) , IV-rmse={res_ana['rmse_iv']:.4f} ({res_ana['rmse_iv']:.2%}) ({time.time()-t0:.2f}s)") 
    
    # --- 2. Monte Carlo Calibration ---
    t1 = time.time()
    try:
        # Use Analytical results as warm start
        mc_guess = init_guess
        
        res_mc = cal_mc.calibrate(options, mc_guess)
        
        rmse_p_mc = np.sqrt(res_mc['fun'] / len(options)) 
        log(f"MonteCarlo: rmse={rmse_p_mc:.4f} ({rmse_p_mc/avg_mkt_price:.2%}) , IV-rmse={res_mc['rmse_iv']:.4f} ({res_mc['rmse_iv']:.2%}) ({time.time()-t1:.2f}s)")
    except Exception as e:
        log(f"MC Fail: {e}")
        res_mc = res_ana 

    # --- Comparison Table (Now with Initial Guess) ---
    params = ['kappa', 'theta', 'xi', 'rho', 'v0']
    df_params = pd.DataFrame({
        'Init': init_guess,
        'Ana': [res_ana.get(p, 0.0) for p in params],
        'MC': [res_mc.get(p, 0.0) for p in params]
    }, index=params)
    print("\nParameter Comparison (Init -> Ana -> MC):")
    print(df_params.to_string(float_format="{:.4f}".format))
    
    # --- Save & Validate (Includes MC Repricing) ---
    save_results(ticker, S0, my_curve, q, res_ana, res_mc, options, init_guess)

if __name__ == "__main__":
    main()