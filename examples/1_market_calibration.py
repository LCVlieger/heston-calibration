import time
import json
import os
import shutil
import pandas as pd
from datetime import datetime
from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC
from heston_pricer.analytics import HestonAnalyticalPricer
from heston_pricer.data import fetch_options

def clear_numba_cache():
    for root, dirs, files in os.walk("src"):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

def save_results(ticker, S0, res_ana, res_mc, options):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"calibration_{ticker}_{timestamp}"
    
    with open(f"{base_name}_meta.json", "w") as f: 
        json.dump({
            "market": {"S0": S0, "r": 0.045, "q": 0.003}, 
            "analytical": res_ana, 
            "monte_carlo_results": res_mc 
        }, f, indent=4)

    rows = []
    get_params = lambda r: [r.get(k, 0) for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    
    for opt in options:
        p_ana = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, 0.045, 0.003, *get_params(res_ana))
        p_mc = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, 0.045, 0.003, *get_params(res_mc))
        
        rows.append({
            "Maturity": opt.maturity, 
            "Strike": opt.strike, 
            "Market": opt.market_price, 
            "Model_Ana": round(p_ana, 2), 
            "Diff_Ana": round(p_ana - opt.market_price, 2),
            "Model_Euler": round(p_mc, 2), 
            "Diff_Euler": round(p_mc - opt.market_price, 2)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    print("\nPricing Errors (Validation):")
    print(df.to_string())
    print(f"\nSaved: {base_name}_meta.json")

def main():
    clear_numba_cache()
    ticker = "NVDA" 
    
    options, S0 = fetch_options(ticker)
    if not options:
        print(f"No options found for {ticker}")
        return

    options.sort(key=lambda x: (x.maturity, x.strike))
    print(f"Target: {ticker} (S0={S0:.2f}) | Instruments: {len(options)}")

    r, q = 0.045, 0.003
    cal_ana = HestonCalibrator(S0, r, q)
    cal_mc = HestonCalibratorMC(S0, r, q, n_paths=100_000, n_steps=252)
    init_guess = [2.0, 0.04, 0.5, -0.7, 0.04]

    # Analytical Calibration
    print("\nRunning Analytical Calibration (Fourier)...")
    t0 = time.time()
    res_ana = cal_ana.calibrate(options, init_guess)
    print(f"Result: RMSE={res_ana['rmse_iv']*100:.2f}% (Time: {time.time()-t0:.2f}s)")

    # Monte Carlo Calibration
    print("\nRunning Monte Carlo Calibration (Euler)...")
    t1 = time.time()
    try:
        res_mc = cal_mc.calibrate(options, init_guess)
        print(f"Result: RMSE={res_mc['rmse_iv']*100:.2f}% (Time: {time.time()-t1:.2f}s)")
    except Exception as e:
        print(f"MC Calibration failed: {e}")
        res_mc = res_ana 

    print("\nParameter Convergence:")
    params = ['kappa', 'theta', 'xi', 'rho', 'v0']
    df_params = pd.DataFrame({
        'Analytical': [res_ana.get(p, 0.0) for p in params],
        'MC_Euler': [res_mc.get(p, 0.0) for p in params]
    }, index=params)
    df_params['Diff'] = (df_params['Analytical'] - df_params['MC_Euler']).abs()
    
    print(df_params.to_string(float_format="{:.4f}".format))
    
    save_results(ticker, S0, res_ana, res_mc, options)

if __name__ == "__main__":
    main()