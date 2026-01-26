import time
import json
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC, implied_volatility
from heston_pricer.analytics import HestonAnalyticalPricer
from heston_pricer.data import fetch_options
from matplotlib.colors import LightSource

""" 
Calibrate the Heston model to market prices of real-time market options. 
Compares Monte Carlo pricing and semi-analytical pricing implementations.
Reference: 'The Volatility Surface: A Practitioners Guide, Jim Gatheral'. 
"""

def save_results(ticker, S0, r, q, res_ana, res_mc, options):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"calibration_{ticker}_{timestamp}"
    
    with open(f"{base_name}_meta.json", "w") as f: 
        json.dump({
            "market": {"S0": S0, "r": r, "q": q}, 
            "analytical": res_ana, 
            "monte_carlo_results": res_mc 
        }, f, indent=4)

    get_params = lambda res: [res.get(k, 0) for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    rows = []
    
    for opt in options:
        p_ana = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, r, q, *get_params(res_ana))
        p_mc = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, r, q, *get_params(res_mc))
        
        # Calculate IVs internally for potential CSV analysis, but not for the dense terminal print
        iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q)
        iv_mc = implied_volatility(p_mc, S0, opt.strike, opt.maturity, r, q)

        rows.append({
            "T": opt.maturity, "K": opt.strike, "Mkt": opt.market_price, 
            "Ana": round(p_ana, 2), "Err_A": round(p_ana - opt.market_price, 2),
            "MC": round(p_mc, 2), "Err_MC": round(p_mc - opt.market_price, 2),
            "IV_Mkt": iv_mkt, "IV_MC": iv_mc
        })

    df = pd.DataFrame(rows)
    
    # Print the requested dense price table
    print(df[["T", "K", "Mkt", "Ana", "Err_A", "MC", "Err_MC"]].to_string(index=False))
    
    df.to_csv(f"{base_name}_prices.csv", index=False)
    plot_surface(S0, r, q, res_mc if res_mc.get('success') else res_ana, ticker, base_name)
    log(f"Artifacts: {base_name}_meta.json")
    

def plot_surface(S0, r, q, params, ticker, filename):
    T_range = np.linspace(0.1, 2.5, 40)
    M_range = np.linspace(0.7, 1.3, 40) 
    
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)
    
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val = Y[i, j]
            K_val = S0 / X[i, j] 
            price = HestonAnalyticalPricer.price_european_call(S0, K_val, T_val, r, q, kappa, theta, xi, rho, v0)
            Z[i, j] = implied_volatility(price, S0, K_val, T_val, r, q)
    
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlBu_r, edgecolor='black', 
                            linewidth=0.08, alpha=0.95, antialiased=True)
        
        # Title position preserved as correct
        ax.set_title(f"Implied Volatility Surface: {ticker}", color='white', y=1.02, fontsize=12)
        
        ax.set_xlabel('Moneyness M = S/K', color='white', labelpad=5)
        ax.set_ylabel('Time to Maturity T', color='white', labelpad=5)
        
        # CORRECTED: Reduced labelpad to 10 to pull label closer to axis
        ax.set_zlabel(r'Implied Volatility $\sigma(T, M)$', color='white', labelpad=10)
        
        ax.set_ylim(2.5, 0) 
        ax.view_init(elev=25, azim=-140) 
        
        ax.dist = 8 
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        
        fig.colorbar(surf, ax=ax, shrink=0.4, aspect=12, pad=0)
        
        # Explicitly track the text objects you want to guarantee are included
        z_lbl = ax.set_zlabel(r'Implied Volatility $\sigma(T, M)$', color='white', labelpad=10)
        # Update save command
        plt.savefig(
        f"{filename}_surface.png", 
        dpi=300, 
        facecolor='black',
        bbox_inches='tight',          # Force tight fit
        pad_inches=0.1,               # Small padding to prevent edge clipping
        bbox_extra_artists=[z_lbl]    # Force calculation of Z-label geometry
        )
        plt.close()


        
def main():
    clear_numba_cache()
    ticker = "NVDA" 
    
    options, S0 = fetch_options(ticker)
    if not options:
        log(f"No liquidity for {ticker}")
        return

    options.sort(key=lambda x: (x.maturity, x.strike))
    log(f"Target: {ticker} (S0={S0:.2f}) | N={len(options)}")

    r, q = 0.045, 0.002
    cal_ana = HestonCalibrator(S0, r, q)
    cal_mc = HestonCalibratorMC(S0, r, q, n_paths=50_000, n_steps=252)
    init_guess = [2.7, 0.282, 1.73, -0.25, 0.209] #[3.0, 0.05, 0.3, -0.7, 0.04]

    # Analytical calibration run
    t0 = time.time()
    res_ana = cal_ana.calibrate(options, init_guess)
    rmse_p_ana = np.sqrt(res_ana['sse'] / len(options))
    log(f"Analytical: RMSE=${rmse_p_ana:.4f} | IV-RMSE={res_ana['rmse_iv']:.2%} ({time.time()-t0:.2f}s)")

    init_guess = [2.7, 0.284, 1.73, -0.27, 0.213] 
    # Monte Carlo calibration run
    t1 = time.time()
    try:
        res_mc = cal_mc.calibrate(options, init_guess)
        rmse_p_mc = np.sqrt(res_mc['fun'] / len(options))
        log(f"MonteCarlo: RMSE=${rmse_p_mc:.4f} | IV-RMSE={res_mc['rmse_iv']:.2%} ({time.time()-t1:.2f}s)")
    except Exception as e:
        log(f"MC Fail: {e}")
        res_mc = res_ana 

    # Parameter output
    params = ['kappa', 'theta', 'xi', 'rho', 'v0']
    df_params = pd.DataFrame({
        'Ana': [res_ana.get(p, 0.0) for p in params],
        'MC': [res_mc.get(p, 0.0) for p in params]
    }, index=params)
    print(df_params.to_string(float_format="{:.4f}".format))
    
    save_results(ticker, S0, r, q, res_ana, res_mc, options)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def clear_numba_cache():
    for root, dirs, files in os.walk("src"):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

if __name__ == "__main__":
    main()