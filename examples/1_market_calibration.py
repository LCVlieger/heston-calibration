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

# Local package imports
try:
    from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC, implied_volatility
    from heston_pricer.analytics import HestonAnalyticalPricer
    from heston_pricer.data import fetch_options
    from heston_pricer.models.mc_pricer import MonteCarloPricer
    from heston_pricer.models.process import HestonProcess
    from heston_pricer.market import MarketEnvironment
    from heston_pricer.instruments import EuropeanOption, OptionType
except ImportError:
    raise ImportError("heston_pricer package not found. Ensure PYTHONPATH is set correctly.")

""" 
Calibrate the Heston model to market prices of real-time market options. 
Integrates high-fidelity 'Bloomberg-style' visualization for the volatility surface.
"""

def save_results(ticker, S0, r, q, res_ana, res_mc, options):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"calibration_{ticker}_{timestamp}"
    
    # Save Metadata
    with open(f"{base_name}_meta.json", "w") as f: 
        json.dump({
            "market": {"S0": S0, "r": r, "q": q}, 
            "analytical": res_ana, 
            "monte_carlo_results": res_mc 
        }, f, indent=4)

    # --- 1. VALIDATION TABLE (Proof of Convergence) ---
    # Construct MC environment from calibrated parameters
    env_mc = MarketEnvironment(
        S0=S0, r=r, q=q,
        kappa=res_mc['kappa'], theta=res_mc['theta'], 
        xi=res_mc['xi'], rho=res_mc['rho'], v0=res_mc['v0']
    )
    process_mc = HestonProcess(env_mc)
    pricer_mc = MonteCarloPricer(process_mc)

    get_params_ana = lambda res: [res.get(k, 0) for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    rows = []
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments with Monte Carlo engine...")

    for opt in options:
        # Analytical Price (Fourier)
        p_ana = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, r, q, *get_params_ana(res_ana))
        
        # Monte Carlo Price (Engine Check)
        steps = int(max(20, opt.maturity * 252)) 
        instrument = EuropeanOption(opt.strike, opt.maturity, OptionType.CALL)
        
        mc_result = pricer_mc.price(instrument, n_paths=100_000, n_steps=steps)
        p_mc = mc_result.price
        
        # Convert prices to Implied Volatility for comparison
        iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q)
        iv_mc = implied_volatility(p_mc, S0, opt.strike, opt.maturity, r, q)

        rows.append({
            "T": opt.maturity, "K": opt.strike, "Mkt": opt.market_price, 
            "Ana": round(p_ana, 2), "Err_A": round(p_ana - opt.market_price, 2),
            "MC": round(p_mc, 2), "Err_MC": round(p_mc - opt.market_price, 2),
            "IV_Mkt": iv_mkt, "IV_MC": iv_mc
        })

    df = pd.DataFrame(rows)
    print(df[["T", "K", "Mkt", "Ana", "Err_A", "MC", "Err_MC"]].to_string(index=False))
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    # --- 2. VISUALIZATION (Analytical Surface) ---
    # Generates the high-fidelity surface plot using the new styling
    plot_surface(S0, r, q, res_mc, ticker, base_name)
    
    log(f"Artifacts: {base_name}_meta.json")

def plot_surface(S0, r, q, params, ticker, filename):
    """
    Generates a high-fidelity 3D Volatility Surface using the styled configuration.
    Uses calibrated parameters (res_mc) to project the surface.
    """
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']

    # --- 1. GRID GENERATION (REALISTIC RANGES) ---
    # Moneyness M = K/S. 
    # Reality check: Liquid equity surfaces rarely exceed 50%-150% moneyness [0.5, 1.5].
    # Time T: 0.1 to 3.0 years covers the standard liquid term structure.
    M_range = np.linspace(0.4, 2.5, 100)
    T_range = np.linspace(0.12, 2.5, 100)
    
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    # --- 2. SURFACE CALCULATION ---
    # We use the Analytical pricer for the surface to ensure visual smoothness (Monte Carlo noise is undesirable in plots).
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val = Y[i, j]
            M_val = X[i, j]
            K_val = S0 * M_val
            
            price = HestonAnalyticalPricer.price_european_call(
                S0, K_val, T_val, r, q, kappa, theta, xi, rho, v0
            )
            
            try:
                iv = implied_volatility(price, S0, K_val, T_val, r, q)
                
                # [FIX] Sanity Check / Filtering
                # Remove artifacts where solver converged to bounds (0.0 or > 200%)
                if 0.01 < iv < 2.5:  
                    Z[i, j] = iv
                else:
                    Z[i, j] = np.nan
                    
            except Exception:
                Z[i, j] = np.nan

    # --- 3. VISUALIZATION (STYLED) ---
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Surface Plot
        # RdYlBu_r colormap with black edges creates the distinct 'thermal' look
        surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlBu_r, edgecolor='black', 
                               linewidth=0.1, alpha=0.9, antialiased=True)

        # 2. Camera and Bounds
        ax.dist = 11  # Pull back camera
        ax.set_xlim(0.4, 2.5)
        ax.set_ylim(2.5, 0.12) # Inverted Y for Near Maturity at bottom/front if desired, or standard
        
        # 3. Labels and Titles
        ax.set_title(rf"Heston IV Surface $\sigma(T, K/S)$: {ticker}", color='white', y=1.0, fontsize=14, fontweight='bold')
        ax.set_xlabel('Moneyness ($K/S_0$)', color='white', labelpad=15)
        ax.set_ylabel('Maturity ($T$ Years)', color='white', labelpad=15)
        ax.set_zlabel(r'Implied Volatility (%)', color='white', labelpad=15)

        # 4. View Perspective
        ax.view_init(elev=20, azim=-103.5)
        
        z_lbl = ax.set_zlabel(r'Implied Volatility $\sigma(T, M)$', color='white', labelpad=10)

        # 5. Pane and Grid Styling (Transparent Panes)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Brighten grid lines
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

        # 6. Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=14, pad=0.05)
        cbar.ax.yaxis.set_tick_params(color='white')

        # 7. Export
        output_file = f"{filename}_surface.png"
        plt.savefig(output_file, dpi=300, facecolor='black', bbox_inches='tight', pad_inches=0.1, bbox_extra_artists=[z_lbl])
        # plt.show() # Optional: Comment out for batch processing
        plt.draw()
        plt.close()

def main():
    clear_numba_cache()
    ticker = "NVDA" 
    
    # Fetch Market Data
    options, S0 = fetch_options(ticker)
    if not options:
        log(f"No liquidity for {ticker}")
        return

    options.sort(key=lambda x: (x.maturity, x.strike))
    log(f"Target: {ticker} (S0={S0:.2f}) | N={len(options)}")
    
    avg_mkt_price = np.mean([o.market_price for o in options]) if options else 1.0
    r, q = 0.045, 0.0002

    # Setup Calibrators
    cal_ana = HestonCalibrator(S0, r, q)
    cal_mc = HestonCalibratorMC(S0, r, q, n_paths=50_000, n_steps=252)
    init_guess = [3.0, 0.05, 0.3, -0.7, 0.04] 

    # --- 1. Analytical Calibration ---
    t0 = time.time()
    res_ana = cal_ana.calibrate(options, init_guess)
    
    rmse_p_ana = np.sqrt(res_ana['sse'] / len(options))
    log(f"Analytical: rmse={rmse_p_ana:.4f} ({rmse_p_ana/avg_mkt_price:.2%}) , IV-rmse={res_ana['rmse_iv']:.4f} ({res_ana['rmse_iv']:.2%}) ({time.time()-t0:.2f}s)") 
    
    # --- 2. Monte Carlo Calibration ---
    t1 = time.time()
    try:
        res_mc = cal_mc.calibrate(options, init_guess)
        rmse_p_mc = np.sqrt(res_mc['fun'] / len(options)) 
        log(f"MonteCarlo: rmse={rmse_p_mc:.4f} ({rmse_p_mc/avg_mkt_price:.2%}) , IV-rmse={res_mc['rmse_iv']:.4f} ({res_mc['rmse_iv']:.2%}) ({time.time()-t1:.2f}s)")
    except Exception as e:
        log(f"MC Fail: {e}")
        res_mc = res_ana 

    # --- 3. Parameter Comparison ---
    params = ['kappa', 'theta', 'xi', 'rho', 'v0']
    df_params = pd.DataFrame({
        'Ana': [res_ana.get(p, 0.0) for p in params],
        'MC': [res_mc.get(p, 0.0) for p in params]
    }, index=params)
    print(df_params.to_string(float_format="{:.4f}".format))
    
    # --- 4. Save and Visualize ---
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