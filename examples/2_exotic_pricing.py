import json
import glob
import os
import numpy as np
from heston_pricer.market import MarketEnvironment
from heston_pricer.models.process import HestonProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer
from heston_pricer.instruments import BarrierOption, BarrierType, AsianOption, OptionType

def load_latest_calibration():
    """
    Finds the most recent 'calibration_*_meta.json' file.
    """
    # Search patterns to find the file you generated in Step 1
    patterns = [
        'calibration_*_meta.json',          # If running from root
        'examples/calibration_*_meta.json', # If file saved in examples
        '../calibration_*_meta.json'        # If running from inside examples
    ]
    
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
        
    if not files:
        raise FileNotFoundError(
            "Could not find any 'calibration_*_meta.json' file.\n"
            "Make sure you ran 'python examples/1_market_calibration.py' successfully."
        )
    
    # Pick the newest one
    latest_file = max(files, key=os.path.getctime)
    
    print(f"   [IO] Loading parameters from: {latest_file}")
    with open(latest_file, 'r') as f:
        data = json.load(f)
    return data

def main():
    print("=== 2. EXOTIC PRICING & RISK (GREEKS) ===\n")
    
    # 1. Load Environment
    try:
        data = load_latest_calibration()
    except Exception as e:
        print(f"Error: {e}")
        return

    # Use Monte Carlo parameters for the simulation engine
    params = data['monte_carlo_results'] 
    mkt = data['market']
    
    # Reconstruct Environment
    env = MarketEnvironment(
        S0=mkt['S0'], 
        r=mkt['r'], 
        q=mkt['q'],
        kappa=params['kappa'], 
        theta=params['theta'], 
        xi=params['xi'], 
        rho=params['rho'], 
        v0=params['v0']
    )
    
    process = HestonProcess(env)
    pricer = MonteCarloPricer(process)
    
    S0 = mkt['S0']
    T = 1.0  # 1 Year maturity
    K = S0 * 1.05 # 5% OTM Call
    
    print(f"\n[Scenario] Spot: {S0:.2f} | T: {T:.1f}y | Vol (v0^0.5): {np.sqrt(params['v0']):.1%}")

    # ----------------------------------------------------
    # Product A: Down-and-Out Call (Barrier)
    # ----------------------------------------------------
    barrier_level = S0 * 0.80
    
    # FIX: Changed 'strike' -> 'K' and 'maturity' -> 'T'
    barrier_opt = BarrierOption(
        K=K, 
        T=T, 
        barrier=barrier_level, 
        barrier_type=BarrierType.DOWN_AND_OUT, 
        option_type=OptionType.CALL
    )
    
    # 1. DOWN-AND-OUT CALL
    print(f"\n--- Product A: Down-and-Out Call (K={K:.2f}, B={barrier_level:.2f}) ---")
    
    # CRITICAL: n_steps=400 reduces discretization bias for barriers
    print("Computing Greeks (Finite Difference - 100k paths, 400 steps)...")
    res_barrier = pricer.compute_greeks(barrier_opt, n_paths=100_000, n_steps=400, seed=42)
    
    print(f"{'Metric':<10} | {'Value':<12} | {'Note'}")
    print("-" * 45)
    print(f"{'Price':<10} | {res_barrier['price']:<12.4f} |")
    print(f"{'Delta':<10} | {res_barrier['delta']:<12.4f} |")
    print(f"{'Gamma':<10} | {res_barrier['gamma']:<12.4f} | Negative Gamma = Hedging Bleed")
    print(f"{'Vega':<10} | {res_barrier['vega_v0']:<12.4f} | Short Volatility Exposure")
    print("-" * 45)

    # 2. ASIAN CALL
    print(f"\n--- Product B: Arithmetic Asian Call (K={K:.2f}) ---")
    print("Computing Greeks (100k paths, 400 steps)...")
    asian_opt = AsianOption(K=K, T=T, option_type=OptionType.CALL)
    res_asian = pricer.compute_greeks(asian_opt, n_paths=100_000, n_steps=400, seed=42)
    
    print(f"{'Metric':<10} | {'Value':<12} | {'Note'}")
    print("-" * 45)
    print(f"{'Price':<10} | {res_asian['price']:<12.4f} |")
    print(f"{'Delta':<10} | {res_asian['delta']:<12.4f} |")
    print(f"{'Gamma':<10} | {res_asian['gamma']:<12.4f} |")
    print(f"{'Vega':<10} | {res_asian['vega_v0']:<12.4f} |")
    print("-" * 45)

if __name__ == "__main__":
    main()