import json
import glob
import os
import numpy as np
import pandas as pd
from heston_pricer.market import MarketEnvironment
from heston_pricer.models.process import HestonProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer
from heston_pricer.instruments import BarrierOption, BarrierType, AsianOption, OptionType

def load_calibration():
    # Locate latest metadata file
    patterns = [
        'calibration_*_meta.json', 
        'examples/calibration_*_meta.json', 
        '../calibration_*_meta.json'
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
        
    if not files:
        raise FileNotFoundError("Calibration metadata not found.")
    
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        return json.load(f)

def run_pricing_engine(pricer, option, label):
    # n_steps=400 ensures barrier discretization stability
    res = pricer.compute_greeks(option, n_paths=100_000, n_steps=400, seed=42)
    return {
        "Instrument": label,
        "Price": res['price'],
        "Delta": res['delta'],
        "Gamma": res['gamma'],
        "Vega": res['vega_v0']
    }

def main():
    try:
        data = load_calibration()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    params = data['monte_carlo_results']
    mkt = data['market']
    
    env = MarketEnvironment(
        S0=mkt['S0'], r=mkt['r'], q=mkt['q'],
        kappa=params['kappa'], theta=params['theta'], 
        xi=params['xi'], rho=params['rho'], v0=params['v0']
    )
    
    process = HestonProcess(env)
    pricer = MonteCarloPricer(process)
    
    # Instruments
    S0, T = mkt['S0'], 1.0
    K = S0 * 1.05
    barrier_level = S0 * 0.80
    
    instruments = [
        (BarrierOption(K, T, barrier_level, BarrierType.DOWN_AND_OUT, OptionType.CALL), 
         f"Down-and-Out Call (K={K:.2f}, B={barrier_level:.2f})"),
        (AsianOption(K, T, OptionType.CALL), 
         f"Arithmetic Asian Call (K={K:.2f})")
    ]

    print(f"Pricing Date: {pd.Timestamp.now()} | Spot: {S0:.2f} | Vol: {np.sqrt(params['v0']):.2%}")
    
    results = [run_pricing_engine(pricer, opt, lbl) for opt, lbl in instruments]
    df = pd.DataFrame(results)
    
    print("\nExotic Pricing & Risk:")
    print(df.set_index("Instrument").to_string(float_format="{:.4f}".format))

if __name__ == "__main__":
    main()