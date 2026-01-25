import json
import glob
import os
import numpy as np
import pandas as pd
from heston_pricer.market import MarketEnvironment
from heston_pricer.models.process import HestonProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer
from heston_pricer.instruments import BarrierOption, BarrierType, AsianOption, OptionType

""" Use market-retreived calibrated parameters from the Monte Carlo pricer 
of 1. to price exotics (barrier & asian). """


# Load calibration from 1 (the MC parameters). 
def load_calibration():
    patterns = ['calibration_*_meta.json', 'examples/calibration_*_meta.json', '../calibration_*_meta.json']
    files = []
    for p in patterns: files.extend(glob.glob(p))
    if not files: raise FileNotFoundError("No calibration meta file found.")
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as f: return json.load(f)

# Compute prices and greeks 
def compute(pricer, option, name):
    result = pricer.compute_greeks(option, n_paths=100_000, n_steps=400, seed=42)
    return {
        "Product": name, "Price": result['price'], 
        "Delta": result['delta'], "Gamma": result['gamma'], "Vega": result['vega_v0']
    }


# price asian and barrier options. 
def main():
    try:
        data = load_calibration()
    except Exception as e:
        print(f"[FATAL] {e}")
        return

    p = data['monte_carlo_results']
    m = data['market']
    env = MarketEnvironment(m['S0'], m['r'], m['q'], p['kappa'], p['theta'], p['xi'], p['rho'], p['v0'])
    pricer = MonteCarloPricer(HestonProcess(env))
    
    # define option parameters
    S0, K = m['S0'], m['S0'] * 1.05
    B = S0 * 0.80

    # run 
    print(f"[{pd.Timestamp.now().time()}] Pricing Exotic Structures (S0={S0:.2f})...")
    df = pd.DataFrame([
        compute(pricer, BarrierOption(K, 1.0, B, BarrierType.DOWN_AND_OUT, OptionType.CALL), "Down-Out Call"),
        compute(pricer, AsianOption(K, 1.0, OptionType.CALL), "Asian Call")
    ])
    
    print(df.set_index("Product").to_string(float_format="{:.4f}".format))

if __name__ == "__main__":
    main()