import numpy as np
from quantlib.market import MarketEnvironment
from quantlib.instruments import BarrierOption, BarrierType, OptionType
from quantlib.models.process import HestonProcess, BlackScholesProcess
from quantlib.models.mc_pricer import MonteCarloPricer
from quantlib.calibration import HestonCalibrator, MarketOption
from quantlib.analytics import HestonAnalyticalPricer

def main():
    print("=== QUANTLIB: Full Cycle Pricing (Calibration -> Exotic) ===\n")

    # 1. The Hidden Truth (Market Data Generation)
    S0, r = 100.0, 0.03
    # True Params: High Crash Risk (rho = -0.7)
    true_params = {'kappa': 1.5, 'theta': 0.04, 'xi': 0.5, 'rho': -0.7, 'v0': 0.04}
    
    strikes = [90, 95, 100, 105, 110]
    maturity = 1.0
    
    market_options = []
    print("[1] Reading Market Data (Liquid Vanillas)...")
    for K in strikes:
        price = HestonAnalyticalPricer.price_european_call(S0, K, maturity, r, **true_params)
        market_options.append(MarketOption(K, maturity, price))
        print(f"    Strike={K:<3} Price={price:.4f}")

    # 2. Calibration
    print("\n[2] Calibrating Heston Model...")
    calibrator = HestonCalibrator(S0, r)
    initial_guess = [1.0, 0.04, 0.1, 0.0, 0.04]
    
    res = calibrator.calibrate(market_options, init_guess=initial_guess)
    
    if not res['success']:
        print("Calibration Failed!")
        return

    print(f"    Recovered Skew (rho):    {res['rho']:.4f}")
    print(f"    Recovered Vol-Vol (xi):  {res['xi']:.4f}")

    # 3. Pricing the Exotic (Down-and-Out Call)
    print("\n[3] Pricing Exotic Product (Down-and-Out Call)...")
    print("    Product: Strike=100, Barrier=85 (Knock-Out)")
    
    barrier_opt = BarrierOption(K=100, T=1.0, barrier=85, barrier_type=BarrierType.DOWN_AND_OUT, option_type=OptionType.CALL)
    
    # --- A. HESTON PRICE (The "Smart" Price) ---
    env_heston = MarketEnvironment(
        S0=S0, r=r, 
        kappa=res['kappa'], theta=res['theta'], xi=res['xi'], rho=res['rho'], v0=res['v0']
    )
    # Using HestonProcess (defaults to QE/Truncated Euler)
    process_heston = HestonProcess(env_heston) 
    pricer_heston = MonteCarloPricer(process_heston)
    
    res_heston = pricer_heston.price(barrier_opt, n_paths=200_000, n_steps=200)

    # --- B. BLACK-SCHOLES PRICE (The "Naive" Price) ---
    # We use the ATM Volatility from the Heston model (sqrt(theta) or sqrt(v0))
    # Since theta=0.04 and v0=0.04, the implied vol is 20%
    sigma_bs = np.sqrt(res['theta']) 
    
    env_bs = MarketEnvironment(S0=S0, r=r, sigma=sigma_bs)
    process_bs = BlackScholesProcess(env_bs) # Same architecture, different physics!
    pricer_bs = MonteCarloPricer(process_bs)
    
    res_bs = pricer_bs.price(barrier_opt, n_paths=200_000, n_steps=200)
    
    print(f"\n=== FINAL PRICING COMPARISON ===")
    print(f"Heston Price:        {res_heston.price:.4f}  (Skew Accounted For)")
    print(f"Black-Scholes Price: {res_bs.price:.4f}  (Assuming Flat Vol)")
    
    diff = res_bs.price - res_heston.price
    print(f"\n[Model Impact]")
    print(f"Difference:          {diff:.4f}")
    print("-" * 30)
    if diff > 0:
        print("Black-Scholes OVERPRICES this product.")
        print("Why? Heston knows that if the market drops, Volatility SPIKES (rho=-0.7).")
        print("This makes hitting the barrier (85) much more likely than BS predicts.")
        print("Result: The option knocks out more often -> It is worth LESS.")
    else:
        print("Black-Scholes UNDERPRICES this product.")

if __name__ == "__main__":
    main()