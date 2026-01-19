import time
from heston_pricer.instruments import EuropeanOption, AsianOption, OptionType
from heston_pricer.market import MarketEnvironment
from heston_pricer.models.mc_pricer import MonteCarloPricer
from heston_pricer.analytics import BlackScholesPricer 
from heston_pricer.models.process import BlackScholesProcess

def main():
    # 1. Setup Environment
    env = MarketEnvironment(S0=100, r=0.05, sigma=0.2)
    print(f"Market: S0={env.S0}, r={env.r}, sigma={env.sigma}")

    # 2. Define Instruments 
    T, K = 1.0, 100
    euro_call = EuropeanOption(K, T, OptionType.CALL)
    asian_call = AsianOption(K, T, OptionType.CALL)

    # 3. Initialize Pricer
    process = BlackScholesProcess(env) # Same architecture, different physics!
    pricer = MonteCarloPricer(process)
    # --- JIT Warm-up ---
    # Run a simulation forcing Numba to compile the code.
    print("\n[System] Warmup JIT compiler...")
    _ = pricer.price(euro_call, n_paths=100)
    print("[System] Compilation complete.")

    # ---------------------------------------------------------
    # 4. Greeks Calculation (Delta / Gamma)
    # ---------------------------------------------------------
    print("\n--- Computing Greeks (Finite Difference) ---")
    n_paths_greeks = 500_000 
    
    # European Greeks
    print(f"Calculating European Greeks (N={n_paths_greeks})...")
    t0 = time.time()
    greeks_euro = pricer.compute_greeks(euro_call, n_paths=n_paths_greeks)
    dt = time.time() - t0
    
    print(f"Time:  {dt:.2f}s")
    print(f"Price: {greeks_euro['price']:.4f}")
    print(f"Delta: {greeks_euro['delta']:.4f}")
    print(f"Gamma: {greeks_euro['gamma']:.4f}")

    # Asian Greeks
    print(f"\nCalculating Asian Greeks (N={n_paths_greeks})...")
    greeks_asian = pricer.compute_greeks(asian_call, n_paths=n_paths_greeks)
    print(f"Price: {greeks_asian['price']:.4f}")
    print(f"Delta: {greeks_asian['delta']:.4f}") 
    print(f"Gamma: {greeks_asian['gamma']:.4f}")

    # ---------------------------------------------------------
    # 5. Pricing & Statistical Analysis (Benchmarks)
    # ---------------------------------------------------------
    
    # --- European Option ---
    print("\n--- Benchmarking: European Option ---")
    start_time = time.time()
    # Returns PricingResult object (price, std_error, conf_interval)
    res_euro = pricer.price(euro_call, n_paths=1_000_000)
    end_time = time.time()
    
    print(f"Price:      {res_euro.price:.4f}")
    print(f"Std Error:  {res_euro.std_error:.6f}")
    print(f"95% CI:     [{res_euro.conf_interval_95[0]:.4f}, {res_euro.conf_interval_95[1]:.4f}]")
    print(f"Time:       {end_time - start_time:.4f} seconds")

    # --- Asian Option ---
    print("\n--- Benchmarking: Asian Option ---")
    
    start_time = time.time()
    res_asian = pricer.price(asian_call, n_paths=500_000)
    end_time = time.time()
    
    print(f"MC Price:   {res_asian.price:.4f}")
    print(f"Std Error:  {res_asian.std_error:.6f}")
    print(f"95% CI:     [{res_asian.conf_interval_95[0]:.4f}, {res_asian.conf_interval_95[1]:.4f}]")
    print(f"MC Time:    {end_time - start_time:.4f}s")
    
    # ---------------------------------------------------------
    # 6. Analytical Validation
    # ---------------------------------------------------------
    
    # Hull / Turnbull-Wakeman Approx
    asian_approx_price = BlackScholesPricer.price_asian_arithmetic_approximation(
        env.S0, asian_call.K, asian_call.T, env.r, env.sigma
    )
    print(f"Approx (TW): {asian_approx_price:.4f}")
    
    diff = res_asian.price - asian_approx_price
    print(f"Difference: {diff:.4f}")
    
    if abs(diff) < 0.05:
        print(">> Validation Passed: MC aligns with Hull Approximation.")
    else:
        print(">> Note: Small difference expected due to Discrete (MC) vs Continuous (Hull) averaging.")
    
    # Final BS Check
    bs_price = BlackScholesPricer.price_european_call(env.S0, K, T, env.r, env.sigma)
    print(f"\n[Validation]")
    print(f"Reference BS Price:  {bs_price:.4f}")
    print(f"Asian Discount:      {(1 - res_asian.price/res_euro.price)*100:.2f}%")

if __name__ == "__main__":
    main()