import time
import numpy as np
import pandas as pd
import math
from heston_pricer.models.mc_kernels import generate_heston_paths
from heston_pricer.analytics import HestonAnalyticalPricer

def heston_pure_python(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    c1 = rho
    c2 = math.sqrt(1 - rho**2)
    
    final_prices = []
    for _ in range(n_paths):
        s_t = S0
        v_t = v0
        for _ in range(n_steps):
            z1 = np.random.normal()
            z2 = np.random.normal()
            zv = c1 * z1 + c2 * z2
            
            v_positive = max(v_t, 0.0)
            dv = kappa * (theta - v_positive) * dt + xi * math.sqrt(v_positive) * sqrt_dt * zv
            v_t += dv
            
            vol_t = math.sqrt(v_positive)
            drift = (r - q - 0.5 * v_positive) * dt
            diffusion = vol_t * sqrt_dt * z1
            s_t *= math.exp(drift + diffusion)
        final_prices.append(s_t)
    return np.array(final_prices)

def heston_numpy_vectorized(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    v_t = np.full(n_paths, v0)
    
    Z1 = np.random.normal(size=(n_steps, n_paths))
    Zv = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=(n_steps, n_paths))
    
    s_t = np.full(n_paths, S0)
    
    for t in range(n_steps):
        v_pos = np.maximum(v_t, 0.0)
        sq_v = np.sqrt(v_pos)
        dv = kappa * (theta - v_pos) * dt + xi * sq_v * sqrt_dt * Zv[t]
        v_t += dv
        s_t *= np.exp((r - q - 0.5 * v_pos) * dt + sq_v * sqrt_dt * Z1[t])
    return s_t

def benchmark_execution(func, *args, **kwargs):
    t0 = time.time()
    func(*args, **kwargs)
    return time.time() - t0

def main():
    S0, r, q, T = 100.0, 0.05, 0.0, 1.0
    v0, kappa, theta, xi, rho = 0.04, 1.0, 0.04, 0.5, -0.7
    n_paths = 2_000_000
    n_steps = 252
    
    print(f"Benchmarking (Paths={n_paths}, Steps={n_steps})...")
    
    # 1. Python (Extrapolated)
    n_py = 50_000
    t_py_raw = benchmark_execution(heston_pure_python, S0, r, q, v0, kappa, theta, xi, rho, T, n_py, n_steps)
    t_py = t_py_raw * (n_paths / n_py)
    
    # 2. NumPy
    t_np = benchmark_execution(heston_numpy_vectorized, S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps)
    
    # 3. Numba (JIT)
    generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, 10, n_steps) # Warmup
    t_numba = benchmark_execution(generate_heston_paths, S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps)

    df = pd.DataFrame([
        {"Engine": "Pure Python", "Time (s)": t_py, "Speedup (vs Py)": 1.0},
        {"Engine": "NumPy Vectorized", "Time (s)": t_np, "Speedup (vs Py)": t_py/t_np},
        {"Engine": "Numba JIT", "Time (s)": t_numba, "Speedup (vs Py)": t_py/t_numba}
    ])
    
    print("\nExecution Metrics:")
    print(df.to_string(float_format="{:.2f}".format))

    # Consistency Check
    K = 100
    p_ana = HestonAnalyticalPricer.price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0)
    paths = generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, 200_000, 100)
    p_mc = np.mean(np.maximum(paths[:, -1] - K, 0)) * np.exp(-r*T)
    
    print("\nNumerical Convergence:")
    print(pd.Series({"Analytical": p_ana, "Monte Carlo": p_mc, "Abs Error": abs(p_ana - p_mc)}).to_string(float_format="{:.4f}".format))

if __name__ == "__main__":
    main()