import time
import numpy as np
import pandas as pd
import math
from heston_pricer.models.mc_kernels import generate_heston_paths
from heston_pricer.analytics import HestonAnalyticalPricer


""" Compare run speeds for pure python (loops), numpy and numba path generation. """



# compute paths with regular loops.
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

# compute Heston paths using numpy / vectorized. 
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

# Run against the Numba jit implementation to compare speeds. 
def main():
    S0, r, q, T = 100.0, 0.05, 0.0, 1.0
    v0, kappa, theta, xi, rho = 0.04, 1.0, 0.04, 0.5, -0.7
    N, M = 2_000_000, 252
    
    print(f"[{pd.Timestamp.now().time()}] Benchmarking Kernels (N={N}, Steps={M})")

    # 1. Python (Est)
    t0 = time.time()
    heston_pure_python(S0, r, q, v0, kappa, theta, xi, rho, T, 50_000, M)
    t_py = (time.time() - t0) * (N / 50_000)

    # 2. NumPy
    t0 = time.time()
    heston_numpy_vectorized(S0, r, q, v0, kappa, theta, xi, rho, T, N, M)
    t_np = time.time() - t0

    # 3. Numba
    generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, 10, M) # JIT Warmup
    t0 = time.time()
    generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, N, M)
    t_numba = time.time() - t0

    print(pd.DataFrame([
        {"Engine": "Python (Est)", "Time": t_py, "Rel": 1.0},
        {"Engine": "NumPy", "Time": t_np, "Rel": t_py/t_np},
        {"Engine": "Numba", "Time": t_numba, "Rel": t_py/t_numba}
    ]).set_index("Engine").to_string(float_format="{:.2f}".format))

    # Validation
    p_ana = HestonAnalyticalPricer.price_european_call(S0, 100, T, r, q, kappa, theta, xi, rho, v0)
    paths = generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, 200_000, 100)
    p_mc = np.mean(np.maximum(paths[:, -1] - 100, 0)) * np.exp(-r*T)
    
    print(f"\nConvergence Check: |Ana - MC| = {abs(p_ana - p_mc):.4f}")

if __name__ == "__main__":
    main()