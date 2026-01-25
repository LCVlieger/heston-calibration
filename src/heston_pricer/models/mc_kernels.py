import numpy as np
from numba import jit

# --- KERNELS ---

@jit(nopython=True, cache=True, fastmath=True)
def generate_paths_kernel(S0: float, r: float, q: float, sigma: float, 
                          T: float, n_paths: int, n_steps: int) -> np.ndarray:
    """
    Standard Geometric Brownian Motion (Black-Scholes) kernel with Antithetic Sampling.
    """
    dt = T / n_steps
    half_paths = n_paths // 2
    
    # Variance reduction: Antithetic variates [Z, -Z]
    Z = np.concatenate((np.random.standard_normal((half_paths, n_steps)), 
                        -np.random.standard_normal((half_paths, n_steps))), axis=0)
    
    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    for i in range(n_paths):
        log_ret = 0.0
        for j in range(n_steps):
            log_ret += drift + diff * Z[i, j]
            prices[i, j + 1] = S0 * np.exp(log_ret)
            
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    """
    Heston Model simulation using Full Truncation Euler discretization.
    Ref: Lord, R., Koekkoek, R., & Van Dijk, D. (2010). "A comparison of biased simulation schemes..."
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        Zv = c1 * Z1 + c2 * Z2
        
        # Full Truncation: Treat negative variance as 0 in drift/diffusion, 
        # but update the underlying process normally.
        v_pos = np.maximum(curr_v, 0.0)
        curr_v += kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * Zv
        
        v_pos = np.maximum(curr_v, 0.0)
        curr_s *= np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Z1)
        prices[:, j + 1] = curr_s
        
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths_crn(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps, noise_matrix):
    """
    Calibration-specific kernel supporting Common Random Numbers (CRN).
    Accepts pre-generated noise matrix to guarantee gradient stability.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1 = noise_matrix[0, j]
        Z2 = noise_matrix[1, j]
        Zv = c1 * Z1 + c2 * Z2
        
        v_pos = np.maximum(curr_v, 0.0)
        curr_v += kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * Zv
        
        v_pos = np.maximum(curr_v, 0.0)
        curr_s *= np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Z1)
        prices[:, j + 1] = curr_s
        
    return prices