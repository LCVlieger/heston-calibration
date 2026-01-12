import numpy as np

# Week 2 Note: This is where we will add @jit later
def generate_paths_kernel(S0: float, r: float, sigma: float, 
                          T: float, n_paths: int, n_steps: int) -> np.ndarray:
    dt = T / n_steps
    
    # Antithetic Variates for Variance Reduction
    if n_paths % 2 != 0:
        n_paths += 1
    
    half_paths = n_paths // 2
    Z_half = np.random.standard_normal((half_paths, n_steps))
    Z = np.concatenate((Z_half, -Z_half), axis=0)
    
    # Vectorized Path Generation
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = np.cumsum(drift + diffusion, axis=1)
    
    # Construct Price Matrix
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    prices[:, 1:] = S0 * np.exp(log_returns)
    
    return prices