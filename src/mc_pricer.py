import numpy as np
from src.instruments import Option

class MonteCarloPricer:
    def __init__(self, r: float, sigma: float):
        """
        r: Risk-free rate
        sigma: Volatility
        """
        self.r = r
        self.sigma = sigma

    def simulate_paths(self, S0: float, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        
        # Make n_paths even
        if n_paths % 2 != 0:
            n_paths += 1
            
        half_paths = n_paths // 2
        
        # Generate random shocks
        Z_half = np.random.standard_normal((half_paths, n_steps))
        
        # Generate Z and -Z  (both valid paths, equal probability)
        Z = np.concatenate((Z_half, -Z_half), axis=0)
        
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        
        log_returns = np.cumsum(drift + diffusion, axis=1)
        
        prices = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = S0
        prices[:, 1:] = S0 * np.exp(log_returns)
        
        return prices

    def price_option(self, option: Option, S0: float, n_paths: int = 10000, n_steps: int = 100) -> float:
        """
        Price each option from 'Option' class.
        """
        # 1. Generate paths (until option.T)
        paths = self.simulate_paths(S0, option.T, n_paths, n_steps)
        
        # 2. Compute payoffs (use option payoff method)
        payoffs = option.payoff(paths)
        
        # 3. Discount to today
        avg_payoff = np.mean(payoffs)
        discounted_price = avg_payoff * np.exp(-self.r * option.T)
        
        return discounted_price