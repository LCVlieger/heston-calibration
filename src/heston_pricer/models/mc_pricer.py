import numpy as np
from dataclasses import dataclass, replace
from typing import Dict
from ..instruments import Option
from .process import StochasticProcess

@dataclass
class PricingResult:
    price: float
    std_error: float
    conf_interval_95: tuple[float, float]

class MonteCarloPricer:
    def __init__(self, process: StochasticProcess):
        self.process = process

    def price(self, option: Option, n_paths: int = 10000, n_steps: int = 100, **kwargs) -> PricingResult:
        # Pass kwargs (like 'noise') down to generate_paths
        paths = self.process.generate_paths(option.T, n_paths, n_steps, **kwargs)
        
        # 2. Compute Payoffs
        payoffs = option.payoff(paths)
        
        # 3. Discount
        discount_factor = np.exp(-self.process.market.r * option.T)
        discounted_payoffs = payoffs * discount_factor
        
        # 4. Statistics
        mean_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
        
        return PricingResult(
            price=mean_price,
            std_error=std_error,
            conf_interval_95=(mean_price - 1.96 * std_error, mean_price + 1.96 * std_error)
        )

    # --- UPDATE: Added n_steps argument here ---
    def compute_greeks(self, option: Option, n_paths: int = 10000, n_steps: int = 252, bump_ratio: float = 0.01, seed: int = 42) -> Dict[str, float]:
        """
        Computes Greeks with explicit n_steps control. 
        Default increased to 252 (Daily monitoring) for better barrier accuracy.
        """
        original_market = self.process.market
        original_S0 = original_market.S0
        epsilon_s = original_S0 * bump_ratio
        epsilon_v = 0.001 
        
        # 1. Base Price
        np.random.seed(seed)
        # Pass n_steps explicitly
        res_curr = self.price(option, n_paths, n_steps)
        
        # 2. Delta & Gamma (Spot Bumps)
        self.process.market = replace(original_market, S0 = original_S0 + epsilon_s)
        np.random.seed(seed) 
        res_up = self.price(option, n_paths, n_steps)
        
        self.process.market = replace(original_market, S0 = original_S0 - epsilon_s)
        np.random.seed(seed)
        res_down = self.price(option, n_paths, n_steps)
        
        # 3. Vega (Vol Bump)
        self.process.market = replace(original_market, v0 = original_market.v0 + epsilon_v, S0=original_S0)
        np.random.seed(seed)
        res_vega = self.price(option, n_paths, n_steps)
        
        # Restore Market
        self.process.market = original_market
        
        # Calc
        delta = (res_up.price - res_down.price) / (2 * epsilon_s)
        gamma = (res_up.price - 2 * res_curr.price + res_down.price) / (epsilon_s ** 2)
        vega = (res_vega.price - res_curr.price) / epsilon_v
        
        return {
            "price": res_curr.price,
            "delta": delta,
            "gamma": gamma,
            "vega_v0": vega
        }