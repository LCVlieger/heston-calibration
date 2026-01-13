import numpy as np
from dataclasses import dataclass, replace
from typing import Dict, Optional
from ..instruments import Option
from ..market import MarketEnvironment
from .mc_kernels import generate_paths_kernel

@dataclass
class PricingResult:
    price: float
    std_error: float
    conf_interval_95: tuple[float, float]

class MonteCarloPricer:
    def __init__(self, market: MarketEnvironment):
        self.market = market

    def price_option(self, option: Option, n_paths: int = 10000, n_steps: int = 100) -> PricingResult:
        # 1. Generate Paths
        paths = generate_paths_kernel(
            self.market.S0,
            self.market.r,
            self.market.sigma,
            option.T,
            n_paths,
            n_steps
        )
        
        # 2. Compute Payoffs (Array of size N)
        payoffs = option.payoff(paths)
        discount_factor = np.exp(-self.market.r * option.T)
        discounted_payoffs = payoffs * discount_factor
        
        # 3. Handle Antithetic Statistics
        # We must average pairs (i and i + N/2) to get independent samples
        half_paths = n_paths // 2
        
        # Average the antithetic pairs to create independent estimators
        pair_averages = 0.5 * (discounted_payoffs[:half_paths] + discounted_payoffs[half_paths:])
        
        # 4. Statistics
        mean_price = np.mean(pair_averages)
        std_error = np.std(pair_averages, ddof=1) / np.sqrt(half_paths)
        
        # 95% Confidence Interval (1.96 * SE)
        ci_lower = mean_price - 1.96 * std_error
        ci_upper = mean_price + 1.96 * std_error
        
        return PricingResult(
            price=mean_price,
            std_error=std_error,
            conf_interval_95=(ci_lower, ci_upper)
        )

    def compute_greeks(self, option: Option, n_paths: int = 10000, bump_ratio: float = 0.01, seed: int = 42) -> Dict[str, float]:
        """
        Computes Delta and Gamma using Common Random Numbers (CRN).
        We fix the seed before each run to ensure the paths are identical.
        """
        original_S0 = self.market.S0
        epsilon = original_S0 * bump_ratio
        
        # Up/Down Environments
        market_up = replace(self.market, S0 = original_S0 + epsilon)
        market_down = replace(self.market, S0 = original_S0 - epsilon)
        
        # 1. Central Price (Seed Fixed)
        np.random.seed(seed)
        res_curr = self.price_option(option, n_paths)
        
        # 2. Up Price (Same Seed -> Same Paths)
        self.market = market_up
        np.random.seed(seed) 
        res_up = self.price_option(option, n_paths)
        
        # 3. Down Price (Same Seed -> Same Paths)
        self.market = market_down
        np.random.seed(seed)
        res_down = self.price_option(option, n_paths)
        
        # Restore State
        self.market = replace(self.market, S0 = original_S0)

        # Finite Differences
        delta = (res_up.price - res_down.price) / (2 * epsilon)
        gamma = (res_up.price - 2 * res_curr.price + res_down.price) / (epsilon ** 2)
        
        return {
            "price": res_curr.price,
            "std_error": res_curr.std_error,
            "delta": delta,
            "gamma": gamma
        }