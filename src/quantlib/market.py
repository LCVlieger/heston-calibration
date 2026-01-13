from dataclasses import dataclass

@dataclass
class MarketEnvironment:
    S0: float    # Spot Price
    r: float     # Risk-free Rate
    sigma: float = 0.2  # for Black-Scholes
    # Heston Parameters (Defaults set to mimic Black-Scholes if not used)
    v0: float = 0.04      # Initial Variance (e.g., 20%^2 = 0.04)
    kappa: float = 1.0    # Mean Reversion Speed
    theta: float = 0.04   # Long-Run Variance
    xi: float = 0.1       # Vol of Vol (0.0 = Constant Vol)
    rho: float = -0.7     # Correlation (Stock vs Vol)