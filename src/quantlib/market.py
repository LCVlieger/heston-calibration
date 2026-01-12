from dataclasses import dataclass

@dataclass
class MarketEnvironment:
    S0: float    # Spot Price
    r: float     # Risk-free Rate
    sigma: float # Volatility