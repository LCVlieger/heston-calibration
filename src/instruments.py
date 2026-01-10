from abc import ABC, abstractmethod
import numpy as np

class Option(ABC):
    """
    Base Class for options.
    Forces each option (Asian, Barrier, etc.)
    to have a payoff method. 
    """
    def __init__(self, K: float, T: float): # Initializes the option 
        self.K = K  # Strike price
        self.T = T  # Time to maturity (in years)

    @abstractmethod
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute payoff given a price path.
        prices shape: (N_paths, N_timesteps)
        """
        pass
    

class EuropeanCallOption(Option):
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        # For an European call option, only look at price at expiration (final column: -1)
        S_T = prices[:, -1] 
        # Vectorized payoff: max(S_T - K, 0)
        return np.maximum(S_T - self.K, 0)

class EuropeanPutOption(Option):
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        S_T = prices[:, -1]
        return np.maximum(self.K - S_T, 0)