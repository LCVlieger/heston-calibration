from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

class OptionType(Enum):
    CALL = 1
    PUT = -1

class Option(ABC):
    def __init__(self, K: float, T: float, option_type: OptionType):
        self.K = K
        self.T = T
        self.option_type = option_type

    @abstractmethod
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        pass

class EuropeanOption(Option):
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        S_T = prices[:, -1]
        phi = self.option_type.value
        return np.maximum(phi * (S_T - self.K), 0)