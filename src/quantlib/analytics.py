import numpy as np
from scipy.stats import norm

class BlackScholesPricer:
    @staticmethod
    def price_european_call(S0, K, T, r, sigma):
        """
        Analytical price for European Call.
        """
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price