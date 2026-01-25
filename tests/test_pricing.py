import pytest
import numpy as np
from heston_pricer.market import MarketEnvironment
from heston_pricer.instruments import EuropeanOption, AsianOption, OptionType
from heston_pricer.models.process import BlackScholesProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer
from heston_pricer.analytics import BlackScholesPricer

@pytest.fixture
def default_market():
    """Standard market environment for all tests."""
    # Note: We must ensure q=0 because the basic BS analytical pricer 
    # in this repo currently doesn't support q, but the MC engine does.
    return MarketEnvironment(S0=100, r=0.05, q=0.0, sigma=0.2)

def test_european_call_convergence(default_market):
    """
    Test 1: Does Monte Carlo (Black-Scholes Mode) converge to exact BS price?
    Tolerance: < 0.05.
    """
    # 1. Setup
    T, K = 1.0, 100
    option = EuropeanOption(K=K, T=T, option_type=OptionType.CALL)
    
    # FIX: Initialize Process, then Pricer
    process = BlackScholesProcess(default_market)
    pricer = MonteCarloPricer(process)
    
    # 2. Execution 
    # FIX: Use .price() and access .price attribute
    # We use fewer paths for unit tests speed (100k is enough for 0.05 tolerance)
    result = pricer.price(option, n_paths=200_000, n_steps=100)
    mc_price = result.price
    
    bs_price = BlackScholesPricer.price_european_call(
        default_market.S0, K, T, default_market.r, default_market.sigma
    )
    
    # 3. Validation
    error = abs(mc_price - bs_price)
    print(f"\nEuropean Error: {error:.4f} | MC: {mc_price:.2f} | BS: {bs_price:.2f}")
    assert error < 0.05, f"MC Price {mc_price} deviating from Black-Scholes {bs_price}"

def test_asian_call_approximation(default_market):
    """
    Test 2: Does Monte Carlo align with the Turnbull-Wakeman Approximation?
    Tolerance: < 0.20 (Approximations are not exact).
    """
    # 1. Setup
    T, K = 1.0, 100
    option = AsianOption(K=K, T=T, option_type=OptionType.CALL)
    
    process = BlackScholesProcess(default_market)
    pricer = MonteCarloPricer(process)
    
    # 2. Execution
    result = pricer.price(option, n_paths=200_000, n_steps=100)
    mc_price = result.price
    
    tw_price = BlackScholesPricer.price_asian_arithmetic_approximation(
        default_market.S0, K, T, default_market.r, default_market.sigma
    )
    
    # 3. Validation
    error = abs(mc_price - tw_price)
    print(f"Asian Error: {error:.4f} | MC: {mc_price:.2f} | TW: {tw_price:.2f}")
    assert error < 0.20, f"Asian MC {mc_price} diverged from TW Approx {tw_price}"

def test_put_call_parity(default_market):
    """
    Test 3: Put-Call Parity Consistency Check.
    Call - Put = S - K * exp(-rT)
    """
    T, K = 1.0, 100
    call = EuropeanOption(K, T, OptionType.CALL)
    put = EuropeanOption(K, T, OptionType.PUT)
    
    process = BlackScholesProcess(default_market)
    pricer = MonteCarloPricer(process)
    
    # Use same random seed implicitly or run enough paths to average out
    c_price = pricer.price(call, n_paths=100_000, n_steps=100).price
    p_price = pricer.price(put, n_paths=100_000, n_steps=100).price
    
    # Theoretical Parity
    discounted_k = K * np.exp(-default_market.r * T)
    lhs = c_price - p_price
    rhs = default_market.S0 - discounted_k
    
    diff = abs(lhs - rhs)
    print(f"Parity Diff: {diff:.4f}")
    assert diff < 0.15, f"Put-Call Parity violated by {diff:.4f}"