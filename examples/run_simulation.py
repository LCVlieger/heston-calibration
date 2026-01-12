from quantlib.instruments import EuropeanOption, OptionType
from quantlib.market import MarketEnvironment
from quantlib.models.mc_pricer import MonteCarloPricer
from quantlib.analytics import BlackScholesPricer 

def main():
    # 1. Setup Environment
    env = MarketEnvironment(S0=100, r=0.05, sigma=0.2)
    
    # 2. Define Instrument
    call_opt = EuropeanOption(K=100, T=1.0, option_type=OptionType.CALL)

    # 3. Analytical Price
    bs_price = BlackScholesPricer.price_european_call(env.S0, call_opt.K, call_opt.T, env.r, env.sigma)
    print(f"Analytical Price:  {bs_price:.4f}")

    # 4. Monte Carlo Price
    pricer = MonteCarloPricer(env)
    mc_price = pricer.price_option(call_opt, n_paths=100_000)
    print(f"Monte Carlo Price: {mc_price:.4f}")

if __name__ == "__main__":
    main()