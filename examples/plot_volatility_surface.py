import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import brentq
from scipy.stats import norm
from quantlib.analytics import HestonAnalyticalPricer

def impl_vol(price, S0, K, T, r):
    """ Inverts Black-Scholes to find Implied Vol """
    if price <= 0: return 0.0
    def obj(sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return bs_price - price
    try:
        return brentq(obj, 0.001, 5.0)
    except:
        return np.nan

def main():
    print("Generating 3D Volatility Surface...")

    # 1. Calibrated Parameters (The "DNA" of the market)
    S0, r = 100.0, 0.03
    params = {'kappa': 1.5, 'theta': 0.04, 'xi': 0.56, 'rho': -0.71, 'v0': 0.04}
    
    # 2. Define the Grid (Moneyness x Time)
    # Moneyness = K / S0 (80% to 120%)
    moneyness = np.linspace(0.8, 1.2, 30) 
    strikes = S0 * moneyness
    
    # Maturities (0.25 years to 3.0 years)
    maturities = np.linspace(0.25, 3.0, 30)
    
    # Create Meshgrid for Plotting
    X, Y = np.meshgrid(strikes, maturities)
    Z = np.zeros_like(X)
    
    # 3. Calculate Implied Vol for every point on the grid
    print("Computing surface points...")
    for i in range(len(maturities)):
        for j in range(len(strikes)):
            T = maturities[i]
            K = strikes[j]
            
            # Heston Price
            price = HestonAnalyticalPricer.price_european_call(S0, K, T, r, **params)
            
            # Convert to Implied Vol
            iv = impl_vol(price, S0, K, T, r)
            Z[i, j] = iv * 100 # Convert to %

    # 4. Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Surface
    # cmap='viridis' is standard for heatmaps
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)
    
    # Labels
    ax.set_xlabel('Strike Price ($K$)')
    ax.set_ylabel('Maturity ($T$)')
    ax.set_zlabel('Implied Volatility (%)')
    ax.set_title(f'Heston Volatility Surface\n(rho={params["rho"]}, xi={params["xi"]})')
    
    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Rotate for better view
    ax.view_init(elev=30, azim=-120)
    
    filename = "volatility_surface.png"
    plt.savefig(filename)
    print(f"Surface saved to {filename}")
    plt.show()

if __name__ == "__main__":
    main()