import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Assumes heston_pricer package is available in the environment
try:
    from heston_pricer.analytics import HestonAnalyticalPricer
    from heston_pricer.calibration import implied_volatility
except ImportError:
    # Fallback or error if package is missing
    raise ImportError("heston_pricer package not found. Ensure the library is in the PYTHONPATH.")

def plot_surface_standalone():
    # --- 1. HARDCODED PARAMETERS ---
    # Market Constants
    S0 = 100.0       # Normalized spot
    r = 0.045        # Risk-free rate (from previous context)
    q = 0.015        # Dividend yield (from previous context)
    
    # Heston Parameters (Requested)
    kappa = 2.72
    theta = 0.357
    xi = 2.23
    rho = -0.34
    v0 = 0.213

    print(f"Generating surface for: k={kappa}, theta={theta}, xi={xi}, rho={rho}, v0={v0}")

    # --- 2. GRID GENERATION ---
    # Moneyness M = K/S (0.7 to 1.3)
    # Maturity T (0.1 to 2.5 years)
    M_range = np.linspace(0.7, 3.5, 60)
    T_range = np.linspace(0.1, 2.5, 60)
    
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    # --- 3. SURFACE CALCULATION ---
    # Iterating grid to calculate Price -> Implied Volatility
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val = Y[i, j]
            M_val = X[i, j]
            K_val = S0 * M_val
            
            # Calculate Heston Price
            price = HestonAnalyticalPricer.price_european_call(
                S0, K_val, T_val, r, q, kappa, theta, xi, rho, v0
            )
            
            # Invert for Implied Volatility
            try:
                iv = implied_volatility(price, S0, K_val, T_val, r, q)
                Z[i, j] = iv
            except Exception:
                Z[i, j] = np.nan

    # --- 4. VISUALIZATION ---
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Plot the surface with a refined black mesh
        # Using a slightly higher linewidth for the mesh enhances the "Bloomberg" definition
        surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlBu_r, edgecolor='black', 
                               linewidth=0.1, alpha=0.9, antialiased=True)

        # 2. ZOOM AND BOUNDARY FIXES
        # 'dist' is the camera distance. Default is 10. 
        # Increase it to 11 or 12 to "pull back" and prevent clipping.
        ax.dist = 11 
        
        # Explicitly set limits to ensure the grid doesn't expand beyond your data
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(2.5, 0.1)
        
        # 3. LABEL PLACEMENT
        # Increased labelpad prevents text from overlapping with axis tick numbers
        ax.set_title(r"Heston IV Surface $\sigma(T, K/S)$", color='white', y=1.0, fontsize=14, fontweight='bold')
        ax.set_xlabel('Moneyness ($K/S_0$)', color='white', labelpad=15)
        ax.set_ylabel('Maturity ($T$ Years)', color='white', labelpad=15)
        ax.set_zlabel(r'Implied Volatility (%)', color='white', labelpad=15)

        # 4. VIEW PERSPECTIVE
        ax.view_init(elev=28, azim=-138)
        
        # 5. PANE AND GRID STYLING
        # Setting pane colors to fully transparent makes the dark background pop
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Brighten the wall grid for better structural definition
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

        # 6. COLORBAR TIGHTENING
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=14, pad=0.05)
        cbar.ax.yaxis.set_tick_params(color='white')

        # 7. EXPORT FIX
        # 'bbox_inches=tight' with a small 'pad_inches' ensures labels are NOT cut off
        output_file = "heston_surface_standalone.png"
        plt.savefig(output_file, dpi=300, facecolor='black', bbox_inches='tight', pad_inches=0.2)
        plt.show()

if __name__ == "__main__":
    plot_surface_standalone()