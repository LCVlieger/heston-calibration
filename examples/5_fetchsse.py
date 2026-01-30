import numpy as np
import pandas as pd 

df = pd.read_csv("results/calibration_NVDA_20260128_222958_prices.csv")

# Filter out rows where abs error > 100
df_filt = df #df[(df["Err_A"].abs() <= 100) & (df["Err_MC"].abs() <= 100)]
print(np.size(df) - np.size(df_filt ))
# True RMSE (price space)
rmse_A = np.sqrt(np.mean(df_filt["Err_A"]**2))
rmse_MC = np.sqrt(np.mean(df_filt["Err_MC"]**2))

# Relative / weighted SSE (your calibrator objective)
rel_sse_A = np.sum((df_filt["Err_A"]**2 / (1e-5 + df_filt["Mkt"])**2))
rel_sse_MC = np.sum((df_filt["Err_MC"]**2 / (1e-5 + df_filt["Mkt"])**2))

# IV RMSE
rmse_iv_MC = np.sqrt(np.mean((df_filt["IV_MC"] - df_filt["IV_Mkt"])**2))

print("RMSE_A:", rmse_A)
print("RMSE_MC:", rmse_MC)
print("RelSSE_A:", rel_sse_A)
print("RelSSE_MC:", rel_sse_MC)
print("RMSE_IV_MC:", rmse_iv_MC)
