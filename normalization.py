import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate sample data
np.random.seed(42)
original_data = np.random.normal(loc=10, scale=3, size=1000)

# 2. Calculate the mean and standard deviation
mu = np.mean(original_data)
sigma = np.std(original_data)

# 3. Apply Z-score standardization (Normalization)
normalized_data = (original_data - mu) / sigma

# 4. Create the plot
plt.figure(figsize=(12, 5))

# --- Plot 1: Original Data ---
plt.subplot(1, 2, 1)
sns.histplot(original_data, kde=True, color='skyblue', bins=30)
plt.axvline(mu, color='red', linestyle='--', label=f'Mean (μ): {mu:.2f}')
plt.title(f'Original Data (μ={mu:.2f}, σ={sigma:.2f})')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()

# --- Plot 2: Normalized Data ---
plt.subplot(1, 2, 2)
sns.histplot(normalized_data, kde=True, color='salmon', bins=30)
norm_mu = np.mean(normalized_data)
# Add a vertical line at 0 for the normalized mean
plt.axvline(0, color='blue', linestyle='--', label='Mean (μ): 0.00 (approx. 0)')
plt.title(f'Normalized Data (μ={norm_mu:.2f}, σ={np.std(normalized_data):.2f})')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig('normalization_effect_graph.png')