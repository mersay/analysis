import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Step 1: Load data ===
filename="race_data_2024_100_20250422.csv"
filepath = os.path.join("pastRacePredictions", filename)
df = pd.read_csv(filepath)

# === Step 2: Add race_id and is_winner columns ===
df['race_id'] = df['date'].astype(str) + "_" + df['race'].astype(str)
df['is_winner'] = (df['place'] == 1).astype(int)

# === Step 3: Bin predicted model probabilities ===
bins = [0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0]
labels = [
    '.000-.010', '.010-.025', '.025-.050', '.050-.100', '.100-.150',
    '.150-.200', '.200-.250', '.250-.300', '.300-.400', '>.400'
]
df['bin'] = pd.cut(df['model_prob'], bins=bins, labels=labels, include_lowest=True)

# === Step 4: Group and calculate summary ===
summary = df.groupby('bin').agg(
    n=('model_prob', 'count'),
    exp=('model_prob', 'mean'),
    act=('is_winner', 'mean')
).reset_index()

# === Step 5: Compute Z-scores ===
summary['Z'] = (summary['act'] - summary['exp']) / np.sqrt((summary['exp'] * (1 - summary['exp'])) / summary['n'])

# === Step 6: Print the calibration table ===
print("\nModel Probability Calibration Table:")
print(summary.to_string(index=False))

# === Step 7: Plot calibration curve ===
plt.figure(figsize=(8, 6))
plt.plot(summary['exp'], summary['act'], marker='o', label='Actual vs Expected')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.xlabel('Expected Win Probability')
plt.ylabel('Actual Win Rate')
plt.title('Model Calibration Plot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()