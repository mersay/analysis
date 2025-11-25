import joblib
import pandas as pd
import os
import numpy as np

# === Config ===
MAX_RUNNERS = 14
TOP_N = 500  # how many top coefficients to show

# === Paths (adjust folder name as needed) ===
model_folder = "models/multinomial_logistic_model_20250420_132913"  # <- replace with your actual timestamped folder
model_path = os.path.join(model_folder, "model.pkl")
runner_encoder_path = os.path.join(model_folder, "runner_encoder.pkl")
race_encoder_path = os.path.join(model_folder, "race_encoder.pkl")

# === Load model and encoders ===
model = joblib.load(model_path)
runner_encoder = joblib.load(runner_encoder_path)
race_encoder = joblib.load(race_encoder_path)

# === Get feature names from encoders ===
runner_feature_names = runner_encoder.get_feature_names_out()
race_feature_names = race_encoder.get_feature_names_out()
combined_features = list(runner_feature_names) + list(race_feature_names)

# === Flatten feature names across runners ===
all_feature_names = []
for i in range(MAX_RUNNERS):
    for name in combined_features:
        all_feature_names.append(f"R{i}_{name}")

# === Get model coefficients ===
coef_df = pd.DataFrame(model.coef_, columns=all_feature_names)

# === Aggregate feature importances
# For multinomial, model.coef_ is shape (num_classes, num_features)
# We take the average absolute importance across all classes
importance = np.mean(np.abs(model.coef_), axis=0)

# === Create dataframe and sort
importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importance
})
importance_df = importance_df.sort_values(by="importance", ascending=False)

# === Print top N
print(f"\nTop {TOP_N} Most Important Features:")
print(importance_df.head(TOP_N).to_string(index=False))

# === Save to CSV ===
coef_output_path = os.path.join(model_folder, "coefficients.csv")
coef_df.to_csv(coef_output_path, index=False)

print(f"[✓] Coefficients saved to: {coef_output_path}")
