from datetime import datetime, timedelta
from pymongo import MongoClient
import os

from functions.avesprat import avesprat
from functions.daysSinceLastRace import daysSinceLastRace
from functions.jocMisData import jocMisData
from functions.jocWinCount import jocWinCount
from functions.lifeWin import lifeWin
from functions.newDist import newDist
from functions.getRaces import getRaces
from generateFeatures import generateFeaturesForPastRaces
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

import joblib
import json


# Example usage:
dbUri = os.getenv("MONGO_URI")
client = MongoClient(dbUri)
db = client["hkjc"]

# Retrieve the 50 races you want to analyze (you may already have these from the previous code)
races = getRaces(db)

# Calculate stats for each horse in the raceResults
races_with_stats = generateFeaturesForPastRaces(db, races)

flattened_results = []
for race in races_with_stats:
    for result in race['results']:
        flattened_result = {**race, **result}
        flattened_results.append(flattened_result)


# Convert to a pandas DataFrame for better view
df = pd.DataFrame(flattened_results)
#df.to_csv("races_with_stats.csv", index=False, encoding="utf-8")

drop_cols= ["race", "ratingLower", "ratingUpper", "date", "money", "horseNumber","rawlbw","runningPosition", "horseCode", "jockey", "trainer", "finishTime", "lbw", "results"]
df = df.drop(columns=drop_cols)
categorical_cols = ['racecourse', 'venue', 'racetrack', 'going', 'class', "sex"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=True)

y = df["place"]  # Target variable
X = df.drop(columns=["place"])  # Features (after encoding)

# Check if any columns in X contain NaN values
nan_columns = X.isna().any()
print("NAN columns:")
# Print the columns that contain NaN values
print(nan_columns)

for col in df.columns:
    mask = df[col].apply(lambda x: isinstance(x, dict))
    if mask.any():
        print(f"Column '{col}' has dicts in these rows:")
        print(df[mask][[col]])

# Standardize numeric features (optional, but improves convergence)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Target variable
y = df["place"].astype(int)  

# Split Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# Standardize Features (Optional but helps optimization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Train Multinomial Logistic Regression Model
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
model.fit(X_train, y_train)
# # Make Predictions

y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print(classification_report(y_test, y_pred))

# Convert coefficients into a DataFrame
coefficients = pd.DataFrame(model.coef_, columns=X.columns, index=[f"Place_{i}" for i in model.classes_])

# Print the coefficients
print(coefficients)


# Create a timestamped filename

# Create timestamp and folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_folder = f"models/multinomial_logistic_model_{timestamp}"
os.makedirs(model_folder, exist_ok=True)

# === Save the model ===
model_path = os.path.join(model_folder, "model.pkl")
joblib.dump(model, model_path)

# === Save metadata ===
metadata = {
    "timestamp": timestamp,
    "accuracy": accuracy_score(y_test, y_pred),
    "model_type": "Multinomial Logistic Regression",
    "solver": "lbfgs",
    "max_iter": 500,
    "target_variable": "place"
}
with open(os.path.join(model_folder, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# === Save feature names ===
with open(os.path.join(model_folder, "features.txt"), "w") as f:
    for col in X.columns:
        f.write(col + "\n")

print(f"Model and metadata saved in: {model_folder}")
