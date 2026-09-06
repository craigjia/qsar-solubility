# qsar_solubility.py

import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ==========================================
# 1. Read CSV
# ==========================================

df = pd.read_csv("solubility.csv")

print("Original data shape:", df.shape)


# ==========================================
# 2. Remove missing values
# ==========================================

df = df.dropna(subset=["smiles", "logS"])

print("After removing missing values:", df.shape)


# ==========================================
# 3. SMILES -> RDKit Mol
# ==========================================

df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)

# Remove invalid SMILES
df = df[df["mol"].notna()].copy()

print("After removing invalid SMILES:", df.shape)


# ==========================================
# 4. Morgan fingerprint function
# ==========================================

def make_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )


# Generate fingerprints
fps = df["mol"].apply(make_fingerprint)


# ==========================================
# 5. Fingerprints -> X
# ==========================================

X = np.array([np.array(fp) for fp in fps])

y = df["logS"].values

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==========================================
# 6. Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# ==========================================
# 7. Create Random Forest model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 8. Train model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 9. Predictions
# ==========================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ==========================================
# 10. Calculate metrics
# ==========================================

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)


print("\n===== Model Performance =====")

print("Train R² :", round(train_r2, 3))
print("Test R²  :", round(test_r2, 3))
print("Test RMSE:", round(test_rmse, 3))
print("Test MAE :", round(test_mae, 3))


# ==========================================
# 11. Predict a new molecule
# ==========================================

new_smiles = "CCOC(=O)C"

new_mol = Chem.MolFromSmiles(new_smiles)

if new_mol is None:
    raise ValueError("Invalid SMILES for new molecule.")

new_fp = make_fingerprint(new_mol)

X_new = np.array(new_fp).reshape(1, -1)

predicted_logS = model.predict(X_new)[0]

print("\n===== New Molecule Prediction =====")

print("SMILES:", new_smiles)
print("Predicted logS:", round(predicted_logS, 3))


# ==========================================
# 12. Save model
# ==========================================

joblib.dump(model, "solubility_rf.pkl")

print("\nModel saved as: solubility_rf.pkl")
