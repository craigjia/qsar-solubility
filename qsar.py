import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. Load dataset
# ============================================================

#df = pd.read_csv("solubility.csv")

url2 = "https://raw.githubusercontent.com/PatWalters/solubility/master/delaney.csv"
df = pd.read_csv(url2)


print("Dataset shape:", df.shape)


# ============================================================
# 2. Convert SMILES to RDKit molecules
# ============================================================

mols = df["SMILES"].apply(Chem.MolFromSmiles)

# Remove molecules whose SMILES cannot be parsed
valid = ~mols.isna()

mols_valid = mols[valid]

y = df.loc[
    valid,
    "measured log(solubility:mol/L)"
]

print("Valid molecules:", len(mols_valid))


# ============================================================
# 3. Calculate molecular descriptors
# ============================================================

molwt = mols_valid.apply(Descriptors.MolWt)
logp = mols_valid.apply(Descriptors.MolLogP)
tpsa = mols_valid.apply(Descriptors.TPSA)
hbd = mols_valid.apply(Descriptors.NumHDonors)
hba = mols_valid.apply(Descriptors.NumHAcceptors)


# ============================================================
# 4. Build feature matrix X
# ============================================================

X = pd.DataFrame({
    "MolWt": molwt,
    "LogP": logp,
    "TPSA": tpsa,
    "HBD": hbd,
    "HBA": hba
})

print("Feature matrix:", X.shape)


# ============================================================
# 5. Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


# ============================================================
# 6. Train Random Forest model
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained!")


# ============================================================
# 7. Predict test set
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. Evaluate model
# ============================================================

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-----------------")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")


# ============================================================
# 9. Feature importance
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance")
print("------------------")
print(importance)


# ============================================================
# 10. Plot measured vs predicted
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Measured logS")
plt.ylabel("Predicted logS")
plt.title("Measured vs Predicted Solubility")

plt.tight_layout()

plt.savefig(
    "measured_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFigure saved as measured_vs_predicted.png")
