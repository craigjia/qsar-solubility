# QSAR Solubility Prediction

A simple machine-learning project for predicting aqueous solubility from molecular structures.

## Technologies

- Python
- Pandas
- NumPy
- RDKit
- scikit-learn
- Matplotlib
- 
## Workflow

SMILES → RDKit → Molecular Descriptors → Random Forest →Solubility Prediction

1. Load molecular solubility data from CSV
2. Convert SMILES strings into molecular descriptors using RDKit
3. Split compounds into training and test sets
4. Train a machine-learning regression model
5. Predict solubility for held-out compounds
6. Evaluate model performance

## Project Structure

- `qsar.py` - main QSAR program
-`https://raw.githubusercontent.com/PatWalters/solubility/master/delaney.csv"
` - input dataset 1144 molecules
- `requirements.txt` - Python dependencies

## Example

Run:

```bash
python qsar.py
```

## Molecular Descriptors

Five RDKit molecular descriptors are calculated from SMILES:

- Molecular Weight (MolWt)
- LogP
- Topological Polar Surface Area (TPSA)
- Hydrogen Bond Donors (HBD)
- Hydrogen Bond Acceptors (HBA)

## Machine Learning Model

A Random Forest Regressor from scikit-learn is used to predict measured log solubility.

The data are divided into:

- 80% training set

- 20% test set

A fixed random seed (random_state=42) 
is used for reproducibility.

## Model Performance

On the held-out test set:

- RMSE: ~0.64

- R²: ~0.897

These results are based on a single random train/test split and should not be interpreted as a robust estimate of generalization performance.
