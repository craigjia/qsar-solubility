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
- `solubility.csv` - input dataset
- `requirements.txt` - Python dependencies

## Example

Run:

```bash
python qsar.py
