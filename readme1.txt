QSAR Solubility Prediction with RDKit and Random Forest
A small QSAR machine-learning project for predicting aqueous solubility from molecular structure.
Project Overview
This project demonstrates a basic cheminformatics and machine-learning workflow:
SMILES → RDKit → Molecular Descriptors → Random Forest → Solubility Prediction
The dataset contains 143 compounds. One compound has an invalid/incomplete SMILES string, so 142 compounds are used for modeling.
Molecular Descriptors
Five RDKit molecular descriptors are calculated from SMILES:
Molecular Weight (MolWt)
LogP
Topological Polar Surface Area (TPSA)
Hydrogen Bond Donors (HBD)
Hydrogen Bond Acceptors (HBA)
Machine Learning Model
A Random Forest Regressor from scikit-learn is used to predict measured log solubility.
The data are divided into:
80% training set
20% test set
A fixed random seed (random_state=42) is used for reproducibility.
Model Performance
On the held-out test set:
RMSE: ~0.62
R²: ~0.93
These results are based on a single random train/test split and should not be interpreted as a robust estimate of generalization performance.
Workflow
Load the solubility dataset from CSV.
Convert SMILES strings into RDKit molecular objects.
Calculate molecular descriptors.
Remove the invalid molecular structure.
Split the data into training and test sets.
Train a Random Forest regression model.
Predict solubility for the test set.
Calculate RMSE and R².
Examine feature importance.
Generate a measured-vs-predicted plot.
How to Run
Install the required packages:
pip install -r requirements.txt
Run the QSAR model:
python qsar.py
The program prints model performance and feature importance and generates:
measured_vs_predicted.png
Future Improvements
Possible next steps include:
Cross-validation
Scaffold-based splitting
Molecular fingerprints
Comparison with linear regression and other ML models
Hyperparameter optimization
Larger solubility datasets
Applicability-domain analysis
External validation
Technologies
Python
pandas
NumPy
RDKit
scikit-learn
Matplotlib
