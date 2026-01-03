# coding: utf-8
"""
Predictions sur test.csv en utilisant le meilleur modele
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import mlflow.pyfunc
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PREDICTIONS SUR LE TEST SET")
print("="*70)

# 1. Charger le modele Production
print("\nChargement du modele Production depuis MLflow...")
model = mlflow.pyfunc.load_model("models:/HousePrices-BestModel/Production")
print("Modele charge avec succes!")

# 2. Charger les donnees
print("\nChargement des donnees...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['Id'].copy()

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 3. Preparer test.csv avec les MEMES colonnes que train.csv
print("\nPreparation des donnees de test...")

# Garder toutes les colonnes communes (PAS de suppression de colonnes!)
# Le notebook ne supprime aucune colonne, il remplit juste les NaN
train_cols = [col for col in train.columns if col not in ['SalePrice', 'Id']]
test_cols = [col for col in test.columns if col != 'Id']

# Trouver les colonnes communes
common_cols = [col for col in train_cols if col in test_cols]

# Preparer test avec les colonnes communes
test_filtered = test[common_cols].copy()

# Identifier les colonnes categorielles et numeriques
categorical_features = test_filtered.select_dtypes(include=['object']).columns
numeric_features = test_filtered.select_dtypes(include=['int64', 'float64']).columns

# Remplir les valeurs manquantes
for col in numeric_features:
    test_filtered[col].fillna(test_filtered[col].median(), inplace=True)

for col in categorical_features:
    mode_val = test_filtered[col].mode()[0] if len(test_filtered[col].mode()) > 0 else 'Unknown'
    test_filtered[col].fillna(mode_val, inplace=True)

# Encoder les variables categorielles (utiliser train pour fit)
for col in categorical_features:
    le = LabelEncoder()
    # Combiner train et test pour eviter les erreurs
    combined = pd.concat([train[col].astype(str), test_filtered[col].astype(str)])
    le.fit(combined)
    test_filtered[col] = le.transform(test_filtered[col].astype(str))

print(f"Test prepare: {test_filtered.shape}")
print(f"Colonnes: {list(test_filtered.columns)}")

# 4. Faire les predictions
print("\nPrediction en cours...")
predictions = model.predict(test_filtered)

print(f"\nPredictions terminees!")
print(f"Prix min: ${predictions.min():,.2f}")
print(f"Prix max: ${predictions.max():,.2f}")
print(f"Prix moyen: ${predictions.mean():,.2f}")
print(f"Prix median: ${np.median(predictions):,.2f}")

# 5. Creer le fichier de soumission
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})

submission.to_csv('my_submission.csv', index=False)

print("\nFichier de soumission cree: my_submission.csv")
print(f"Nombre de predictions: {len(submission)}")

print("\nApercu des predictions:")
print(submission.head(10))

print("\n" + "="*70)
print("TERMINÉ!")
print("="*70)
print("\nProchaines etapes:")
print("1. Verifiez my_submission.csv")
print("2. Soumettez sur Kaggle si necessaire")
print("3. Executez hyperparameter_tuning.py pour ameliorer le modele")
