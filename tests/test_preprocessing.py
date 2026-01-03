"""
Tests unitaires pour les fonctions de data preprocessing
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def test_data_loading():
    """Test du chargement des données"""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    assert train_df.shape[0] == 1460
    assert test_df.shape[0] == 1459
    assert 'SalePrice' in train_df.columns
    assert 'SalePrice' not in test_df.columns

def test_missing_values_handling():
    """Test de la gestion des valeurs manquantes"""
    df = pd.DataFrame({
        'numeric_col': [1, 2, np.nan, 4],
        'categorical_col': ['A', 'B', np.nan, 'A']
    })
    
    # Numérique: remplir avec médiane
    median_val = df['numeric_col'].median()
    df['numeric_col'].fillna(median_val, inplace=True)
    assert df['numeric_col'].isna().sum() == 0
    
    # Catégoriel: remplir avec mode
    mode_val = df['categorical_col'].mode()[0]
    df['categorical_col'].fillna(mode_val, inplace=True)
    assert df['categorical_col'].isna().sum() == 0

def test_label_encoding():
    """Test de l'encodage des variables catégorielles"""
    df = pd.DataFrame({
        'category': ['A', 'B', 'C', 'A', 'B']
    })
    
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    
    assert df['category_encoded'].dtype in [np.int32, np.int64]
    assert len(df['category_encoded'].unique()) == 3
    assert df['category_encoded'].min() >= 0
