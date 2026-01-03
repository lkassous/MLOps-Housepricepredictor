"""
Data Preparation Script for House Prices Dataset
Loads and prepares the house prices dataset for machine learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path='train.csv'):
    """Load the house prices dataset"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    return df

def preprocess_data(df):
    """
    Preprocess the house prices dataset:
    - Handle missing values
    - Encode categorical variables
    - Separate features and target
    """
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Separate target variable
    if 'SalePrice' in data.columns:
        y = data['SalePrice']
        X = data.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
    else:
        raise ValueError("Target column 'SalePrice' not found in dataset")
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Handle missing values for numeric features (fill with median)
    for col in numeric_features:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Handle missing values for categorical features (fill with mode)
    for col in categorical_features:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    print(f"Preprocessing complete. Final shape: {X.shape}")
    
    return X, y, label_encoders

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function to demonstrate data preparation"""
    print("=" * 50)
    print("House Prices Data Preparation")
    print("=" * 50)
    
    # Load data
    df = load_data('train.csv')
    
    # Display basic info
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nTarget variable statistics:")
    print(df['SalePrice'].describe())
    
    # Preprocess data
    X, y, label_encoders = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("\nData preparation completed successfully!")
    print(f"Ready for model training with {X_train.shape[1]} features")
    
    return X_train, X_test, y_train, y_test, label_encoders

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_encoders = main()
