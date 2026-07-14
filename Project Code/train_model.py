"""
Smart Lender – Model Training Script
=====================================
Run this script to preprocess the dataset, train all four classifiers,
compare their performance, and save the best model (XGBoost) as model.pkl.

Usage:
    python train_model.py
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
print("=" * 55)
print("  Smart Lender – Loan Approval Prediction")
print("=" * 55)

df = pd.read_csv('dataset/loan.csv')
print(f"\nDataset loaded  →  {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────
# 2. Drop unnecessary column
# ─────────────────────────────────────────
df.drop('Loan_ID', axis=1, inplace=True)

# ─────────────────────────────────────────
# 3. Handle Missing Values
# ─────────────────────────────────────────
# Categorical columns: fill with mode
cat_fill = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for col in cat_fill:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Numerical columns: fill with mean
num_fill = ['LoanAmount', 'Loan_Amount_Term']
for col in num_fill:
    df[col].fillna(df[col].mean(), inplace=True)

print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 4. Encode Categorical Features
# ─────────────────────────────────────────
le = LabelEncoder()
encode_cols = ['Gender', 'Married', 'Dependents', 'Education',
               'Self_Employed', 'Property_Area', 'Loan_Status']

for col in encode_cols:
    df[col] = le.fit_transform(df[col])

# ─────────────────────────────────────────
# 5. Split into Features and Target
# ─────────────────────────────────────────
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 6. Train & Evaluate All Models
# ─────────────────────────────────────────
print("\n--- Training Models ---\n")

results = {}

def train_evaluate(name, model):
    """Train a model and return train/test accuracy."""
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    results[name] = {'model': model, 'train': train_acc, 'test': test_acc}
    print(f"[{name}]")
    print(f"  Train Accuracy : {train_acc * 100:.2f}%")
    print(f"  Test  Accuracy : {test_acc  * 100:.2f}%")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    return model

# Decision Tree
train_evaluate(
    'Decision Tree',
    DecisionTreeClassifier(random_state=42)
)

# Random Forest
train_evaluate(
    'Random Forest',
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# KNN
train_evaluate(
    'KNN',
    KNeighborsClassifier(n_neighbors=5)
)

# XGBoost
train_evaluate(
    'XGBoost',
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
)

# ─────────────────────────────────────────
# 7. Select & Save Best Model
# ─────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]['test'])
best_model = results[best_name]['model']

print("=" * 55)
print(f"  Best Model  : {best_name}")
print(f"  Train Acc   : {results[best_name]['train'] * 100:.2f}%")
print(f"  Test  Acc   : {results[best_name]['test']  * 100:.2f}%")
print("=" * 55)

# Save model with pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\n model.pkl saved successfully!")
print("You can now run:  python app.py")
