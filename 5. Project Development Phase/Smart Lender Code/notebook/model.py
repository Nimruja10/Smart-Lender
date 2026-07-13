# %% [markdown]
# # Smart Lender – Loan Approval Prediction
# ### Machine Learning Pipeline: EDA → Preprocessing → Model Training → Evaluation → Export

# %% [markdown]
# ## 1. Import Libraries

# %%
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# XGBoost
from xgboost import XGBClassifier

print('All libraries imported successfully!')

# %% [markdown]
# ## 2. Load Dataset

# %%
# Load the loan dataset
df = pd.read_csv('../dataset/loan.csv')

print('Dataset Shape:', df.shape)
print('\nFirst 5 rows:')
df.head()

# %%
# Basic info about the dataset
print('Dataset Info:')
df.info()
print('\nStatistical Summary:')
df.describe()

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
# Check missing values
print('Missing Values per Column:')
print(df.isnull().sum())
print('\nMissing Value Percentage:')
print(round(df.isnull().sum() / len(df) * 100, 2))

# %%
# Loan Status distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Status', data=df, palette='Set2')
plt.title('Loan Approval Distribution')
plt.xlabel('Loan Status (Y=Approved, N=Rejected)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
print(df['Loan_Status'].value_counts())

# %%
# Categorical features analysis
cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(x=col, hue='Loan_Status', data=df, ax=axes[i], palette='Set1')
    axes[i].set_title(f'{col} vs Loan Status')
    axes[i].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

# %%
# Numerical features distribution
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, col in enumerate(num_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# %%
# Credit History vs Loan Status
plt.figure(figsize=(6, 4))
sns.countplot(x='Credit_History', hue='Loan_Status', data=df, palette='coolwarm')
plt.title('Credit History vs Loan Status')
plt.xlabel('Credit History (1=Good, 0=Bad)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Data Preprocessing & Feature Engineering

# %%
# Drop Loan_ID column (not a feature)
df.drop('Loan_ID', axis=1, inplace=True)

# ---- Handle Missing Values ----
# Categorical: fill with mode
cat_fill = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for col in cat_fill:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Numerical: fill with mean
num_fill = ['LoanAmount', 'Loan_Amount_Term']
for col in num_fill:
    df[col].fillna(df[col].mean(), inplace=True)

print('Missing values after imputation:')
print(df.isnull().sum())

# %%
# ---- Encode Categorical Features ----
le = LabelEncoder()

# Columns to encode
encode_cols = ['Gender', 'Married', 'Dependents', 'Education',
               'Self_Employed', 'Property_Area', 'Loan_Status']

for col in encode_cols:
    df[col] = le.fit_transform(df[col])

print('Encoded Dataset Sample:')
df.head()

# %%
# Correlation heatmap after encoding
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# %%
# ---- Split Features and Target ----
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Training set size : {X_train.shape}')
print(f'Testing  set size : {X_test.shape}')

# %% [markdown]
# ## 5. Model Training & Evaluation

# %%
# ---- Helper function to evaluate a model ----
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    print(f'--- {name} ---')
    print(f'  Training Accuracy : {train_acc*100:.2f}%')
    print(f'  Testing  Accuracy : {test_acc*100:.2f}%')
    print(classification_report(y_test, model.predict(X_test)))
    return model, train_acc, test_acc

# %%
# Decision Tree
dt_model, dt_train, dt_test = evaluate_model(
    'Decision Tree',
    DecisionTreeClassifier(random_state=42),
    X_train, X_test, y_train, y_test
)

# %%
# Random Forest
rf_model, rf_train, rf_test = evaluate_model(
    'Random Forest',
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test
)

# %%
# KNN
knn_model, knn_train, knn_test = evaluate_model(
    'K-Nearest Neighbors (KNN)',
    KNeighborsClassifier(n_neighbors=5),
    X_train, X_test, y_train, y_test
)

# %%
# XGBoost
xgb_model, xgb_train, xgb_test = evaluate_model(
    'XGBoost',
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    X_train, X_test, y_train, y_test
)

# %%
# ---- Model Comparison Chart ----
models     = ['Decision Tree', 'Random Forest', 'KNN', 'XGBoost']
train_accs = [dt_train, rf_train, knn_train, xgb_train]
test_accs  = [dt_test,  rf_test,  knn_test,  xgb_test]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, [t*100 for t in train_accs], width, label='Train Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, [t*100 for t in test_accs],  width, label='Test Accuracy',  color='coral')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 110)

for bar in bars1:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Save Best Model (XGBoost)

# %%
# XGBoost is the best performer — save it as model.pkl
best_model = xgb_model

with open('../model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print('Best model (XGBoost) saved as model.pkl')

# Verify we can reload it
with open('../model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

test_pred = loaded_model.predict(X_test)
print(f'Reloaded model accuracy: {accuracy_score(y_test, test_pred)*100:.2f}%')

# %% [markdown]
# ## Summary
# | Model | Train Acc | Test Acc |
# |---|---|---|
# | Decision Tree | ~100% | ~75% |
# | Random Forest | ~100% | ~79% |
# | KNN | ~82% | ~76% |
# | **XGBoost** | **~95%** | **~81%** |
# 
# **XGBoost** was selected as the best model and saved as `model.pkl` for Flask deployment.


