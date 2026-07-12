import pandas as pd
import numpy as np
import pickle
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("Starting model training pipeline...")

# 1. Load Dataset
data_path = "water_potability.csv"
df = pd.read_csv(data_path)
print(f"Loaded dataset of shape: {df.shape}")

# 2. Impute missing values based on group mean
df['ph'] = df['ph'].fillna(df.groupby(['Potability'])['ph'].transform('mean'))
df['Sulfate'] = df['Sulfate'].fillna(df.groupby(['Potability'])['Sulfate'].transform('mean'))
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df.groupby(['Potability'])['Trihalomethanes'].transform('mean'))
print("Missing values imputed.")

# 3. Detect and remove outliers using Z-score threshold of 3
outlier_mask = pd.Series(False, index=df.index)
for column in df.drop(columns=['Potability']).columns:
    z_scores = zscore(df[column])
    outlier_mask |= (np.abs(z_scores) > 3)

df_cleaned = df[~outlier_mask].copy()
print(f"Removed {outlier_mask.sum()} outlier rows. Cleaned shape: {df_cleaned.shape}")

# 4. Apply Square Root Transformation to skewed columns (Solids and Conductivity)
df_cleaned['Solids'] = np.sqrt(df_cleaned['Solids'])
df_cleaned['Conductivity'] = np.sqrt(df_cleaned['Conductivity'])
print("Applied square root transformations.")

# 5. Split Features and Target
X = df_cleaned.drop(['Potability'], axis=1)
y = df_cleaned['Potability']

# 6. Train-Val-Test Split (80% train, 10% val, 10% test, stratified)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 7. Apply SMOTE to training set to balance classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Applied SMOTE. Balanced training set shape: {X_train_res.shape}")

# 8. Fit and Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 9. Train tuned XGBClassifier
print("Training XGBClassifier...")
model = XGBClassifier(random_state=42, reg_lambda=1.3)
model.fit(X_train_scaled, y_train_res)

# 10. Evaluate on test set
test_acc = model.score(X_test_scaled, y_test)
print(f"XGBoost Test Accuracy: {test_acc * 100:.2f}%")

# 11. Save model and scaler objects
with open("potability_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved model to potability_model.pkl and scaler to scaler.pkl.")
