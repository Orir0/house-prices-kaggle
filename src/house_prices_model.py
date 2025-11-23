import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

print("Scikit-learn Random Forest for House Prices Prediction")

# Load the data
print("\n1. Loading data...")
train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Display basic info about the data
print("\n2. Data overview:")
print("Training data columns:", train_df.columns.tolist())
print("\nFirst few rows of training data:")
print(train_df.head())

print("\nTarget variable (SalePrice) statistics:")
print(train_df['SalePrice'].describe())

# Check for missing values
print("\n3. Missing values analysis:")
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()
print("Missing values in training set:")
print(missing_train[missing_train > 0].sort_values(ascending=False))
print("\nMissing values in test set:")
print(missing_test[missing_test > 0].sort_values(ascending=False))

# Basic preprocessing
print("\n4. Basic preprocessing...")

# Separate features and target
y = train_df['SalePrice']
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
X_test = test_df.drop(['Id'], axis=1)

print(f"Features shape: {X_train.shape}")
print(f"Target shape: {y.shape}")

# Handle missing values - simple approach
# Fill numerical columns with median
numerical_cols = X_train.select_dtypes(include=[np.number]).columns
X_train[numerical_cols] = X_train[numerical_cols].fillna(X_train[numerical_cols].median())
X_test[numerical_cols] = X_test[numerical_cols].fillna(X_train[numerical_cols].median())

# Fill categorical columns with mode
categorical_cols = X_train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = X_train[col].mode()[0] if not X_train[col].mode().empty else 'Unknown'
    X_train[col] = X_train[col].fillna(mode_value)
    X_test[col] = X_test[col].fillna(mode_value)

print("Missing values handled!")

# Encode categorical variables
print("\n5. Encoding categorical variables...")
from sklearn.preprocessing import LabelEncoder

# Apply label encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined data to handle unseen categories
    combined_data = pd.concat([X_train[col], X_test[col]])
    le.fit(combined_data)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

print("Categorical variables encoded!")
print(f"Final features shape: {X_train.shape}")

# Split training data for validation
print("\n6. Splitting data for validation...")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

print(f"Training split: {X_train_split.shape}")
print(f"Validation split: {X_val_split.shape}")

# Train Random Forest model
print("\n7. Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_split, y_train_split)

# Make predictions on validation set
print("\n8. Making predictions on validation set...")
y_val_pred = rf_model.predict(X_val_split)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
print(f"Validation RMSE: ${rmse:,.2f}")

# Calculate Log RMSE (competition metric)
log_rmse = np.sqrt(mean_squared_error(np.log(y_val_split), np.log(y_val_pred)))
print(f"Validation Log RMSE: {log_rmse:.4f}")

# Feature importance
print("\n9. Feature importance (top 10):")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# Train final model on full training data
print("\n10. Training final model on full training data...")
final_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y)

# Make predictions on test set
print("\n11. Making predictions on test set...")
test_predictions = final_model.predict(X_test)

print(f"Test predictions shape: {test_predictions.shape}")
print(f"Test predictions range: ${test_predictions.min():,.0f} - ${test_predictions.max():,.0f}")

# Create submission file
print("\n12. Creating submission file...")
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created!")

print("\n13. Submission file preview:")
print(submission.head(10))

print(f"\n14. Final model performance summary:")
print(f"- Training samples: {X_train.shape[0]}")
print(f"- Features: {X_train.shape[1]}")
print(f"- Test samples: {X_test.shape[0]}")
print(f"- Validation Log RMSE: {log_rmse:.4f}")
print(f"- Model: Random Forest (100 trees, max_depth=10)")
print(f"- Submission file: submission.csv")
