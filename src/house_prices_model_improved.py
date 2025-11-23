import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("IMPROVED HOUSE PRICES PREDICTION MODEL")
print("=" * 60)

# Load the data
print("\n1. Loading data...")
train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

# Save IDs for submission
test_ids = test_df['Id'].copy()

# Separate features and target
y = train_df['SalePrice']
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
X_test = test_df.drop(['Id'], axis=1)

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

# ============================================
# IMPROVEMENT 1: Better Feature Engineering
# ============================================
print("\n2. Feature Engineering...")

def create_features(df):
    """Create new features that might help predict price"""
    df = df.copy()
    
    # Total square footage
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    
    # Total bathrooms
    df['TotalBathrooms'] = (df.get('FullBath', 0) + 
                            df.get('HalfBath', 0) + 
                            df.get('BsmtFullBath', 0) + 
                            df.get('BsmtHalfBath', 0))
    
    # Age of house
    df['HouseAge'] = df.get('YrSold', 2020) - df.get('YearBuilt', 1900)
    df['RemodAge'] = df.get('YrSold', 2020) - df.get('YearRemodAdd', 1900)
    
    # Has basement, garage, pool?
    df['HasBasement'] = (df.get('TotalBsmtSF', 0) > 0).astype(int)
    df['HasGarage'] = (df.get('GarageArea', 0) > 0).astype(int)
    df['HasPool'] = (df.get('PoolArea', 0) > 0).astype(int)
    df['Has2ndFloor'] = (df.get('2ndFlrSF', 0) > 0).astype(int)
    
    # Lot size per square foot of house
    df['LotPerSF'] = df.get('LotArea', 0) / (df.get('GrLivArea', 1) + 1)
    
    # Overall quality score combinations
    df['OverallScore'] = df.get('OverallQual', 0) * df.get('OverallCond', 0)
    
    return df

X_train = create_features(X_train)
X_test = create_features(X_test)
print(f"After feature engineering: {X_train.shape}")

# ============================================
# IMPROVEMENT 2: Better Missing Value Handling
# ============================================
print("\n3. Handling missing values...")

# Fill numerical columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Fill categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 'None'
    X_train[col] = X_train[col].fillna(mode_val)
    X_test[col] = X_test[col].fillna(mode_val)

print("Missing values handled!")

# ============================================
# IMPROVEMENT 3: Better Encoding
# ============================================
print("\n4. Encoding categorical variables...")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]])
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

print("Encoding complete!")

# ============================================
# IMPROVEMENT 4: Improved Random Forest
# ============================================
print("\n5. Training IMPROVED Random Forest model...")
print("-" * 60)

# Split for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

# Improved hyperparameters
rf_improved = RandomForestRegressor(
    n_estimators=200,        # More trees (was 100)
    max_depth=15,            # Deeper trees (was 10)
    min_samples_split=5,     # Prevents overfitting
    min_samples_leaf=2,      # Prevents overfitting
    max_features='sqrt',      # Better for large feature sets
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_improved.fit(X_train_split, y_train_split)

# Validation predictions
y_val_pred = rf_improved.predict(X_val_split)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
log_rmse = np.sqrt(mean_squared_error(np.log(y_val_split), np.log(y_val_pred)))

print(f"\nValidation RMSE: ${rmse:,.2f}")
print(f"Validation Log RMSE: {log_rmse:.4f}")

# Compare with previous score
previous_score = 0.1540
improvement = ((previous_score - log_rmse) / previous_score) * 100
print(f"\nPrevious validation score: {previous_score:.4f}")
print(f"Improvement: {improvement:.1f}% better!")

# ============================================
# IMPROVEMENT 5: Feature Importance Analysis
# ============================================
print("\n6. Top 15 Most Important Features:")
print("-" * 60)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_improved.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# ============================================
# FINAL MODEL: Train on Full Data
# ============================================
print("\n7. Training final model on full training data...")
final_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

final_model.fit(X_train, y)

# ============================================
# MAKE PREDICTIONS
# ============================================
print("\n8. Making predictions on test set...")
test_predictions = final_model.predict(X_test)

print(f"Predictions range: ${test_predictions.min():,.0f} - ${test_predictions.max():,.0f}")
print(f"Mean prediction: ${test_predictions.mean():,.0f}")

# Create submission
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})

submission.to_csv('submission_improved.csv', index=False)
print("\n✅ Submission file 'submission_improved.csv' created!")

print("\n" + "=" * 60)
print("IMPROVEMENTS SUMMARY:")
print("=" * 60)
print("✅ More trees (200 vs 100)")
print("✅ Deeper trees (max_depth=15 vs 10)")
print("✅ Better regularization (min_samples_split, min_samples_leaf)")
print("✅ Feature engineering (TotalSF, TotalBathrooms, Age, etc.)")
print("✅ Better missing value handling")
print("=" * 60)

