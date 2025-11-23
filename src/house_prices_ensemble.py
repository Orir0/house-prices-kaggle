"""
Advanced House Prices Model with Ensemble Methods
Try different models and ensemble them for best performance
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost - optional improvement
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available - will use it in ensemble!")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - install with: pip install xgboost")

print("=" * 70)
print("ENSEMBLE HOUSE PRICES PREDICTION MODEL")
print("=" * 70)

# Load and prepare data (reuse feature engineering from improved version)
print("\n1. Loading and preparing data...")
train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

test_ids = test_df['Id'].copy()
y = train_df['SalePrice']
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
X_test = test_df.drop(['Id'], axis=1)

# Feature engineering
def create_features(df):
    df = df.copy()
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    df['TotalBathrooms'] = (df.get('FullBath', 0) + df.get('HalfBath', 0) + 
                            df.get('BsmtFullBath', 0) + df.get('BsmtHalfBath', 0))
    df['HouseAge'] = df.get('YrSold', 2020) - df.get('YearBuilt', 1900)
    df['RemodAge'] = df.get('YrSold', 2020) - df.get('YearRemodAdd', 1900)
    df['HasBasement'] = (df.get('TotalBsmtSF', 0) > 0).astype(int)
    df['HasGarage'] = (df.get('GarageArea', 0) > 0).astype(int)
    df['HasPool'] = (df.get('PoolArea', 0) > 0).astype(int)
    df['Has2ndFloor'] = (df.get('2ndFlrSF', 0) > 0).astype(int)
    df['LotPerSF'] = df.get('LotArea', 0) / (df.get('GrLivArea', 1) + 1)
    df['OverallScore'] = df.get('OverallQual', 0) * df.get('OverallCond', 0)
    return df

X_train = create_features(X_train)
X_test = create_features(X_test)

# Handle missing values
numerical_cols = X_train.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

categorical_cols = X_train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 'None'
    X_train[col] = X_train[col].fillna(mode_val)
    X_test[col] = X_test[col].fillna(mode_val)

# Encode categorical
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]])
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

print(f"Features: {X_train.shape[1]}, Training samples: {X_train.shape[0]}")

# Split for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

print("\n" + "=" * 70)
print("TRAINING MULTIPLE MODELS")
print("=" * 70)

# ============================================
# MODEL 1: Improved Random Forest
# ============================================
print("\n2. Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train_split, y_train_split)
rf_pred = rf_model.predict(X_val_split)
rf_log_rmse = np.sqrt(mean_squared_error(np.log(y_val_split), np.log(rf_pred)))
print(f"   Random Forest Log RMSE: {rf_log_rmse:.4f}")

# ============================================
# MODEL 2: XGBoost (if available)
# ============================================
if XGBOOST_AVAILABLE:
    print("\n3. Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse'
    )
    xgb_model.fit(X_train_split, y_train_split)
    xgb_pred = xgb_model.predict(X_val_split)
    xgb_log_rmse = np.sqrt(mean_squared_error(np.log(y_val_split), np.log(xgb_pred)))
    print(f"   XGBoost Log RMSE: {xgb_log_rmse:.4f}")
else:
    print("\n3. Skipping XGBoost (not installed)")
    xgb_model = None
    xgb_pred = None
    xgb_log_rmse = float('inf')

# ============================================
# ENSEMBLE: Weighted Average
# ============================================
print("\n" + "=" * 70)
print("ENSEMBLE PREDICTIONS")
print("=" * 70)

if XGBOOST_AVAILABLE:
    # Ensemble with both models
    print("\n4. Creating ensemble (weighted average)...")
    
    # Find best weights
    best_weight = None
    best_score = float('inf')
    
    for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w2 = 1 - w1
        ensemble_pred = w1 * rf_pred + w2 * xgb_pred
        ensemble_log_rmse = np.sqrt(mean_squared_error(np.log(y_val_split), np.log(ensemble_pred)))
        if ensemble_log_rmse < best_score:
            best_score = ensemble_log_rmse
            best_weight = w1
    
    print(f"   Best RF weight: {best_weight:.2f}, XGBoost weight: {1-best_weight:.2f}")
    print(f"   Ensemble Log RMSE: {best_score:.4f}")
    
    # Use best ensemble for final predictions
    use_ensemble = True
else:
    # Only Random Forest available
    best_score = rf_log_rmse
    use_ensemble = False
    print("\n4. Using Random Forest only (no ensemble)")

# ============================================
# TRAIN FINAL MODELS ON FULL DATA
# ============================================
print("\n" + "=" * 70)
print("TRAINING FINAL MODELS ON FULL DATA")
print("=" * 70)

print("\n5. Training final Random Forest...")
final_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
final_rf.fit(X_train, y)

if XGBOOST_AVAILABLE:
    print("6. Training final XGBoost...")
    final_xgb = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse'
    )
    final_xgb.fit(X_train, y)

# ============================================
# MAKE FINAL PREDICTIONS
# ============================================
print("\n7. Making final predictions...")

rf_test_pred = final_rf.predict(X_test)

if XGBOOST_AVAILABLE and use_ensemble:
    xgb_test_pred = final_xgb.predict(X_test)
    final_predictions = best_weight * rf_test_pred + (1 - best_weight) * xgb_test_pred
    print(f"   Using ensemble: RF={best_weight:.2f}, XGB={(1-best_weight):.2f}")
else:
    final_predictions = rf_test_pred
    print(f"   Using Random Forest only")

# Create submission
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_predictions
})

submission.to_csv('submission_ensemble.csv', index=False)
print(f"\n✅ Submission file 'submission_ensemble.csv' created!")

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Random Forest Validation Log RMSE: {rf_log_rmse:.4f}")
if XGBOOST_AVAILABLE:
    print(f"XGBoost Validation Log RMSE: {xgb_log_rmse:.4f}")
    if use_ensemble:
        print(f"Ensemble Validation Log RMSE: {best_score:.4f}")
        print(f"\n🏆 Best model: Ensemble ({best_score:.4f})")
    else:
        print(f"\n🏆 Best model: XGBoost ({xgb_log_rmse:.4f})")
else:
    print(f"\n🏆 Best model: Random Forest ({rf_log_rmse:.4f})")

print(f"\nYour current best score: 0.14851")
print(f"Expected improvement: Try submission_ensemble.csv!")
print("=" * 70)

