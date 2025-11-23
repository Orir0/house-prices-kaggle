import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("AdaBoost Regressor for House Prices Prediction")

# Load the data
print("\n1. Loading data...")
train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')


print(f"Training data shape: {train_df.shape}")

print("\nTarget variable (SalePrice) statistics:")


# Check for missing values
print("\n3. Missing values analysis:")
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()
print("Missing values in training set:")
print(missing_train[missing_train > 0].sort_values(ascending=False))
print("\nMissing values in test set:")
print(missing_test[missing_test > 0].sort_values(ascending=False))

#basic preprocessing
print("\n Basic preprocessing...")

#separate features and target
y = train_df.SalePrice
X_train = train_df.drop(['Id', "SalePrice"], axis=1)
X_test = test_df.drop(['Id'], axis=1)




#handle missing values
#fill numerical columns with median
numerical_cols = X_train.select_dtypes(include=[np.number]).columns
X_train[numerical_cols] = X_train[numerical_cols].fillna(X_train[numerical_cols].median())
X_test[numerical_cols] = X_test[numerical_cols].fillna(X_train[numerical_cols].median())

#fill categorical columns with mode
categorical_cols = X_train.select_dtypes(include=['object']).columns
X_train[categorical_cols] = X_train[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
X_test[categorical_cols] = X_test[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])

print("Missing values handled!")


print("\n encoding categorical variables...")
from sklearn.preprocessing import LabelEncoder

#apply label encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_data = pd.concat([X_train[col], X_test[col]])
    le.fit(combined_data)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

print("Categorical variables encoded!")
print(f"Final features shape: {X_train.shape}")

#split data for validation
print("\n5. Splitting data for validation...")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

print(f"Training split: {X_train_split.shape}")
print(f"Validation split: {X_val_split.shape}")

#train adaboost regressor
print("\n6. Training AdaBoost Regressor...")
adaboost_model = AdaBoostRegressor(
    n_estimators=100,
    random_state=42
)

adaboost_model.fit(X_train_split, y_train_split)

#make predictions on validation set
y_val_pred = adaboost_model.predict(X_val_split)

rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))

log_rmse = np.sqrt(mean_squared_error(np.log(y_val_split), np.log(y_val_pred)))

print(f"Validation RMSE: ${rmse:,.2f}")
print(f"Validation Log RMSE: {log_rmse:.4f}")

#feature importance
print("\n7. Feature importance (top 10):")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': adaboost_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

#train final model on full training data
print("\n7. Training final model on full training data...")
final_model = AdaBoostRegressor(
    n_estimators=100,
    random_state=42
)

final_model.fit(X_train, y)

#make predictions on test set
print("\n8. Making predictions on test set...")
test_predictions = final_model.predict(X_test)

print(f"Test predictions shape: {test_predictions.shape}")
print(f"Test predictions range: ${test_predictions.min():,.0f} - ${test_predictions.max():,.0f}")

#create submission file
print("\n9. Creating submission file...")
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_predictions
})

submission.to_csv('submission_adaboost.csv', index=False)
print("Submission file 'submission_adaboost.csv' created!")

print("\n10. Submission file preview:")
print(submission.head(10))

print(f"\n11. Final model performance summary:")
print(f"- Training samples: {X_train.shape[0]}")
print(f"- Features: {X_train.shape[1]}")
print(f"- Test samples: {X_test.shape[0]}")
print(f"- Validation Log RMSE: {log_rmse:.4f}")
print(f"- Model: AdaBoost Regressor (100 estimators)")
print(f"- Submission file: submission_adaboost.csv")