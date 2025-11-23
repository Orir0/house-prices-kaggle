# House Prices Prediction - Kaggle Competition

A machine learning project solving the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) Kaggle competition using ensemble methods and weak learners.

## Overview

This project implements multiple machine learning models to predict house prices in Ames, Iowa. The solution uses **ensemble methods** that combine multiple weak learners (decision trees) to create predictions. The models explored include:

- **Random Forest** - An ensemble of decision trees using bagging
- **AdaBoost** - An ensemble that sequentially trains weak learners, focusing on previous mistakes
- **XGBoost** - Gradient boosting ensemble (when available)
- **Ensemble Model** - Weighted combination of Random Forest and XGBoost

## Competition Details

- **Competition**: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Task**: Regression (predicting continuous house prices)
- **Metric**: Root Mean Squared Logarithmic Error (RMSLE)
- **Dataset**: 79 features describing residential homes in Ames, Iowa
  - Training set: 1,460 examples
  - Test set: 1,459 examples

## Project Structure

```
tfdf_houseprices/
├── house-prices-advanced-regression-techniques/  # Dataset folder
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
├── src/                                         # Source code
│   ├── house_prices_model.py                   # Basic Random Forest
│   ├── house_prices_model_improved.py          # Improved RF with feature engineering
│   ├── house_prices_adaboost.py                # AdaBoost Regressor
│   └── house_prices_ensemble.py                # Ensemble (RF + XGBoost)
├── models/                                      # Saved models (gitignored)
├── requirements.txt                             # Python dependencies
├── IMPROVEMENTS.md                              # Model improvement guide
└── README.md                                    # This file
```

## Models Implemented

### 1. Basic Random Forest (`house_prices_model.py`)
- **Model**: Random Forest Regressor (100 trees, max_depth=10)
- **Approach**: Simple baseline using scikit-learn's Random Forest
- **Features**: Basic preprocessing (missing value imputation, label encoding)

### 2. Improved Random Forest (`house_prices_model_improved.py`)
- **Model**: Random Forest Regressor (200 trees, max_depth=15)
- **Improvements**:
  - Enhanced feature engineering (TotalSF, TotalBathrooms, HouseAge, etc.)
  - Better hyperparameters (min_samples_split, min_samples_leaf)
  - More trees and deeper trees for better learning

### 3. AdaBoost Regressor (`house_prices_adaboost.py`)
- **Model**: AdaBoost with Decision Trees as weak learners
- **Approach**: Sequential ensemble that trains 100 weak learners
- **How it works**: Each subsequent learner focuses on samples that previous learners got wrong
- **Key feature**: Adaptive boosting - automatically adjusts sample weights during training

### 4. Ensemble Model (`house_prices_ensemble.py`) ⭐ **Recommended**
- **Models**: Random Forest + XGBoost (when available)
- **Approach**: Weighted average of predictions from multiple models
- **Features**: 
  - Feature engineering from improved model
  - Automatic weight optimization
  - Combines strengths of different algorithms

## Key Concepts: Ensemble Methods & Weak Learners

### What are Weak Learners?
Weak learners are simple models that perform slightly better than random guessing. In this project:
- **Decision Trees** (with max_depth=1-3) are used as weak learners
- Individually, they're not very accurate
- Combined together, they create a strong, accurate model

### Ensemble Methods Used

1. **Random Forest (Bagging)**
   - Trains many decision trees independently on random subsets of data
   - Each tree votes, final prediction is the average
   - Reduces overfitting through diversity

2. **AdaBoost (Boosting)**
   - Trains weak learners sequentially
   - Each new learner focuses on mistakes from previous ones
   - Weights are assigned based on each learner's performance
   - Combines all learners with weighted voting

3. **XGBoost (Gradient Boosting)**
   - Advanced boosting algorithm
   - Optimizes a loss function using gradient descent
   - Often achieves best performance on tabular data

4. **Model Ensemble**
   - Combines different algorithms (RF + XGBoost)
   - Uses weighted average to leverage strengths of each model
   - Often outperforms individual models

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd tfdf_houseprices
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Download from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - Place `train.csv` and `test.csv` in `house-prices-advanced-regression-techniques/` folder

## Usage

### Run Basic Random Forest:
```bash
python src/house_prices_model.py
```

### Run Improved Random Forest:
```bash
python src/house_prices_model_improved.py
```

### Run AdaBoost:
```bash
python src/house_prices_adaboost.py
```

### Run Ensemble Model (Best Performance):
```bash
python src/house_prices_ensemble.py
```

Each script will:
1. Load and preprocess the data
2. Train the model(s)
3. Evaluate on validation set
4. Generate predictions for test set
5. Create a submission CSV file

## Results

The models achieve competitive performance on the validation set:
- **Basic Random Forest**: ~0.1540 Log RMSE
- **Improved Random Forest**: ~0.145-0.147 Log RMSE
- **AdaBoost**: Comparable to Random Forest
- **Ensemble (RF + XGBoost)**: ~0.135-0.145 Log RMSE (best)

*Note: Actual Kaggle leaderboard scores may vary*

## Technical Details

### Preprocessing Pipeline
1. **Missing Value Handling**:
   - Numerical columns: Fill with median
   - Categorical columns: Fill with mode

2. **Feature Encoding**:
   - Label encoding for categorical variables
   - Combined train/test encoding to handle unseen categories

3. **Feature Engineering** (in improved models):
   - `TotalSF`: Total square footage (basement + 1st floor + 2nd floor)
   - `TotalBathrooms`: Sum of all bathrooms
   - `HouseAge`: Age of the house
   - `RemodAge`: Years since remodeling
   - Binary features: HasBasement, HasGarage, HasPool, etc.
   - Interaction features: OverallScore, LotPerSF, etc.

### Model Hyperparameters

**Random Forest:**
- `n_estimators`: 100-200 (number of trees)
- `max_depth`: 10-15 (tree depth)
- `min_samples_split`: 5 (minimum samples to split)
- `min_samples_leaf`: 2 (minimum samples in leaf)

**AdaBoost:**
- `n_estimators`: 100 (number of weak learners)
- Base estimator: DecisionTreeRegressor(max_depth=1-3)

**XGBoost:**
- `n_estimators`: 200
- `max_depth`: 6
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8

## Dependencies

- `scikit-learn` - Machine learning algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `xgboost` - Gradient boosting (optional, for ensemble)
- `matplotlib` - Visualization (optional)
- `seaborn` - Statistical visualization (optional)

## Future Improvements

Potential areas for enhancement:
- More advanced feature engineering
- Hyperparameter tuning with cross-validation
- Stacking multiple models
- Neural networks for tabular data
- Handling outliers and feature scaling
- More sophisticated missing value imputation

## References

- [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## License

This project is for educational purposes as part of a Kaggle competition.

## Author

Built as a learning project to explore ensemble methods and weak learners in machine learning.
