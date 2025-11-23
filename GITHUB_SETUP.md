# GitHub Setup Guide

## Steps to Upload to GitHub

### 1. Initialize Git Repository (if not already done)
```bash
cd /Users/shulamithcalman/Documents/tfdf_houseprices
git init
```

### 2. Add All Files
```bash
git add .
```

### 3. Make Initial Commit
```bash
git commit -m "Initial commit: House prices prediction with ensemble methods (AdaBoost, Random Forest)"
```

### 4. Create Repository on GitHub
1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Name it (e.g., `house-prices-kaggle` or `kaggle-house-prices`)
4. **Don't** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

### 5. Connect Local Repository to GitHub
```bash
# Replace <your-username> and <repo-name> with your actual values
git remote add origin https://github.com/<your-username>/<repo-name>.git
```

### 6. Push to GitHub
```bash
git branch -M main
git push -u origin main
```

## What Gets Uploaded

✅ **Included:**
- All Python source code (`src/`)
- README.md
- requirements.txt
- IMPROVEMENTS.md
- .gitignore
- Dataset folder structure (but not the actual CSV files if they're large)

❌ **Excluded (via .gitignore):**
- `venv/` - Virtual environment
- `submission*.csv` - Submission files
- `models/*.pkl`, `*.joblib` - Saved model files
- `__pycache__/` - Python cache
- `.DS_Store` - macOS system files

## Note About Dataset

The actual `train.csv` and `test.csv` files are typically **not** uploaded to GitHub because:
- They're large files
- They're available on Kaggle
- GitHub has file size limits

If you want to include them, you can:
1. Use [Git LFS](https://git-lfs.github.com/) for large files
2. Or add a note in README that users should download from Kaggle

## Recommended Repository Description

When creating the GitHub repo, use this description:
```
Machine learning solution for Kaggle House Prices competition using ensemble methods (AdaBoost, Random Forest, XGBoost). Implements weak learners and ensemble techniques to predict residential home prices.
```

## Recommended Topics/Tags

Add these topics to your GitHub repository:
- `machine-learning`
- `kaggle`
- `ensemble-learning`
- `random-forest`
- `adaboost`
- `xgboost`
- `regression`
- `scikit-learn`
- `python`

