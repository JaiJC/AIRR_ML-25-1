# AIRR ML Pipeline - Quick Reference Guide

## 🎯 What's Been Implemented

This document provides a quick reference for the comprehensive AIRR ML pipeline.

---

## 📊 Notebook Structure (20 Sections Total)

### **Basic EDA (Sections 1-6)**
1. Dataset Overview
2. Single Dataset Deep Dive
3. Sequence-Level Analysis
4. Positive vs Negative Comparison
5. Cross-Dataset Variability
6. Initial Key Findings

### **Advanced Analysis (Sections 7-14)**
7. Statistical Rigor (FDR correction, effect sizes, VIF)
8. Clonotype Analysis (diversity, expansion, sharing)
9. Outlier Detection (multi-method consensus)
10. Dimensionality Reduction (PCA, t-SNE)
11. Advanced Features (V-J pairing, motifs)
12. Feature Correlation (multicollinearity)
13. Baseline Models (CV, calibration, learning curves)
14. Reproducibility Setup (seeds, logging)

### **Production ML (Sections 15-19)**
15. Feature Selection (VIF, FDR, effect size)
16. Outlier Investigation & Treatment
17. Ensemble Methods (XGBoost, LightGBM)
18. Voting & Stacking Ensembles
19. Permutation Feature Importance

### **Final Summary (Section 20)**
20. Deployment Checklist & Recommendations

---

## 🚀 Quick Start Commands

### Run the Entire Pipeline
```bash
# Open notebook
jupyter notebook jc-airr-ml-25-1.ipynb

# Or in VS Code
code jc-airr-ml-25-1.ipynb
```

### Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels tqdm
pip install xgboost lightgbm  # Optional but recommended
```

### Generate Predictions
```python
# After training (Section 17-18)
predictions = final_model.predict_proba(X_test)
```

---

## 🔑 Key Functions Reference

### Statistical Analysis
```python
# Group comparison with FDR correction
comparison_results = comprehensive_group_comparison(
    pos_features, neg_features, feature_cols
)

# Effect sizes
cohens_d = cohens_d(group1, group2)
cliffs_delta = cliffs_delta(group1, group2)
```

### Clonotype Metrics
```python
# Diversity and expansion
metrics = calculate_clonotype_metrics(repertoire_df)
# Returns: shannon_entropy, simpson_index, gini_coefficient, top10_dominance
```

### Outlier Detection
```python
# Multi-method detection
outlier_results = detect_outliers_multimethod(
    feature_df, contamination=0.1
)
```

### Feature Selection
```python
# Comprehensive selection
X_cleaned, selected_features, report = select_features_comprehensive(
    X_df, y, comparison_df, vif_df
)
```

### Model Training
```python
# Hyperparameter tuning
best_model, best_params, best_score = hyperparameter_tuning_randomized(
    model, param_distributions, X, y, cv_folds=5
)
```

---

## 📈 Expected Performance

| Metric | Target Value |
|--------|--------------|
| **Baseline AUC** | 0.75-0.77 |
| **After Feature Selection** | 0.76-0.79 |
| **After Tuning** | 0.78-0.82 |
| **Ensemble** | 0.80-0.85+ |

---

## ⚠️ Common Issues & Solutions

### Issue: Import errors for XGBoost/LightGBM
**Solution:** `pip install xgboost lightgbm` or use Random Forest only

### Issue: Memory errors during feature extraction
**Solution:** Process in batches, use sparse matrices, reduce k-mer size

### Issue: High CV variance
**Solution:** Increase folds, use nested CV, check for data leakage

### Issue: Overfitting (train >> val AUC)
**Solution:** Reduce complexity, add regularization, use ensemble

---

## 📋 Checklist Before Running

- [ ] Data paths configured correctly
- [ ] All packages installed
- [ ] Random seeds set (done automatically)
- [ ] Sufficient RAM available (8GB+)
- [ ] Output directories exist

---

## 🎯 Key Takeaways

1. **Always use FDR correction** for multiple comparisons
2. **Report effect sizes** with p-values
3. **Investigate outliers** before removing
4. **Use stratified CV** for imbalanced data
5. **Ensemble > single model** in most cases
6. **Document everything** for reproducibility

---

## 📞 Support

- **Full Documentation:** See `README.md`
- **Code Comments:** In notebook cells
- **GitHub Issues:** After competition
- **Kaggle Forum:** During competition

---

**Last Updated:** November 26, 2025  
**Status:** Production-Ready ✅
