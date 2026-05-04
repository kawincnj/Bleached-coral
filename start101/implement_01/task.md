# 🪸 Coral Bleaching ML Project — Checklist

## 📁 Baseline (Reproduce the Paper)


- [ ] Add **Dummy Regressor** (predict `mean(y_train)` for all rows) and compare RMSE vs model

---

## 🔁 Baseline + K-Fold Cross Validation

- [ ] Apply **K-Fold CV** (k=5 or k=10) to the baseline Random Forest

---

## 🧪 New Experiment 1 — PCA Features (instead of manual correlation selection)

- [ ] Run PCA and keep **5 components** (matching the paper's 5 features)

---

## 🗓️ New Experiment 2 — One-Hot Encode Month
- [ ] One-Hot Encode Month

---

## 🤖 New Experiment 3 — Ensemble Models

### 3a. More than 2 models stacked
- [ ] Train multiple base models (e.g. `RandomForest`, `XGBoost`, `GradientBoosting`, `ExtraTrees`)

### 3b. RF + XGB + ... → Neural Network pipeline
- [ ] Predict on test set using the full pipeline: RF + XGB +  ... → NN

---
