"""
Find Best PCA Variance for Coral Bleaching Prediction.

Workflow:
  For each PCA variance threshold (0.65 to 0.95):
    1. Preprocess: Impute → Scale → PCA(variance) → reattach Month + Paper features
    2. Run 4-fold CV with ExtraTrees, RandomForest, HistGradientBoost
    3. Compute Voting Ensemble score
    4. Print all results

Run:
    python find_best_pca.py 2>&1 | tee find_best_pca_log.txt
"""

import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor,
    HistGradientBoostingRegressor, VotingRegressor
)
from sklearn.base import clone

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_PATH        = "best_data.csv"
TARGET           = "Percent_Bleaching"
PAPER_FEATURES   = ['Latitude_Degrees', 'Longitude_Degrees', 'Depth_m', 'ClimSST', 'SSTA']
MONTH_COLS       = [f'Month_{i}' for i in range(1, 13)]
VARIANCES_TO_TEST = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
N_FOLDS          = 4
RANDOM_STATE     = 42
TEST_SIZE        = 0.2


# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
def log(msg, level=0):
    ts = datetime.now().strftime("%H:%M:%S")
    indent = "  " * level
    print(f"[{ts}] {indent}{msg}", flush=True)


def section(title, char='='):
    bar = char * 75
    print(f"\n{bar}", flush=True)
    print(f"{title}", flush=True)
    print(f"{bar}", flush=True)


# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
def load_data():
    log("Loading data...")
    df = pd.read_csv(DATA_PATH)
    log(f"Shape: {df.shape}")

    df = df.dropna(subset=[TARGET])
    log(f"After drop NaN target: {df.shape}")

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Separate month and paper features (will bypass PCA)
    X_train_month = X_train_raw[MONTH_COLS].copy().reset_index(drop=True)
    X_train_paper = X_train_raw[PAPER_FEATURES].copy().reset_index(drop=True)

    # Numeric features for PCA (drop month, but keep paper features)
    X_train_num = X_train_raw.drop(columns=MONTH_COLS)

    y_train = y_train.reset_index(drop=True)

    log(f"X_train shape (numeric for PCA): {X_train_num.shape}")
    log(f"X_train_month shape: {X_train_month.shape}")
    log(f"X_train_paper shape: {X_train_paper.shape}")

    return {
        'X_num': X_train_num,
        'X_month': X_train_month,
        'X_paper': X_train_paper,
        'y': y_train
    }


# ─────────────────────────────────────────────────────────────────
# K-FOLD CV with PCA + reattach (per fold)
# ─────────────────────────────────────────────────────────────────
def kfold_cv_with_pca(model, data, variance, n_folds=4):
    """
    K-Fold CV that:
      - fits imputer + scaler + PCA per fold (no leakage)
      - reattaches Month + Paper features after PCA
      - returns mean R², RMSE, MAE
    """
    X_num = data['X_num'].reset_index(drop=True)
    X_month = data['X_month'].reset_index(drop=True)
    X_paper = data['X_paper'].reset_index(drop=True)
    y = data['y'].reset_index(drop=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    r2s, rmses, maes = [], [], []

    for train_idx, val_idx in kf.split(X_num):
        # Split numeric
        X_num_tr = X_num.iloc[train_idx]
        X_num_val = X_num.iloc[val_idx]

        # Split month + paper (will reattach)
        X_mon_tr = X_month.iloc[train_idx].reset_index(drop=True)
        X_mon_val = X_month.iloc[val_idx].reset_index(drop=True)
        X_pap_tr = X_paper.iloc[train_idx].reset_index(drop=True)
        X_pap_val = X_paper.iloc[val_idx].reset_index(drop=True)

        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        # Preprocess: impute → scale → PCA (numeric only)
        imp = SimpleImputer(strategy='median')
        X_tr_i = imp.fit_transform(X_num_tr)
        X_val_i = imp.transform(X_num_val)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_i)
        X_val_s = sc.transform(X_val_i)

        pca = PCA(n_components=variance, random_state=RANDOM_STATE)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_val_p = pca.transform(X_val_s)

        # Reattach month + paper features
        n_pcs = X_tr_p.shape[1]
        pc_cols = [f'PC{i+1}' for i in range(n_pcs)]

        X_tr_final = pd.concat([
            pd.DataFrame(X_tr_p, columns=pc_cols),
            X_mon_tr,
            X_pap_tr
        ], axis=1)
        X_val_final = pd.concat([
            pd.DataFrame(X_val_p, columns=pc_cols),
            X_mon_val,
            X_pap_val
        ], axis=1)

        # Train and predict
        m = clone(model)
        m.fit(X_tr_final, y_tr)
        preds = m.predict(X_val_final)

        r2s.append(r2_score(y_val, preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        maes.append(mean_absolute_error(y_val, preds))

    return {
        'r2': np.mean(r2s),
        'r2_std': np.std(r2s),
        'rmse': np.mean(rmses),
        'mae': np.mean(maes),
        'n_pcs': n_pcs
    }


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    section("FIND BEST PCA VARIANCE — Coral Bleaching", char='█')
    log(f"Variances to test: {VARIANCES_TO_TEST}")
    log(f"K-Fold splits: {N_FOLDS}")
    log(f"Random state: {RANDOM_STATE}")

    # Load data once
    data = load_data()
    t_global = time.time()

    # Define baseline models (default params; for finding best variance only)
    base_models = {
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=200, max_depth=20, max_features='sqrt',
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=20, max_features='sqrt',
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        'HistGradientBoost': HistGradientBoostingRegressor(
            max_iter=200, max_depth=10, learning_rate=0.05,
            random_state=RANDOM_STATE
        )
    }

    # Voting ensemble
    voting = VotingRegressor([
        ('et', base_models['ExtraTrees']),
        ('rf', base_models['RandomForest']),
        ('hgb', base_models['HistGradientBoost']),
    ])

    # Test each variance
    all_results = {}

    for variance in VARIANCES_TO_TEST:
        section(f"  ▶ PCA Variance = {variance}", char='=')
        t_var = time.time()
        results_for_variance = {}

        for name, model in base_models.items():
            log(f"Testing {name}...", level=1)
            t_model = time.time()
            res = kfold_cv_with_pca(model, data, variance, n_folds=N_FOLDS)
            elapsed = time.time() - t_model
            results_for_variance[name] = res

            log(f"R² = {res['r2']:.4f} ± {res['r2_std']:.4f} | "
                f"RMSE = {res['rmse']:.4f} | "
                f"MAE = {res['mae']:.4f} | "
                f"PCs = {res['n_pcs']} | "
                f"time = {elapsed:.1f}s",
                level=2)

        # Voting ensemble
        log(f"Testing Voting (Ensemble)...", level=1)
        t_v = time.time()
        res_v = kfold_cv_with_pca(voting, data, variance, n_folds=N_FOLDS)
        elapsed = time.time() - t_v
        results_for_variance['Voting'] = res_v
        log(f"R² = {res_v['r2']:.4f} ± {res_v['r2_std']:.4f} | "
            f"RMSE = {res_v['rmse']:.4f} | "
            f"MAE = {res_v['mae']:.4f} | "
            f"time = {elapsed:.1f}s",
            level=2)

        all_results[variance] = results_for_variance
        log(f"\nTotal time for variance={variance}: {time.time()-t_var:.1f}s",
            level=1)

    # ─── FINAL SUMMARY ────────────────────────────────────────────
    section("FINAL SUMMARY — All Variances", char='█')

    print(f"\n{'Variance':>10} {'PCs':>4} | "
          f"{'ET R²':>8} {'RF R²':>8} {'HGB R²':>8} {'Voting R²':>10} | "
          f"{'Voting RMSE':>12} {'Voting MAE':>11}", flush=True)
    print("-" * 90, flush=True)

    best_voting_r2 = -np.inf
    best_variance = None

    for variance, results in all_results.items():
        et_r2 = results['ExtraTrees']['r2']
        rf_r2 = results['RandomForest']['r2']
        hgb_r2 = results['HistGradientBoost']['r2']
        v_r2 = results['Voting']['r2']
        v_rmse = results['Voting']['rmse']
        v_mae = results['Voting']['mae']
        n_pcs = results['Voting']['n_pcs']

        print(f"{variance:>10.2f} {n_pcs:>4} | "
              f"{et_r2:>8.4f} {rf_r2:>8.4f} {hgb_r2:>8.4f} {v_r2:>10.4f} | "
              f"{v_rmse:>12.4f} {v_mae:>11.4f}", flush=True)

        if v_r2 > best_voting_r2:
            best_voting_r2 = v_r2
            best_variance = variance

    # Best per individual model
    print("-" * 90, flush=True)
    section(f"  ★ BEST PCA VARIANCE = {best_variance}  →  Voting R² = {best_voting_r2:.4f}",
            char='★')

    # Print best per model too
    log("\nBest variance per model:")
    for model_name in ['ExtraTrees', 'RandomForest', 'HistGradientBoost', 'Voting']:
        best_r2_this = -np.inf
        best_var_this = None
        for var, results in all_results.items():
            if results[model_name]['r2'] > best_r2_this:
                best_r2_this = results[model_name]['r2']
                best_var_this = var
        log(f"{model_name:<22}: variance = {best_var_this}, R² = {best_r2_this:.4f}",
            level=1)

    log(f"\nTotal time: {(time.time()-t_global)/60:.1f} min")
    log("Done!")


if __name__ == "__main__":
    main()