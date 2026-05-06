"""
PCA Component Sweep — find best n_components for PCA.

Workflow per n_components value (5, 6, 7, 8, 9, 10):
1. Fix PCA n_components
2. Run LazyPredict to find top 3 models
3. Hyperparameter tune each top-3 model with Optuna
4. Build VotingRegressor ensemble of 3 tuned models
5. Single split evaluation (R², RMSE, MAE)
6. K-Fold (k=4) cross validation
7. Print all results to log

Run:
    python pca_sweep.py 2>&1 | tee pca_sweep_log.txt
"""

import sys
import time
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, VotingRegressor,
    HistGradientBoostingRegressor, BaggingRegressor,
    GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, PassiveAggressiveRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_PATH       = r"best_data.csv"
TARGET_COL      = "Average_Bleaching"
N_COMPONENTS_LIST = [5, 6, 7, 8, 9, 10]
N_OPTUNA_TRIALS = 30
N_KFOLD_SPLITS  = 4
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

MONTH_COLS = [f'Month_{i}' for i in range(1, 13)]


# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
def log(msg, level='info'):
    """Print timestamped message."""
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {'info': '  ', 'header': '\n', 'sub': '    '}.get(level, '  ')
    print(f"[{ts}] {prefix}{msg}", flush=True)


def section(title, char='='):
    """Print a section header."""
    bar = char * 75
    print(f"\n{bar}", flush=True)
    print(f"{title}", flush=True)
    print(f"{bar}", flush=True)


# ─────────────────────────────────────────────────────────────────
# DATA PREP (one-time)
# ─────────────────────────────────────────────────────────────────
def load_and_prep_data():
    log("Loading data...", level='header')
    df = pd.read_csv(DATA_PATH)

    # Drop rolling columns if exist
    drop_if_exist = ['SSTA_Rolling_Mean', 'TSA_Rolling_Mean', 'Unnamed: 0']
    for col in drop_if_exist:
        if col in df.columns:
            df = df.drop(columns=col)

    log(f"Shape: {df.shape}")
    log(f"Target: {TARGET_COL}")

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Separate month columns
    X_train_month = X_train_raw[MONTH_COLS].copy().reset_index(drop=True)
    X_test_month  = X_test_raw[MONTH_COLS].copy().reset_index(drop=True)
    X_train_num   = X_train_raw.drop(columns=MONTH_COLS)
    X_test_num    = X_test_raw.drop(columns=MONTH_COLS)

    return {
        'X': X, 'y': y,
        'X_train_num': X_train_num, 'X_test_num': X_test_num,
        'X_train_month': X_train_month, 'X_test_month': X_test_month,
        'y_train': y_train.reset_index(drop=True),
        'y_test': y_test.reset_index(drop=True),
    }


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING for a given n_components
# ─────────────────────────────────────────────────────────────────
def preprocess(data, n_components):
    """Impute → Scale → PCA → reattach month."""
    imputer = SimpleImputer(strategy='median')
    Xtr_imp = imputer.fit_transform(data['X_train_num'])
    Xte_imp = imputer.transform(data['X_test_num'])

    scaler = StandardScaler()
    Xtr_ss = scaler.fit_transform(Xtr_imp)
    Xte_ss = scaler.transform(Xte_imp)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    Xtr_pca = pca.fit_transform(Xtr_ss)
    Xte_pca = pca.transform(Xte_ss)

    pc_cols = [f'PC{i+1}' for i in range(Xtr_pca.shape[1])]
    Xtr_df = pd.DataFrame(Xtr_pca, columns=pc_cols)
    Xte_df = pd.DataFrame(Xte_pca, columns=pc_cols)

    X_train_final = pd.concat([Xtr_df, data['X_train_month']], axis=1)
    X_test_final  = pd.concat([Xte_df, data['X_test_month']], axis=1)

    var_explained = pca.explained_variance_ratio_.sum()
    log(f"PCA: n_components={n_components}, "
        f"variance explained = {var_explained:.4f}, "
        f"final shape = {X_train_final.shape}")

    return X_train_final, X_test_final, var_explained


# ─────────────────────────────────────────────────────────────────
# LAZY PREDICT — pick top 3
# ─────────────────────────────────────────────────────────────────
def lazy_predict_top3(X_train, X_test, y_train, y_test):
    """Use LazyPredict to find top 3 models by R²."""
    log("Running LazyPredict to find top 3 models...", level='header')

    try:
        from lazypredict.Supervised import LazyRegressor
        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, _ = reg.fit(X_train, X_test, y_train, y_test)

        # Get top 3 by R²
        top3 = models.head(3).index.tolist()
        log(f"Top 3 models by R²: {top3}")
        log(f"Top 5 R² scores:")
        for name, row in models.head(5).iterrows():
            log(f"  {name:<35} R²={row['R-Squared']:.4f}, RMSE={row['RMSE']:.4f}", level='sub')
        return top3
    except Exception as e:
        log(f"LazyPredict failed: {e}, using default top 3 (ET, RF, XGB)")
        return ['ExtraTreesRegressor', 'RandomForestRegressor', 'XGBRegressor']


# ─────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVES per model
# ─────────────────────────────────────────────────────────────────
def get_objective(model_name, X_train, y_train):
    """Return Optuna objective function for the given model name."""

    def make_pipeline_with_model(model):
        return Pipeline([('model', model)])

    def cv_score(pipeline):
        return cross_val_score(
            pipeline, X_train, y_train,
            cv=4, scoring='r2', n_jobs=-1
        ).mean()

    if model_name == 'ExtraTreesRegressor':
        def obj(trial):
            m = ExtraTreesRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 5, 40),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                n_jobs=-1, random_state=RANDOM_STATE
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    elif model_name == 'RandomForestRegressor':
        def obj(trial):
            m = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 5, 40),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                n_jobs=-1, random_state=RANDOM_STATE
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    elif model_name == 'HistGradientBoostingRegressor':
        def obj(trial):
            m = HistGradientBoostingRegressor(
                max_iter=trial.suggest_int("max_iter", 50, 400),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50),
                l2_regularization=trial.suggest_float("l2_regularization", 0.0, 1.0),
                random_state=RANDOM_STATE
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    elif model_name == 'XGBRegressor':
        def obj(trial):
            m = xgb.XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 400),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                n_jobs=-1, random_state=RANDOM_STATE
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    elif model_name == 'BaggingRegressor':
        def obj(trial):
            base = DecisionTreeRegressor(
                max_depth=trial.suggest_int("base_max_depth", 5, 40),
                random_state=RANDOM_STATE
            )
            m = BaggingRegressor(
                estimator=base,
                n_estimators=trial.suggest_int("n_estimators", 10, 200),
                max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
                max_features=trial.suggest_float("max_features", 0.5, 1.0),
                n_jobs=-1, random_state=RANDOM_STATE
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    elif model_name == 'GradientBoostingRegressor':
        def obj(trial):
            m = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 400),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                random_state=RANDOM_STATE
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    elif model_name == 'KNeighborsRegressor':
        def obj(trial):
            m = KNeighborsRegressor(
                n_neighbors=trial.suggest_int("n_neighbors", 3, 20),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                p=trial.suggest_int("p", 1, 2),
                n_jobs=-1
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    elif model_name == 'DecisionTreeRegressor':
        def obj(trial):
            m = DecisionTreeRegressor(
                max_depth=trial.suggest_int("max_depth", 3, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                random_state=RANDOM_STATE
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    elif model_name == 'ExtraTreeRegressor':
        def obj(trial):
            m = ExtraTreeRegressor(
                max_depth=trial.suggest_int("max_depth", 3, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                random_state=RANDOM_STATE
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    elif model_name in ('AdaBoostRegressor',):
        def obj(trial):
            m = AdaBoostRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 1.0),
                random_state=RANDOM_STATE
            )
            return cv_score(make_pipeline_with_model(m))
        return obj

    # elif model_name == 'LGBMRegressor':
    #     # Optional — only if lightgbm installed
    #     def obj(trial):
    #         import lightgbm as lgb
    #         m = lgb.LGBMRegressor(
    #             n_estimators=trial.suggest_int("n_estimators", 50, 400),
    #             learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1),
    #             max_depth=trial.suggest_int("max_depth", 3, 15),
    #             num_leaves=trial.suggest_int("num_leaves", 15, 100),
    #             random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
    #         )
    #         return cv_score(make_pipeline_with_model(m))
    #     return obj

    else:
        # Fallback: return None — caller will skip
        return None


def build_model_from_params(model_name, params):
    """Reconstruct a model with the best params from Optuna."""
    if model_name == 'ExtraTreesRegressor':
        return ExtraTreesRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE)
    elif model_name == 'RandomForestRegressor':
        return RandomForestRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE)
    elif model_name == 'HistGradientBoostingRegressor':
        return HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    elif model_name == 'XGBRegressor':
        return xgb.XGBRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE)
    elif model_name == 'BaggingRegressor':
        base_max_depth = params.pop('base_max_depth', None)
        base = DecisionTreeRegressor(max_depth=base_max_depth, random_state=RANDOM_STATE)
        return BaggingRegressor(estimator=base, **params, n_jobs=-1, random_state=RANDOM_STATE)
    elif model_name == 'GradientBoostingRegressor':
        return GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    elif model_name == 'KNeighborsRegressor':
        return KNeighborsRegressor(**params, n_jobs=-1)
    elif model_name == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(**params, random_state=RANDOM_STATE)
    elif model_name == 'ExtraTreeRegressor':
        return ExtraTreeRegressor(**params, random_state=RANDOM_STATE)
    elif model_name == 'AdaBoostRegressor':
        return AdaBoostRegressor(**params, random_state=RANDOM_STATE)
    # elif model_name == 'LGBMRegressor':
    #     import lightgbm as lgb
    #     return lgb.LGBMRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1)
    else:
        return None


# ─────────────────────────────────────────────────────────────────
# K-FOLD with PCA + month reattach (per fold)
# ─────────────────────────────────────────────────────────────────
def kfold_evaluate(model, data, n_components, n_splits=4):
    """K-Fold CV with imputer + scaler + PCA refit each fold + month reattach."""
    from sklearn.base import clone

    X = data['X']
    y = data['y']

    # Pre-split month from full X
    X_month  = X[MONTH_COLS].copy()
    X_no_mon = X.drop(columns=MONTH_COLS)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2s, rmses, maes = [], [], []

    for train_idx, val_idx in kf.split(X_no_mon):
        Xtr_num = X_no_mon.iloc[train_idx]
        Xva_num = X_no_mon.iloc[val_idx]
        Xtr_mon = X_month.iloc[train_idx].reset_index(drop=True)
        Xva_mon = X_month.iloc[val_idx].reset_index(drop=True)
        ytr = y.iloc[train_idx].reset_index(drop=True)
        yva = y.iloc[val_idx].reset_index(drop=True)

        imp = SimpleImputer(strategy='median')
        Xtr_i = imp.fit_transform(Xtr_num)
        Xva_i = imp.transform(Xva_num)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_i)
        Xva_s = sc.transform(Xva_i)

        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        Xtr_p = pca.fit_transform(Xtr_s)
        Xva_p = pca.transform(Xva_s)

        pc_cols = [f'PC{i+1}' for i in range(Xtr_p.shape[1])]
        Xtr_f = pd.concat([pd.DataFrame(Xtr_p, columns=pc_cols), Xtr_mon], axis=1)
        Xva_f = pd.concat([pd.DataFrame(Xva_p, columns=pc_cols), Xva_mon], axis=1)

        m = clone(model)
        m.fit(Xtr_f, ytr)
        preds = m.predict(Xva_f)

        r2s.append(r2_score(yva, preds))
        rmses.append(np.sqrt(mean_squared_error(yva, preds)))
        maes.append(mean_absolute_error(yva, preds))

    return {
        'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
        'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
        'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
        'r2_folds': r2s, 'rmse_folds': rmses, 'mae_folds': maes,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE per n_components
# ─────────────────────────────────────────────────────────────────
def run_for_n_components(data, n_components):
    section(f"  ▶ N_COMPONENTS = {n_components}", char='=')
    t0 = time.time()

    # Step 1-2: Preprocess + LazyPredict
    X_train, X_test, var_explained = preprocess(data, n_components)
    y_train = data['y_train']
    y_test  = data['y_test']

    top3 = lazy_predict_top3(X_train, X_test, y_train, y_test)

    # Step 3: Optuna tune top 3
    log(f"\nTuning top 3 models with Optuna ({N_OPTUNA_TRIALS} trials each)...",
        level='header')

    tuned_models = {}
    for name in top3:
        objective = get_objective(name, X_train, y_train)
        if objective is None:
            log(f"⚠️  No Optuna objective for {name} — using default params", level='sub')
            try:
                tuned_models[name] = build_model_from_params(name, {})
            except Exception:
                log(f"❌ Cannot build {name} with default params, skipping", level='sub')
            continue

        log(f"  Tuning {name}...")
        tstart = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
        elapsed = time.time() - tstart

        log(f"    Best R²:    {study.best_value:.4f}", level='sub')
        log(f"    Best params: {study.best_params}", level='sub')
        log(f"    Time:       {elapsed:.1f}s", level='sub')

        # Build model with best params
        try:
            tuned_models[name] = build_model_from_params(name, dict(study.best_params))
        except Exception as e:
            log(f"    ❌ Build failed: {e}", level='sub')

    if len(tuned_models) < 2:
        log("⚠️  Less than 2 tuned models — cannot build voting ensemble", level='header')
        return None

    # Step 4: Build voting
    log("\nBuilding VotingRegressor ensemble...", level='header')
    estimators = [(f"m{i}", m) for i, (n, m) in enumerate(tuned_models.items())]
    voting = VotingRegressor(estimators=estimators)

    # Step 5: Single split eval for each model + voting
    log("\nSingle-split evaluation:", level='header')
    print(f"  {'Model':<40} {'R²':>8} {'RMSE':>8} {'MAE':>8}", flush=True)
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}", flush=True)

    single_results = {}
    for name, model in tuned_models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2  = r2_score(y_test, pred)
            rm  = np.sqrt(mean_squared_error(y_test, pred))
            ma  = mean_absolute_error(y_test, pred)
            single_results[name] = {'r2': r2, 'rmse': rm, 'mae': ma}
            print(f"  {name:<40} {r2:>8.4f} {rm:>8.4f} {ma:>8.4f}", flush=True)
        except Exception as e:
            log(f"  ❌ {name} failed single-fit: {e}")

    # Voting
    try:
        voting.fit(X_train, y_train)
        pred = voting.predict(X_test)
        r2 = r2_score(y_test, pred)
        rm = np.sqrt(mean_squared_error(y_test, pred))
        ma = mean_absolute_error(y_test, pred)
        single_results['Voting (Ensemble)'] = {'r2': r2, 'rmse': rm, 'mae': ma}
        print(f"  {'Voting (Ensemble)':<40} {r2:>8.4f} {rm:>8.4f} {ma:>8.4f}", flush=True)
    except Exception as e:
        log(f"  ❌ Voting failed: {e}")

    # Step 6: K-Fold for each model + voting
    log("\nK-Fold (k=4) evaluation:", level='header')
    print(f"  {'Model':<40} {'Mean R²':>10} {'Mean RMSE':>10} {'Mean MAE':>10}", flush=True)
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}", flush=True)

    kfold_results = {}
    for name, model in tuned_models.items():
        try:
            res = kfold_evaluate(model, data, n_components, n_splits=N_KFOLD_SPLITS)
            kfold_results[name] = res
            print(f"  {name:<40} {res['r2_mean']:>10.4f} {res['rmse_mean']:>10.4f} {res['mae_mean']:>10.4f}",
                  flush=True)
        except Exception as e:
            log(f"  ❌ K-Fold {name} failed: {e}")

    # Voting K-Fold
    try:
        # Need a fresh voting (same estimators)
        voting_fresh = VotingRegressor(estimators=estimators)
        res = kfold_evaluate(voting_fresh, data, n_components, n_splits=N_KFOLD_SPLITS)
        kfold_results['Voting (Ensemble)'] = res
        print(f"  {'Voting (Ensemble)':<40} {res['r2_mean']:>10.4f} {res['rmse_mean']:>10.4f} {res['mae_mean']:>10.4f}",
              flush=True)
    except Exception as e:
        log(f"  ❌ K-Fold Voting failed: {e}")

    elapsed = time.time() - t0
    log(f"\nTotal time for n={n_components}: {elapsed:.1f}s ({elapsed/60:.1f} min)",
        level='header')

    return {
        'n_components': n_components,
        'variance_explained': var_explained,
        'top3': top3,
        'single': single_results,
        'kfold': kfold_results,
        'time_s': elapsed,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    section("PCA COMPONENT SWEEP — Coral Bleaching Prediction", char='█')
    log(f"Optuna trials per model: {N_OPTUNA_TRIALS}")
    log(f"K-Fold splits: {N_KFOLD_SPLITS}")
    log(f"Components to test: {N_COMPONENTS_LIST}")
    log(f"Random state: {RANDOM_STATE}")

    data = load_and_prep_data()

    all_results = {}
    t_global = time.time()

    for n in N_COMPONENTS_LIST:
        try:
            result = run_for_n_components(data, n)
            all_results[n] = result
        except Exception as e:
            log(f"❌ FAILED for n={n}: {e}")
            import traceback
            traceback.print_exc()
            all_results[n] = None

    # ─── FINAL SUMMARY TABLE ───────────────────────────────────────
    section("FINAL SUMMARY — Voting Ensemble across all n_components", char='█')

    print(f"\n{'n_components':>14} {'Var.Explained':>14} {'Single R²':>12} "
          f"{'Single RMSE':>13} {'KFold R²':>11} {'KFold RMSE':>12}", flush=True)
    print(f"{'-'*14} {'-'*14} {'-'*12} {'-'*13} {'-'*11} {'-'*12}", flush=True)

    best_n = None
    best_kfold_r2 = -float('inf')

    for n, res in all_results.items():
        if res is None:
            print(f"{n:>14}  FAILED", flush=True)
            continue
        sing = res['single'].get('Voting (Ensemble)', {})
        kf   = res['kfold'].get('Voting (Ensemble)', {})
        s_r2 = sing.get('r2', float('nan'))
        s_rm = sing.get('rmse', float('nan'))
        k_r2 = kf.get('r2_mean', float('nan'))
        k_rm = kf.get('rmse_mean', float('nan'))

        print(f"{n:>14} {res['variance_explained']:>14.4f} "
              f"{s_r2:>12.4f} {s_rm:>13.4f} "
              f"{k_r2:>11.4f} {k_rm:>12.4f}", flush=True)

        if not np.isnan(k_r2) and k_r2 > best_kfold_r2:
            best_kfold_r2 = k_r2
            best_n = n

    section(f"  ★ BEST: n_components = {best_n}  →  K-Fold Voting R² = {best_kfold_r2:.4f}",
            char='★')

    log(f"\nTotal sweep time: {(time.time()-t_global)/60:.1f} min")
    log("Done!", level='header')


if __name__ == "__main__":
    main()