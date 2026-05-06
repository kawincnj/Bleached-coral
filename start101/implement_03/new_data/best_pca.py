"""
PCA Component Sweep — find best n_components for PCA.

Asks at startup which single n_components value to test.
  python pca_sweep.py        → prompts for value
  python pca_sweep.py 8      → uses 8

Workflow:
1. Fix PCA n_components
2. Run LazyPredict to find top 3 models
3. Hyperparameter tune each top-3 model with Optuna
4. Build VotingRegressor ensemble of 3 tuned models
5. Single split evaluation (R², RMSE, MAE)
6. K-Fold (k=4) cross validation
7. Print all results

GPU is auto-detected for XGBoost (CUDA 13 compatible), and for LightGBM /
CatBoost if installed. n_jobs=-1 is applied to every estimator that supports it.

Run:
    python pca_sweep.py 2>&1 | tee pca_sweep_log.txt
"""

import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, VotingRegressor,
    HistGradientBoostingRegressor, BaggingRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_PATH        = r"best_data.csv"
TARGET_COL       = "Percent_Bleaching"
N_OPTUNA_TRIALS  = 30
N_KFOLD_SPLITS   = 4
RANDOM_STATE     = 42
TEST_SIZE        = 0.2
MONTH_COLS       = [f'Month_{i}' for i in range(1, 13)]


# ─────────────────────────────────────────────────────────────────
# GPU / OPTIONAL LIBRARY DETECTION
# ─────────────────────────────────────────────────────────────────
def detect_gpu_xgb():
    """Return True if XGBoost can use CUDA on this machine."""
    try:
        m = xgb.XGBRegressor(
            n_estimators=2, tree_method="hist", device="cuda", verbosity=0
        )
        m.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
        return True
    except Exception:
        return False


# def detect_lightgbm():
#     try:
#         import lightgbm as lgb  # noqa: F401
#         return True
#     except Exception:
#         return False


# def detect_lightgbm_gpu():
#     if not detect_lightgbm():
#         return False
#     try:
#         import lightgbm as lgb
#         m = lgb.LGBMRegressor(n_estimators=2, device_type="gpu", verbosity=-1)
#         m.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
#         return True
#     except Exception:
#         return False


# def detect_catboost():
#     try:
#         import catboost  # noqa: F401
#         return True
#     except Exception:
#         return False


# def detect_catboost_gpu():
#     if not detect_catboost():
#         return False
#     try:
#         from catboost import CatBoostRegressor
#         m = CatBoostRegressor(iterations=2, task_type="GPU", verbose=0)
#         m.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
#         return True
#     except Exception:
#         return False


GPU_XGB  = detect_gpu_xgb()
# HAS_LGBM = detect_lightgbm()
# GPU_LGBM = detect_lightgbm_gpu()
# HAS_CAT  = detect_catboost()
# GPU_CAT  = detect_catboost_gpu()


# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
def log(msg, level='info'):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {'info': '  ', 'header': '\n', 'sub': '    '}.get(level, '  ')
    print(f"[{ts}] {prefix}{msg}", flush=True)


def section(title, char='='):
    bar = char * 75
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


# ─────────────────────────────────────────────────────────────────
# ASK FOR n_components
# ─────────────────────────────────────────────────────────────────
def ask_n_components():
    print("\nHow many PCA components would you like to use?")
    while True:
        try:
            raw = input("  n_components > ").strip()
            n = int(raw)
            if n < 1:
                print("  ⚠ Must be >= 1.")
                continue
            return n
        except ValueError:
            print(f"  ⚠ Not an integer: {raw!r}")
        except KeyboardInterrupt:
            print("\n  Aborted.")
            sys.exit(0)


def get_n_components():
    """CLI arg if given and valid, else prompt."""
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
            if n < 1:
                raise ValueError("must be >= 1")
            print(f"  → Using CLI arg: n_components = {n}")
            return n
        except ValueError as e:
            print(f"  ⚠ Bad CLI arg ({e}), falling back to prompt.")
    return ask_n_components()


# ─────────────────────────────────────────────────────────────────
# DATA PREP
# ─────────────────────────────────────────────────────────────────
def load_and_prep_data():
    log("Loading data...", level='header')
    df = pd.read_csv(DATA_PATH)

    for col in ('SSTA_Rolling_Mean', 'TSA_Rolling_Mean', 'Unnamed: 0'):
        if col in df.columns:
            df = df.drop(columns=col)

    log(f"Shape: {df.shape}")
    log(f"Target: {TARGET_COL}")

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return {
        'X': X, 'y': y,
        'X_train_num':   X_train_raw.drop(columns=MONTH_COLS),
        'X_test_num':    X_test_raw.drop(columns=MONTH_COLS),
        'X_train_month': X_train_raw[MONTH_COLS].copy().reset_index(drop=True),
        'X_test_month':  X_test_raw[MONTH_COLS].copy().reset_index(drop=True),
        'y_train': y_train.reset_index(drop=True),
        'y_test':  y_test.reset_index(drop=True),
    }


def preprocess(data, n_components):
    """Impute → Scale → PCA → reattach month."""
    imputer = SimpleImputer(strategy='median')
    Xtr_i = imputer.fit_transform(data['X_train_num'])
    Xte_i = imputer.transform(data['X_test_num'])

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_i)
    Xte_s = scaler.transform(Xte_i)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    Xtr_p = pca.fit_transform(Xtr_s)
    Xte_p = pca.transform(Xte_s)

    pc_cols = [f'PC{i+1}' for i in range(Xtr_p.shape[1])]
    Xtr_df = pd.DataFrame(Xtr_p, columns=pc_cols)
    Xte_df = pd.DataFrame(Xte_p, columns=pc_cols)

    X_train_final = pd.concat([Xtr_df, data['X_train_month']], axis=1)
    X_test_final  = pd.concat([Xte_df, data['X_test_month']],  axis=1)

    var_explained = pca.explained_variance_ratio_.sum()
    log(f"PCA: n_components={n_components}, "
        f"variance explained = {var_explained:.4f}, "
        f"final shape = {X_train_final.shape}")
    return X_train_final, X_test_final, var_explained


# ─────────────────────────────────────────────────────────────────
# LAZY PREDICT
# ─────────────────────────────────────────────────────────────────
def lazy_predict_top3(X_train, X_test, y_train, y_test):
    log("Running LazyPredict to find top 3 models...", level='header')
    try:
        from lazypredict.Supervised import LazyRegressor
        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, _ = reg.fit(X_train, X_test, y_train, y_test)
        top3 = models.head(3).index.tolist()
        log(f"Top 3 models by R²: {top3}")
        for name, row in models.head(5).iterrows():
            log(f"  {name:<35} R²={row['R-Squared']:.4f}, RMSE={row['RMSE']:.4f}",
                level='sub')
        return top3
    except Exception as e:
        log(f"LazyPredict failed: {e} — falling back to (ET, RF, XGB)")
        return ['ExtraTreesRegressor', 'RandomForestRegressor', 'XGBRegressor']


# ─────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVES
# ─────────────────────────────────────────────────────────────────
def get_objective(model_name, X_train, y_train):
    """Return Optuna objective function for the given model name."""

    def cv_score(model):
        return cross_val_score(
            Pipeline([('model', model)]), X_train, y_train,
            cv=4, scoring='r2', n_jobs=-1
        ).mean()

    def xgb_kwargs():
        kw = dict(n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)
        if GPU_XGB:
            kw.update(tree_method="hist", device="cuda")
        return kw

    if model_name == 'ExtraTreesRegressor':
        def obj(trial):
            return cv_score(ExtraTreesRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 5, 40),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                n_jobs=-1, random_state=RANDOM_STATE,
            ))
        return obj

    if model_name == 'RandomForestRegressor':
        def obj(trial):
            return cv_score(RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 5, 40),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                n_jobs=-1, random_state=RANDOM_STATE,
            ))
        return obj

    if model_name == 'HistGradientBoostingRegressor':
        def obj(trial):
            return cv_score(HistGradientBoostingRegressor(
                max_iter=trial.suggest_int("max_iter", 50, 400),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50),
                l2_regularization=trial.suggest_float("l2_regularization", 0.0, 1.0),
                random_state=RANDOM_STATE,
            ))
        return obj

    if model_name == 'XGBRegressor':
        def obj(trial):
            return cv_score(xgb.XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 400),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                **xgb_kwargs(),
            ))
        return obj

    if model_name == 'BaggingRegressor':
        def obj(trial):
            base = DecisionTreeRegressor(
                max_depth=trial.suggest_int("base_max_depth", 5, 40),
                random_state=RANDOM_STATE,
            )
            return cv_score(BaggingRegressor(
                estimator=base,
                n_estimators=trial.suggest_int("n_estimators", 10, 200),
                max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
                max_features=trial.suggest_float("max_features", 0.5, 1.0),
                n_jobs=-1, random_state=RANDOM_STATE,
            ))
        return obj

    if model_name == 'GradientBoostingRegressor':
        def obj(trial):
            return cv_score(GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 400),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                random_state=RANDOM_STATE,
            ))
        return obj

    if model_name == 'KNeighborsRegressor':
        def obj(trial):
            return cv_score(KNeighborsRegressor(
                n_neighbors=trial.suggest_int("n_neighbors", 3, 20),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                p=trial.suggest_int("p", 1, 2),
                n_jobs=-1,
            ))
        return obj

    if model_name == 'DecisionTreeRegressor':
        def obj(trial):
            return cv_score(DecisionTreeRegressor(
                max_depth=trial.suggest_int("max_depth", 3, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                random_state=RANDOM_STATE,
            ))
        return obj

    if model_name == 'ExtraTreeRegressor':
        def obj(trial):
            return cv_score(ExtraTreeRegressor(
                max_depth=trial.suggest_int("max_depth", 3, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                random_state=RANDOM_STATE,
            ))
        return obj

    if model_name == 'AdaBoostRegressor':
        def obj(trial):
            return cv_score(AdaBoostRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 1.0),
                random_state=RANDOM_STATE,
            ))
        return obj

    # if model_name == 'LGBMRegressor' and HAS_LGBM:
    #     import lightgbm as lgb

    #     def obj(trial):
    #         kw = dict(
    #             n_estimators=trial.suggest_int("n_estimators", 50, 400),
    #             learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1),
    #             max_depth=trial.suggest_int("max_depth", 3, 15),
    #             num_leaves=trial.suggest_int("num_leaves", 15, 100),
    #             random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    #         )
    #         if GPU_LGBM:
    #             kw['device_type'] = 'gpu'
    #         return cv_score(lgb.LGBMRegressor(**kw))
    #     return obj

    # if model_name == 'CatBoostRegressor' and HAS_CAT:
    #     from catboost import CatBoostRegressor

    #     def obj(trial):
    #         kw = dict(
    #             iterations=trial.suggest_int("iterations", 50, 400),
    #             learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1),
    #             depth=trial.suggest_int("depth", 3, 12),
    #             l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    #             random_state=RANDOM_STATE, verbose=0,
    #         )
    #         if GPU_CAT:
    #             kw['task_type'] = 'GPU'
    #         return cv_score(CatBoostRegressor(**kw))
    #     return obj

    return None


def build_model_from_params(model_name, params):
    """Reconstruct a model with the best params from Optuna."""
    if model_name == 'ExtraTreesRegressor':
        return ExtraTreesRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE)
    if model_name == 'RandomForestRegressor':
        return RandomForestRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE)
    if model_name == 'HistGradientBoostingRegressor':
        return HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    if model_name == 'XGBRegressor':
        kw = dict(**params, n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)
        if GPU_XGB:
            kw.update(tree_method="hist", device="cuda")
        return xgb.XGBRegressor(**kw)
    if model_name == 'BaggingRegressor':
        base_max_depth = params.pop('base_max_depth', None)
        base = DecisionTreeRegressor(max_depth=base_max_depth, random_state=RANDOM_STATE)
        return BaggingRegressor(estimator=base, **params, n_jobs=-1, random_state=RANDOM_STATE)
    if model_name == 'GradientBoostingRegressor':
        return GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    if model_name == 'KNeighborsRegressor':
        return KNeighborsRegressor(**params, n_jobs=-1)
    if model_name == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(**params, random_state=RANDOM_STATE)
    if model_name == 'ExtraTreeRegressor':
        return ExtraTreeRegressor(**params, random_state=RANDOM_STATE)
    if model_name == 'AdaBoostRegressor':
        return AdaBoostRegressor(**params, random_state=RANDOM_STATE)
    # if model_name == 'LGBMRegressor' and HAS_LGBM:
    #     import lightgbm as lgb
    #     kw = dict(**params, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1)
    #     if GPU_LGBM:
    #         kw['device_type'] = 'gpu'
    #     return lgb.LGBMRegressor(**kw)
    # if model_name == 'CatBoostRegressor' and HAS_CAT:
    #     from catboost import CatBoostRegressor
    #     kw = dict(**params, random_state=RANDOM_STATE, verbose=0)
    #     if GPU_CAT:
    #         kw['task_type'] = 'GPU'
    #     return CatBoostRegressor(**kw)
    return None


# ─────────────────────────────────────────────────────────────────
# K-FOLD with PCA refit per fold
# ─────────────────────────────────────────────────────────────────
def kfold_evaluate(model, data, n_components, n_splits=4):
    from sklearn.base import clone

    X, y = data['X'], data['y']
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
        'r2_mean':   np.mean(r2s),   'r2_std':   np.std(r2s),
        'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
        'mae_mean':  np.mean(maes),  'mae_std':  np.std(maes),
        'r2_folds': r2s, 'rmse_folds': rmses, 'mae_folds': maes,
    }


# ─────────────────────────────────────────────────────────────────
# PIPELINE for one n_components value
# ─────────────────────────────────────────────────────────────────
def run_for_n_components(data, n_components):
    section(f"  ▶ N_COMPONENTS = {n_components}", char='=')
    t0 = time.time()

    X_train, X_test, var_explained = preprocess(data, n_components)
    y_train, y_test = data['y_train'], data['y_test']

    top3 = lazy_predict_top3(X_train, X_test, y_train, y_test)

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

        log(f"    Best R²:     {study.best_value:.4f}", level='sub')
        log(f"    Best params: {study.best_params}", level='sub')
        log(f"    Time:        {elapsed:.1f}s", level='sub')

        try:
            tuned_models[name] = build_model_from_params(name, dict(study.best_params))
        except Exception as e:
            log(f"    ❌ Build failed: {e}", level='sub')

    if len(tuned_models) < 2:
        log("⚠️  Less than 2 tuned models — cannot build voting ensemble", level='header')
        return None

    log("\nBuilding VotingRegressor ensemble...", level='header')
    estimators = [(f"m{i}", m) for i, (n, m) in enumerate(tuned_models.items())]
    voting = VotingRegressor(estimators=estimators, n_jobs=-1)

    # Single-split eval
    log("\nSingle-split evaluation:", level='header')
    print(f"  {'Model':<40} {'R²':>8} {'RMSE':>8} {'MAE':>8}", flush=True)
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}", flush=True)

    single_results = {}
    for name, model in tuned_models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            rm = np.sqrt(mean_squared_error(y_test, pred))
            ma = mean_absolute_error(y_test, pred)
            single_results[name] = {'r2': r2, 'rmse': rm, 'mae': ma}
            print(f"  {name:<40} {r2:>8.4f} {rm:>8.4f} {ma:>8.4f}", flush=True)
        except Exception as e:
            log(f"  ❌ {name} failed single-fit: {e}")

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

    # K-Fold eval
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

    try:
        voting_fresh = VotingRegressor(estimators=estimators, n_jobs=-1)
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

    n_components = get_n_components()

    # Hardware / library report
    log(f"XGBoost GPU:    {'YES (CUDA)' if GPU_XGB else 'CPU only'}", level='header')
    # log(f"LightGBM:       {'installed' if HAS_LGBM else 'not installed'}"
    #     + (' (GPU)' if GPU_LGBM else (' (CPU)' if HAS_LGBM else '')))
    # log(f"CatBoost:       {'installed' if HAS_CAT else 'not installed'}"
    #     + (' (GPU)' if GPU_CAT else (' (CPU)' if HAS_CAT else '')))
    log(f"Optuna trials per model: {N_OPTUNA_TRIALS}")
    log(f"K-Fold splits:           {N_KFOLD_SPLITS}")
    log(f"n_components:            {n_components}")
    log(f"Random state:            {RANDOM_STATE}")

    data = load_and_prep_data()

    t_global = time.time()
    result = run_for_n_components(data, n_components)

    # ─── SUMMARY ───────────────────────────────────────────────────
    section("FINAL SUMMARY — Voting Ensemble", char='█')

    if result is None:
        print(f"  Run FAILED for n_components={n_components}", flush=True)
    else:
        sing = result['single'].get('Voting (Ensemble)', {})
        kf   = result['kfold'].get('Voting (Ensemble)', {})
        print(f"\n  n_components       : {n_components}", flush=True)
        print(f"  Variance Explained : {result['variance_explained']:.4f}", flush=True)
        print(f"\n  Single-split  R²   : {sing.get('r2', float('nan')):.4f}", flush=True)
        print(f"  Single-split  RMSE : {sing.get('rmse', float('nan')):.4f}", flush=True)
        print(f"  Single-split  MAE  : {sing.get('mae', float('nan')):.4f}", flush=True)
        print(f"\n  K-Fold  R² mean    : {kf.get('r2_mean', float('nan')):.4f}", flush=True)
        print(f"  K-Fold  RMSE mean  : {kf.get('rmse_mean', float('nan')):.4f}", flush=True)
        print(f"  K-Fold  MAE mean   : {kf.get('mae_mean', float('nan')):.4f}", flush=True)

    log(f"\nTotal time: {(time.time()-t_global)/60:.1f} min")
    log("Done!", level='header')


if __name__ == "__main__":
    main()