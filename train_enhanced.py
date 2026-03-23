# train_enhanced.py
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import classification_report, roc_auc_score

# ----------------------------- CONFIG -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'final_telco_churn_dataset.csv'
MODEL_PATH = ROOT / 'models' / 'churn_model_enhanced.joblib'
FEATURES_PATH = ROOT / 'models' / 'feature_columns_enhanced.joblib'

TARGET = 'Churn'

FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService',
    'PaymentMethod', 'avg_latency_ms', 'signal_strength_dbm', 'packet_loss_rate',
    'num_complaints', 'avg_resolution_time', 'usage_prev_month', 'usage_last_month',
    'usage_drop_pct'
]

NUMERIC = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'avg_latency_ms', 'signal_strength_dbm',
    'packet_loss_rate', 'num_complaints', 'avg_resolution_time', 'usage_prev_month',
    'usage_last_month', 'usage_drop_pct'
]

CATEGORICAL = ['Contract', 'InternetService', 'PaymentMethod']

N_TRIALS = 20          # Increase for better tuning (slower)
RANDOM_STATE = 42
# ------------------------------------------------------------------


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
    }

    model = xgb.XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        eval_metric='auc',
        scale_pos_weight=3.0,          # helps with churn imbalance
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    return auc


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Train-test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Further split for validation (used in Optuna)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
    )

    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC),
        ('cat', categorical_transformer, CATEGORICAL)
    ])

    # Apply preprocessing
    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)

    # Apply SMOTE on training data only
    print("Applying SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)

    # Hyperparameter tuning with Optuna
    print(f"Starting Optuna tuning with {N_TRIALS} trials...")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(lambda trial: objective(trial, X_train_res, y_train_res, X_val_prep, y_val), n_trials=N_TRIALS)

    print(f"Best ROC-AUC from tuning: {study.best_value:.4f}")
    print("Best parameters:", study.best_params)

    # Train final model with best params
    best_params = study.best_params
    final_model = xgb.XGBClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        eval_metric='auc',
        scale_pos_weight=3.0,
        use_label_encoder=False
    )

    print("Training final model on full training data...")
    final_model.fit(X_train_res, y_train_res)

    # Evaluate on test set
    test_probs = final_model.predict_proba(X_test_prep)[:, 1]
    test_preds = final_model.predict(X_test_prep)

    auc = roc_auc_score(y_test, test_probs)
    print(f"\nFinal Test ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))

    # Save the full pipeline (preprocessor + classifier)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', final_model)
    ])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipeline, MODEL_PATH)

    # Save feature names (after one-hot encoding)
    feature_names = preprocessor.get_feature_names_out().tolist()
    joblib.dump(feature_names, FEATURES_PATH)

    print(f"\nEnhanced model saved to: {MODEL_PATH}")
    print(f"Feature columns saved to: {FEATURES_PATH}")
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
