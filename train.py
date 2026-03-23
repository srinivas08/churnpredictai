from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'final_telco_churn_dataset.csv'
MODEL_PATH = ROOT / 'models' / 'churn_model.joblib'
FEATURES_PATH = ROOT / 'models' / 'feature_columns.joblib'

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


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC),
        ('cat', categorical_transformer, CATEGORICAL)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced_subsample'
        ))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    print(f'ROC-AUC: {auc:.4f}')
    print(classification_report(y_test, preds))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out().tolist()
    joblib.dump(feature_names, FEATURES_PATH)
    print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()