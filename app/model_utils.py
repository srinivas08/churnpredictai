
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'churn_model.joblib'
FEATURES_PATH = ROOT / 'models' / 'feature_columns.joblib'

FEATURE_COLUMNS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService',
    'PaymentMethod', 'avg_latency_ms', 'signal_strength_dbm', 'packet_loss_rate',
    'num_complaints', 'avg_resolution_time', 'usage_prev_month', 'usage_last_month',
    'usage_drop_pct'
]

DRIVER_LABELS = {
    'tenure': 'Low tenure',
    'MonthlyCharges': 'High monthly charges',
    'TotalCharges': 'Low customer lifetime value',
    'Contract': 'Month-to-month contract',
    'InternetService': 'Fiber optic service churn pattern',
    'PaymentMethod': 'Payment method risk pattern',
    'avg_latency_ms': 'High latency',
    'signal_strength_dbm': 'Poor signal strength',
    'packet_loss_rate': 'High packet loss',
    'num_complaints': 'High complaint frequency',
    'avg_resolution_time': 'Slow issue resolution',
    'usage_prev_month': 'Low historical usage',
    'usage_last_month': 'Recent usage decline',
    'usage_drop_pct': 'Usage drop'
}


def load_model_bundle():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names


def predict_with_explanations(payload: Dict) -> Tuple[float, List[str]]:
    model, feature_names = load_model_bundle()
    input_df = pd.DataFrame([payload])
    probability = float(model.predict_proba(input_df)[0][1])

    transformed = model.named_steps['preprocessor'].transform(input_df)
    clf = model.named_steps['classifier']

    if hasattr(clf, 'feature_importances_'):
        contributions = np.abs(transformed.toarray()[0] if hasattr(transformed, 'toarray') else transformed[0]) * clf.feature_importances_
    else:
        contributions = np.zeros(len(feature_names), dtype=float)

    top_idx = np.argsort(contributions)[::-1][:8]
    drivers = []
    seen = set()
    for idx in top_idx:
        raw_name = feature_names[idx]
        base_name = raw_name.split('__')[-1]
        if '_' in raw_name and raw_name.startswith(('cat__','num__')):
            base_name = raw_name.split('__',1)[1]
        for key, label in DRIVER_LABELS.items():
            if key in base_name and label not in seen:
                drivers.append(label)
                seen.add(label)
                break
        if len(drivers) == 3:
            break

    if not drivers:
        drivers = ['Contract pattern', 'Network quality', 'Complaint history']

    return probability, drivers


def recommend_action(payload: Dict, drivers: List[str]) -> str:
    joined = ' | '.join(drivers).lower()
    if 'complaint' in joined:
        return 'Prioritize support callback and fast-track complaint resolution.'
    if 'latency' in joined or 'signal' in joined or 'packet loss' in joined:
        return 'Escalate network remediation and provide temporary service credit.'
    if 'charges' in joined or 'contract' in joined:
        return 'Offer a personalized retention plan or discount upgrade.'
    if 'usage' in joined:
        return 'Launch re-engagement campaign with bonus data pack.'
    return 'Review customer profile and trigger targeted retention outreach.'
