from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from app.model_utils import FEATURE_COLUMNS, predict_with_explanations, recommend_action

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "final_telco_churn_dataset.csv"

df = pd.read_csv(DATA_PATH)


def _compute_risk_score(dataframe: pd.DataFrame) -> pd.Series:
    score = (
        dataframe["num_complaints"].fillna(0) * 0.25
        + dataframe["avg_latency_ms"].fillna(0) * 0.005
        + dataframe["packet_loss_rate"].fillna(0) * 5
        + dataframe["usage_drop_pct"].fillna(0) * 1.5
        + (dataframe["Contract"].eq("Month-to-month")).astype(float) * 0.75
        + (dataframe["InternetService"].eq("Fiber optic")).astype(float) * 0.25
        + (1 - dataframe["tenure"].clip(lower=1, upper=72) / 72.0)
    )
    return score


def get_kpis() -> Dict[str, Any]:
    revenue_at_risk = float(df.loc[df["Churn"] == 1, "MonthlyCharges"].sum())
    return {
        "total_customers": int(len(df)),
        "churn_rate": round(float(df["Churn"].mean()), 4),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "avg_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
        "avg_tenure": round(float(df["tenure"].mean()), 2),
        "avg_latency_ms": round(float(df["avg_latency_ms"].mean()), 2),
    }


def get_high_risk_customers(limit: int = 20, search: Optional[str] = None):
    ranked = df.copy()
    ranked["risk_score"] = _compute_risk_score(ranked)
    if search:
        mask = ranked["customerID"].str.contains(search, case=False, na=False)
        ranked = ranked[mask]
    cols = [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "avg_latency_ms",
        "signal_strength_dbm",
        "packet_loss_rate",
        "num_complaints",
        "avg_resolution_time",
        "usage_prev_month",
        "usage_last_month",
        "usage_drop_pct",
        "Churn",
        "risk_score",
    ]
    return (
        ranked.sort_values("risk_score", ascending=False)[cols]
        .head(limit)
        .to_dict(orient="records")
    )


def get_customer(customer_id: str):
    customer = df.loc[df["customerID"] == customer_id]
    if customer.empty:
        return None
    record = customer.iloc[0].to_dict()
    record["risk_score"] = float(_compute_risk_score(customer).iloc[0])
    return record


def predict_churn(payload: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {key: payload[key] for key in FEATURE_COLUMNS}
    probability, drivers = predict_with_explanations(filtered)
    risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"
    action = recommend_action(filtered, drivers)
    return {
        "churn_probability": round(probability, 4),
        "risk_level": risk,
        "top_drivers": drivers,
        "recommended_action": action,
    }


def copilot_response(req):
    msg = req.message.lower()

    if "fiber" in msg:
        return {
            "response": "Fiber users may show higher churn when high charges and poor service quality combine. Check complaints, latency, and contract type for that segment."
        }

    if "high risk" in msg:
        return {
            "response": "High-risk customers usually have short tenure, more complaints, weaker signal, higher latency, and noticeable usage drop."
        }

    if "why" in msg and req.context and req.context.get("customerID"):
        customer_id = req.context["customerID"]
        customer = get_customer(customer_id)
        if "error" not in customer:
            return {
                "response": (
                    f"Customer {customer_id} is {customer['risk_level']} risk "
                    f"with churn probability {customer['churn_probability']}. "
                    f"Key factors include complaints={customer['num_complaints']}, "
                    f"latency={customer['avg_latency_ms']}, "
                    f"signal={customer['signal_strength_dbm']}, "
                    f"usage_drop={round(customer['usage_drop_pct'], 3)}."
                )
            }

    return {
        "response": "Churn in this dataset is mainly influenced by tenure, month-to-month contracts, complaints, latency, signal quality, and usage drop. Use customer context to get a more specific explanation."
    }
