from pathlib import Path

import joblib
import pandas as pd

from app.schemas import CopilotRequest, PredictRequest

DATA_PATH = Path("data/final_telco_churn_dataset.csv")
MODEL_PATH = Path("models/churn_model.joblib")
FEATURES_PATH = Path("models/feature_columns.joblib")

df = pd.read_csv(DATA_PATH)
pipeline = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = joblib.load(FEATURES_PATH)


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def _top_drivers(req: PredictRequest) -> list[str]:
    drivers = []

    if req.num_complaints >= 3:
        drivers.append("High complaint frequency")
    if req.avg_latency_ms >= 140:
        drivers.append("High network latency")
    if req.signal_strength_dbm <= -95:
        drivers.append("Poor signal strength")
    if req.usage_drop_pct >= 0.25:
        drivers.append("Recent usage drop")
    if req.Contract == "Month-to-month":
        drivers.append("Month-to-month contract")
    if req.MonthlyCharges >= 80:
        drivers.append("High monthly charges")
    if req.packet_loss_rate >= 0.05:
        drivers.append("High packet loss")

    return drivers[:3] if drivers else ["General churn risk pattern"]


def _recommended_action(drivers: list[str]) -> str:
    if "High complaint frequency" in drivers:
        return "Priority support callback and service recovery offer"
    if (
        "High network latency" in drivers
        or "Poor signal strength" in drivers
        or "High packet loss" in drivers
    ):
        return "Escalate network issue and provide temporary compensation"
    if "High monthly charges" in drivers:
        return "Offer a better-value plan or retention discount"
    if "Recent usage drop" in drivers:
        return "Run personalized re-engagement campaign"
    if "Month-to-month contract" in drivers:
        return "Offer long-term loyalty plan upgrade"
    return "Review account and trigger standard retention workflow"


def get_kpis():
    churn_mask = df["Churn"] == 1

    features = df.drop(columns=["Churn", "customerID"], errors="ignore")
    features = features[FEATURE_COLUMNS]
    probabilities = pipeline.predict_proba(features)[:, 1]

    high_risk_count = int((probabilities >= 0.7).sum())
    medium_risk_count = int(((probabilities >= 0.4) & (probabilities < 0.7)).sum())
    low_risk_count = int((probabilities < 0.4).sum())

    return {
        "total_customers": int(len(df)),
        "churn_rate": round(float(df["Churn"].mean()), 4),
        "revenue_at_risk": round(float(df.loc[churn_mask, "MonthlyCharges"].sum()), 2),
        "avg_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
        "avg_tenure": round(float(df["tenure"].mean()), 2),
        "avg_latency": round(float(df["avg_latency_ms"].mean()), 2),
        "high_risk_customers": high_risk_count,
        "medium_risk_customers": medium_risk_count,
        "low_risk_customers": low_risk_count,
    }


def get_high_risk_customers():
    scored = df.copy()

    features = scored.drop(columns=["Churn", "customerID"], errors="ignore")
    # Align to trained feature order just in case
    features = features[FEATURE_COLUMNS]

    probabilities = pipeline.predict_proba(features)[:, 1]
    scored["churn_probability"] = probabilities
    scored["risk_level"] = scored["churn_probability"].apply(_risk_level)

    ranked = scored.sort_values("churn_probability", ascending=False).head(20)
    return ranked.to_dict(orient="records")


def get_customer(customer_id: str):
    customer = df[df["customerID"] == customer_id]
    if customer.empty:
        return {"error": "Customer not found"}

    row = customer.iloc[0].to_dict()

    feature_input = pd.DataFrame(
        [
            {
                "tenure": row["tenure"],
                "MonthlyCharges": row["MonthlyCharges"],
                "TotalCharges": row["TotalCharges"],
                "Contract": row["Contract"],
                "InternetService": row["InternetService"],
                "PaymentMethod": row["PaymentMethod"],
                "avg_latency_ms": row["avg_latency_ms"],
                "signal_strength_dbm": row["signal_strength_dbm"],
                "packet_loss_rate": row["packet_loss_rate"],
                "num_complaints": row["num_complaints"],
                "avg_resolution_time": row["avg_resolution_time"],
                "usage_prev_month": row["usage_prev_month"],
                "usage_last_month": row["usage_last_month"],
                "usage_drop_pct": row["usage_drop_pct"],
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    probability = float(pipeline.predict_proba(feature_input)[0][1])
    row["churn_probability"] = round(probability, 4)
    row["risk_level"] = _risk_level(probability)

    return row


def predict_churn(req: PredictRequest):
    payload = req.model_dump()

    filtered = {col: payload.get(col) for col in FEATURE_COLUMNS}
    input_df = pd.DataFrame([filtered], columns=FEATURE_COLUMNS)

    probability = float(pipeline.predict_proba(input_df)[0][1])
    drivers = _top_drivers(req)

    return {
        "churn_probability": round(probability, 4),
        "risk_level": _risk_level(probability),
        "top_drivers": drivers,
        "recommended_action": _recommended_action(drivers),
    }


def copilot_response(req: CopilotRequest):
    msg = req.message.lower()
    context = req.context or {}

    if "fiber" in msg:
        return {
            "response": "Fiber optic users can show higher churn when higher charges combine with service quality issues such as complaints, latency, and weaker perceived value."
        }

    if "high risk" in msg:
        return {
            "response": "High-risk customers usually have short tenure, month-to-month contracts, more complaints, weaker signal, higher latency, and noticeable usage drop."
        }

    if "revenue" in msg and "risk" in msg:
        revenue_at_risk = round(
            float(df.loc[df["Churn"] == 1, "MonthlyCharges"].sum()), 2
        )
        return {
            "response": f"Estimated revenue at risk from churned customers in the current dataset is {revenue_at_risk} based on monthly charges."
        }

    if "why" in msg and context.get("customerID"):
        customer_id = context["customerID"]
        customer = get_customer(customer_id)
        if "error" not in customer:
            return {
                "response": (
                    f"Customer {customer_id} is {customer['risk_level']} risk "
                    f"with churn probability {customer['churn_probability']}. "
                    f"Key factors include complaints={customer['num_complaints']}, "
                    f"latency={customer['avg_latency_ms']}, "
                    f"signal={customer['signal_strength_dbm']}, "
                    f"usage_drop={round(customer['usage_drop_pct'], 3)}, "
                    f"contract={customer['Contract']}."
                )
            }

    if "top churn drivers" in msg or "drivers" in msg:
        return {
            "response": "The main churn drivers in this solution are complaint frequency, contract type, monthly charges, signal strength, latency, packet loss, and usage drop."
        }

    return {
        "response": "Main churn drivers are usually complaints, contract type, monthly charges, signal strength, latency, packet loss, and usage drop. Use the prediction panel for customer-specific risk."
    }
