from fastapi import FastAPI

from app.schemas import CopilotRequest, PredictRequest
from app.service import (
    copilot_response,
    get_customer,
    get_dashboard_analytics,
    get_high_risk_customers,
    get_kpis,
    predict_churn,
)

app = FastAPI(title="ChurnIQ API", version="1.0.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "churniq-api",
        "model_loaded": True,
    }


@app.get("/dashboard/kpis")
def dashboard_kpis():
    return get_kpis()


@app.get("/dashboard/analytics")
def dashboard_analytics():
    return get_dashboard_analytics()


@app.get("/customers/high-risk")
def high_risk():
    return get_high_risk_customers()


@app.get("/customer/{customer_id}")
def customer(customer_id: str):
    return get_customer(customer_id)


@app.post("/predict")
def predict(req: PredictRequest):
    return predict_churn(req)


@app.post("/copilot/query")
def copilot(req: CopilotRequest):
    return copilot_response(req)
