
from fastapi import FastAPI, HTTPException, Query
from app.schemas import CopilotRequest, CopilotResponse, PredictRequest, PredictResponse
from app.service import copilot_query, get_customer, get_high_risk_customers, get_kpis, predict_churn

app = FastAPI(title='ChurnIQ API', version='1.0.0', description='Telecom churn prediction service with dashboard and copilot APIs.')

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/dashboard/kpis')
def dashboard_kpis():
    return get_kpis()

@app.get('/customers/high-risk')
def customers_high_risk(limit: int = Query(20, ge=1, le=100), search: str | None = None):
    return get_high_risk_customers(limit=limit, search=search)

@app.get('/customer/{customer_id}')
def customer_detail(customer_id: str):
    customer = get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail='Customer not found')
    return customer

@app.post('/predict', response_model=PredictResponse)
def predict(payload: PredictRequest):
    return predict_churn(payload.model_dump())

@app.post('/copilot/query', response_model=CopilotResponse)
def copilot(payload: CopilotRequest):
    return copilot_query(payload.message, payload.context)
