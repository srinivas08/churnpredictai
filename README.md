# ChurnIQ Telecom Churn Prediction

ChurnIQ is a production-style reference implementation for telecom churn prediction using customer profile, financial, network quality, and customer experience signals.

## What this repo includes
- Model training pipeline using scikit-learn
- FastAPI backend for churn scoring and dashboard APIs
- Simple AI copilot endpoint grounded in churn patterns from the dataset
- Sample enriched telecom churn dataset
- Docker support

## Architecture
Source systems such as CRM, billing, network monitoring, and usage analytics are consolidated into a unified feature dataset. A binary classification model predicts churn probability and returns top drivers plus a recommended retention action.

## API endpoints
- `GET /health`
- `GET /dashboard/kpis`
- `GET /customers/high-risk?limit=20&search=`
- `GET /customer/{customer_id}`
- `POST /predict`
- `POST /copilot/query`

## Request example: `/predict`
```json
{
  "tenure": 4,
  "MonthlyCharges": 89.5,
  "TotalCharges": 358.0,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check",
  "avg_latency_ms": 148.0,
  "signal_strength_dbm": -103.0,
  "packet_loss_rate": 0.03,
  "num_complaints": 3,
  "avg_resolution_time": 36.0,
  "usage_prev_month": 40.0,
  "usage_last_month": 24.0,
  "usage_drop_pct": 0.40
}
```

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```

Open Swagger UI at `http://localhost:8000/docs`.

## Docker
```bash
docker compose up --build
```

## Notes
- The included dataset is a realistic synthetic telco dataset enriched with network, complaint, and behavioral features.
