# 🚀 Nomiso ChurnIQ – AI-Based Wifi Churn Prediction Platform 


## 🌐 Live Access

- **Dashboard URL:** https://churniq-np03528.public.builtwithrocket.new  
- **Swagger API Docs:** https://churnapi.dedyn.io/docs  
- **Base API URL:** https://churnapi.dedyn.io/  

---

## 📌 Problem Statement

A telecom operator is experiencing **high customer churn**, leading to revenue loss and reduced customer lifetime value.

They require an **AI-driven system** that can:

- Identify customers likely to churn in advance  
- Understand the key reasons behind churn  
- Enable proactive corrective actions  

The solution should leverage signals across:
- Customer experience  
- Financial behavior  
- Network quality  

---

## 🎯 Objective

Build an **end-to-end AI-based churn prediction system** that:

- Predicts churn probability for each customer  
- Classifies risk levels (Low, Medium, High)  
- Explains key churn drivers  
- Enables actionable business decisions  

---

## 💡 Solution Approach

### 🔍 Data Signals Used

| Category | Features |
|----------|--------|
| 👤 Customer | tenure, contract type |
| 💰 Financial | MonthlyCharges, TotalCharges |
| 📶 Network | avg_latency_ms, signal_strength_dbm, packet_loss_rate |
| 📞 Experience | num_complaints, avg_resolution_time |
| 📊 Usage | usage_prev_month, usage_last_month, usage_drop_pct |

---

## 🧠 AI Model

- **Model Used:** XGBoost Classifier  
- **Why XGBoost:** Strong performance on structured/tabular telecom data, handles non-linear relationships well, and is a practical industry choice for churn prediction.

### Model Output
- Churn Probability (0–1)
- Risk Level (Low / Medium / High)
- Top Drivers
- Recommended Retention Action

---

## 🏗️ Architecture

```text
Frontend (Rocket.new Dashboard UI)
        │
        ▼
FastAPI Backend (ChurnIQ API)
        │
        ├── ML Model (XGBoost)
        ├── Feature Engineering Layer
        ├── Dataset (CSV)
        └── Analytics Engine
```

---

## 📂 Dataset & References

### Dataset Used

- **Custom Enhanced Telecom Dataset (CSV):**  
  https://github.com/srinivas08/churnpredictai/blob/main/data/final_telco_churn_dataset.csv  

### Dataset Composition

The dataset is a **hybrid dataset** combining:

#### 1. Base Reference Dataset
Inspired by industry-standard telco churn datasets with common fields such as:
- tenure
- MonthlyCharges
- TotalCharges
- Contract
- InternetService
- PaymentMethod

#### 2. Enhanced Features Added for Real-World Modeling

| Category | Added Fields |
|----------|-------------|
| 📶 Network | avg_latency_ms, signal_strength_dbm, packet_loss_rate |
| 📞 Experience | num_complaints, avg_resolution_time |
| 📊 Usage | usage_prev_month, usage_last_month, usage_drop_pct |

### Why a Custom Dataset Was Used

Real telecom churn depends heavily on:
- network quality
- complaints and service recovery
- customer usage behavior

These signals are typically missing in standard churn datasets, so realistic telecom-oriented features were added to simulate production scenarios.

### Churn Label

The `Churn` column (`0/1`) is derived from a combination of:
- complaint intensity
- network degradation
- usage drop
- price pressure
- short tenure

### Reference Inspiration

- Industry telco churn datasets (IBM / Kaggle style structure)
- Telecom domain knowledge covering network KPIs, support metrics, and usage analytics

---

## 🔗 API Endpoints

- `GET /health`
- `GET /dashboard/kpis`
- `GET /dashboard/analytics`
- `GET /customers/high-risk`
- `GET /customers/potential-churn`
- `GET /customer/{customerID}`
- `POST /predict`
- `POST /copilot/query`

---

## 🔮 Sample Prediction

### Request

```json
{
  "tenure": 3,
  "MonthlyCharges": 95.5,
  "TotalCharges": 286.5,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check",
  "avg_latency_ms": 180,
  "signal_strength_dbm": -105,
  "packet_loss_rate": 0.08,
  "num_complaints": 4,
  "avg_resolution_time": 48,
  "usage_prev_month": 120,
  "usage_last_month": 70,
  "usage_drop_pct": 0.42
}
```

### Response

```json
{
  "churn_probability": 0.91,
  "risk_level": "High",
  "top_drivers": [
    "High complaint frequency",
    "Poor signal strength",
    "Recent usage drop"
  ],
  "recommended_action": "Priority support callback and service recovery offer"
}
```

---

## 📊 How Churn Is Identified

- 🔴 **High Risk** → probability ≥ 0.7
- 🟠 **Medium Risk** → 0.4 to 0.7
- 🟢 **Low Risk** → probability < 0.4

---

## ⚙️ Model Execution Steps

### 1. Prepare Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This generates:
- `churn_model.joblib`
- `feature_columns.joblib`

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

### 4. Test Health Check

```bash
curl http://127.0.0.1:8000/health
```

### 5. Test Prediction API

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "tenure": 3,
  "MonthlyCharges": 95.5,
  "TotalCharges": 286.5,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check",
  "avg_latency_ms": 180,
  "signal_strength_dbm": -105,
  "packet_loss_rate": 0.08,
  "num_complaints": 4,
  "avg_resolution_time": 48,
  "usage_prev_month": 120,
  "usage_last_month": 70,
  "usage_drop_pct": 0.42
}'
```

---

## 🛠️ Installation Steps

### Clone the Repository

```bash
git clone https://github.com/srinivas08/churnpredictai.git
cd churnpredictai
```

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Files Required Before Running API

Make sure the following exist:
- `data/final_telco_churn_dataset.csv`
- `train.py`
- `app/main.py`
- `app/service.py`
- `app/schemas.py`

If model files are missing, run:

```bash
python train.py
```

---

## 🚀 Deployment Steps

### Local Deployment

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Production Deployment (Example)

Using `systemd` / process manager or container hosting:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Recommended Production Setup

- Reverse proxy: Nginx
- App server: Uvicorn / Gunicorn + Uvicorn workers
- TLS: HTTPS termination at Nginx / load balancer
- Health check path: `/health`
- API docs path: `/docs`

### Environment / Ops Notes

- Persist `churn_model.joblib` and `feature_columns.joblib`
- Ensure dataset path is correct on server
- Restrict CORS as needed for dashboard domain
- Add request logging and monitoring for production
- Use process supervision (systemd, pm2, Docker, or platform-native service manager)

---

## 🐳 Optional Docker Deployment

Example commands:

```bash
docker build -t churniq-api .
docker run -p 8000:8000 churniq-api
```

If using Docker Compose:

```bash
docker-compose up --build
```

---

## 📘 Swagger

- **Swagger URL:** https://churnapi.dedyn.io/docs

---

## ⚡ Business Impact

- Reduce churn proactively
- Protect revenue at risk
- Improve customer satisfaction
- Enable targeted retention strategies
- Give business and support teams customer-level AI insights

---

## 🚀 Why This Solution Stands Out

- Combines **network + financial + behavioral signals**
- Provides **explainability through churn drivers**
- Includes **action recommendations**, not just scores
- Exposes a production-style **API + dashboard**
- Designed to scale from single-customer scoring to bulk high-risk identification

---

## 🏁 Conclusion

ChurnIQ transforms telecom data into **actionable AI insights** to:
- predict churn early
- explain root causes
- enable proactive retention action

This makes it useful not only as a machine learning model, but as a practical business decision system for telecom operators.

