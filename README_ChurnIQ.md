# 🚀 ChurnIQ – AI-Based Telecom Churn Prediction Platform

**Base URL:** https://churnapi.dedyn.io/  
**Swagger Docs:** https://churnapi.dedyn.io/docs  

---

## 📌 Problem Statement

Telecom operators face **high customer churn**, leading to:

- 📉 Revenue loss from customer drop-offs  
- ❌ Lack of early visibility into at-risk customers  
- 📊 Fragmented data across customer experience, billing, and network systems  
- ⚠️ Reactive support instead of proactive retention  

Traditional approaches fail because:
- They rely on static reports instead of real-time intelligence  
- They do not combine **network + financial + behavioral signals**  
- They lack actionable insights for business teams  

---

## 💡 Solution Overview

**ChurnIQ** is an **AI-powered churn prediction system** that:

- Predicts **which customers are likely to churn**
- Explains **why they are at risk**
- Recommends **what action to take**

### 🔍 What It Uses

| Category | Signals |
|----------|--------|
| 👤 Customer Behavior | Tenure, contract type |
| 💰 Financial | Monthly charges, total billing |
| 📶 Network | Latency, signal strength, packet loss |
| 📞 Experience | Complaints, resolution time |
| 📊 Usage | Usage drop, engagement trend |

---

## 🎯 Key Capabilities

- 🔮 Predict churn probability
- 🚨 Identify high-risk customers
- 🧠 Explain churn drivers
- ⚡ Recommend corrective actions
- 📊 Provide dashboard analytics
- 🤖 AI Copilot for insights

---

## 🏗️ Solution Architecture

```
Frontend (Rocket.new UI / Dashboard)
        │
        ▼
FastAPI Backend (ChurnIQ API)
        │
        ├── ML Model (XGBoost / RandomForest)
        │       └── churn_model.joblib
        │
        ├── Feature Pipeline
        │       └── feature_columns.joblib
        │
        ├── Data Layer
        │       └── Telecom Dataset (CSV)
        │
        └── Analytics Engine
                └── Aggregations (KPIs, Charts)
```

---

## 🔗 API Endpoints

### Health
GET /health

### KPI Dashboard
GET /dashboard/kpis

### Analytics Dashboard
GET /dashboard/analytics

### High Risk Customers
GET /customers/high-risk

### Potential Churn Customers
GET /customers/potential-churn

### Customer Details
GET /customer/{customerID}

### Predict Churn
POST /predict

### AI Copilot
POST /copilot/query

---

## 🚀 How to Run Locally

```
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload
```

---

## 📘 Swagger UI

👉 https://churnapi.dedyn.io/docs

---

## 🏁 Summary

ChurnIQ transforms telecom data into **actionable intelligence** by predicting churn, explaining reasons, and enabling proactive retention.
