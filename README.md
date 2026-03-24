# 🚀 ChurnIQ – AI-Based Wifi Churn Prediction Platform 

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
| 📶 Network | latency, signal strength, packet loss |
| 📞 Experience | complaints, resolution time |
| 📊 Usage | usage drop, consumption trend |

---

## 🧠 AI Model

- Model Used: **XGBoost Classifier**
- Handles tabular data efficiently and captures non-linear churn patterns

### Output:
- Churn Probability (0–1)
- Risk Level (Low / Medium / High)
- Top Drivers
- Recommended Action

---

## 🏗️ Architecture

```
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

### 🔗 Dataset Used

- **Custom Enhanced Telecom Dataset (CSV):**  
  https://github.com/srinivas08/churnpredictai/blob/main/data/final_telco_churn_dataset.csv  

---

### 🧾 Dataset Composition

The dataset is a **hybrid dataset** combining:

#### 1. Base Reference Dataset
- Inspired by industry-standard Telco Customer Churn dataset  
- Features:
  - tenure  
  - MonthlyCharges  
  - TotalCharges  
  - Contract  
  - InternetService  
  - PaymentMethod  

#### 2. Enhanced Features (Added for Real-World Modeling ⭐)

| Category | Added Fields |
|----------|-------------|
| 📶 Network | avg_latency_ms, signal_strength_dbm, packet_loss_rate |
| 📞 Experience | num_complaints, avg_resolution_time |
| 📊 Usage | usage_prev_month, usage_last_month, usage_drop_pct |

---

### 🧠 Why Custom Dataset?

Real telecom churn depends heavily on:

- Network quality (latency, signal)
- Customer experience (complaints)
- Usage behavior (drop in engagement)

These were **engineered to simulate real-world telecom scenarios**.

---

### ⚙️ Churn Label

The `Churn` column (0/1) is derived based on:

- High complaints  
- Poor network quality  
- Usage drop  
- High charges  
- Short tenure  

---

### 📌 Reference Inspiration

- Industry Telco churn datasets (IBM / Kaggle)
- Telecom domain knowledge (network KPIs, support metrics, usage analytics)

---

## 🔗 API Endpoints

- GET /health  
- GET /dashboard/kpis  
- GET /dashboard/analytics  
- GET /customers/high-risk  
- GET /customers/potential-churn  
- GET /customer/{customerID}  
- POST /predict  
- POST /copilot/query  

---

## 🚀 Why This Solution Stands Out

- Combines **network + financial + behavioral signals**  
- Provides **explainability (drivers)**  
- Includes **action recommendations**  
- Production-ready **API + dashboard**  
- Scalable architecture  

---

## 🏁 Conclusion

Transforms wifi data into **actionable AI insights** to predict churn, explain reasons, and enable proactive retention.
