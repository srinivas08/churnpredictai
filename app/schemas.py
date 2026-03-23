
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str
    avg_latency_ms: float
    signal_strength_dbm: float
    packet_loss_rate: float
    num_complaints: int
    avg_resolution_time: float
    usage_prev_month: float
    usage_last_month: float
    usage_drop_pct: float

class PredictResponse(BaseModel):
    churn_probability: float
    risk_level: str
    top_drivers: List[str]
    recommended_action: str

class CopilotRequest(BaseModel):
    message: str = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = None

class CopilotResponse(BaseModel):
    response: str
