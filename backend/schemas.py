# In backend/schemas.py

from pydantic import BaseModel
from datetime import datetime

# This is the Pydantic model (schema) that FastAPI will use for API responses.
# It includes all the fields from your SQLAlchemy model.
class FallDetection(BaseModel):
    id: int
    mac_addr: str | None = None  # Use | None for optional fields
    
    # Accelerometer Features
    max_Ax: float
    min_Ax: float
    var_Ax: float
    mean_Ax: float
    max_Ay: float
    min_Ay: float
    var_Ay: float
    mean_Ay: float
    max_Az: float
    min_Az: float
    var_Az: float
    mean_Az: float

    # Gyroscope Features
    max_Gx: float
    min_Gx: float
    var_Gx: float
    mean_Gx: float
    max_Gy: float
    min_Gy: float
    var_Gy: float
    mean_Gy: float
    max_Gz: float
    min_Gz: float
    var_Gz: float
    mean_Gz: float

    # Prediction and Metadata
    prediction: int
    prediction_label: str
    timestamp: datetime
    source_type: str

    # This Config class is crucial. It tells Pydantic to read data from
    # ORM models (like your SQLAlchemy FallDetection class).
    class Config:
        # orm_mode = True
        from_attributes = True
        # Note: If you are using Pydantic V2, orm_mode is deprecated.
        # Use `from_attributes = True` instead.