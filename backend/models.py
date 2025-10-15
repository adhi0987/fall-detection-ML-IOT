# from sqlalchemy import Column, Integer, Float, String, DateTime
# from sqlalchemy.sql import func
# from database import Base

# class FallDetection(Base):
#     __tablename__ = "fall_detections"

#     id = Column(Integer, primary_key=True, index=True)
#     mac_addr = Column(String, index=True)
#     max_Ax = Column(Float)
#     min_Ax = Column(Float)
#     var_Ax = Column(Float)
#     mean_Ax = Column(Float)
#     max_Ay = Column(Float)
#     min_Ay = Column(Float)
#     var_Ay = Column(Float)
#     mean_Ay = Column(Float)
#     max_Az = Column(Float)
#     min_Az = Column(Float)
#     var_Az = Column(Float)
#     mean_Az = Column(Float)
#     max_pitch = Column(Float)
#     min_pitch = Column(Float)
#     var_pitch = Column(Float)
#     mean_pitch = Column(Float)
#     prediction = Column(Integer)
#     prediction_label = Column(String)
#     timestamp = Column(DateTime(timezone=True), server_default=func.now())

from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class FallDetection(Base):
    __tablename__ = "fall_detections"

    id = Column(Integer, primary_key=True, index=True)
    mac_addr = Column(String, index=True)
    
    # Accelerometer Features
    max_Ax = Column(Float)
    min_Ax = Column(Float)
    var_Ax = Column(Float)
    mean_Ax = Column(Float)
    max_Ay = Column(Float)
    min_Ay = Column(Float)
    var_Ay = Column(Float)
    mean_Ay = Column(Float)
    max_Az = Column(Float)
    min_Az = Column(Float)
    var_Az = Column(Float)
    mean_Az = Column(Float)

    # Gyroscope Features (New)
    max_Gx = Column(Float)
    min_Gx = Column(Float)
    var_Gx = Column(Float)
    mean_Gx = Column(Float)
    max_Gy = Column(Float)
    min_Gy = Column(Float)
    var_Gy = Column(Float)
    mean_Gy = Column(Float)
    max_Gz = Column(Float)
    min_Gz = Column(Float)
    var_Gz = Column(Float)
    mean_Gz = Column(Float)

    # Prediction and Metadata
    prediction = Column(Integer)
    prediction_label = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    source_type = Column(String) # "predicted" or "labelled"