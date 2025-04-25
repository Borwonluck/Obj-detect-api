import cv2
import numpy as np
import base64
import logging
import shutil
import os
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from datetime import datetime, timedelta
from ultralytics import YOLO
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from io import BytesIO
from PIL import Image

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy Base
class Base(DeclarativeBase):
    pass

# Database Configuration
DB_USER = "postgres"
DB_PASS = "pce112"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "bitterdb"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Adjust timestamp to UTC+7
def get_utc_plus_7():
    return datetime.utcnow() + timedelta(hours=7)

# Table Schema
class DetectionResult(Base):
    __tablename__ = "detection_results"
    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String, index=True)
    confidence = Column(Float, nullable=True)
    seed_count = Column(Integer, default=0)
    width_cm = Column(Float, nullable=True)
    height_cm = Column(Float, nullable=True)
    avgBseedWidth = Column(Float, nullable=True)
    avgBseedHeight = Column(Float, nullable=True)
    output_image = Column(Text)
    created_at = Column(DateTime, default=get_utc_plus_7)

Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(title="Bitterbean Detection API", version="1.0.0")

# Detect GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Load YOLO Model
model = YOLO("my_model.pt").to(DEVICE)

# Response Models
class ImageResponse(BaseModel):
    detections: list
    seedCount: int
    avgBseedWidth: float
    avgBseedHeight: float
    processedImage: str
    createdAt: str

class DetectionHistoryResponse(BaseModel):
    id: int
    class_name: str
    confidence: float
    seed_count: int
    width_cm: float | None
    height_cm: float | None
    avgBseedWidth: float | None
    avgBseedHeight: float | None
    output_image: str
    created_at: str

# Dependency for Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Convert Image to Base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode("utf-8")

# Measure Object Size
def measure_object_size(box, px_to_cm=0.007):
    width_px = box.xyxy[0][2].item() - box.xyxy[0][0].item()
    height_px = box.xyxy[0][3].item() - box.xyxy[0][1].item()
    return round(width_px * px_to_cm, 2), round(height_px * px_to_cm, 2)

@app.post("/detect", response_model=ImageResponse)
def detect_objects(file: UploadFile = File(...), db: Session = Depends(get_db)):
    temp_filename = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        frame = cv2.imread(temp_filename)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        results = model(frame, device=DEVICE)
        annotated_frame = results[0].plot()
        detections, seedCount, bseed_sizes = [], 0, []

        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls[0])]
                conf = round(box.conf[0].item(), 2)
                width_cm, height_cm = measure_object_size(box)

                if cls == "bseed":
                    seedCount += 1
                    bseed_sizes.append((width_cm, height_cm))
                    continue  

                if cls in ["ripe", "unripe", "overripe"]:
                    detections.append({"class": cls, "confidence": conf, "width_cm": width_cm, "height_cm": height_cm})

        avg_bseed_width = round(sum(w for w, _ in bseed_sizes) / len(bseed_sizes), 2) if bseed_sizes else 0
        avg_bseed_height = round(sum(h for _, h in bseed_sizes) / len(bseed_sizes), 2) if bseed_sizes else 0
        encoded_image = image_to_base64(annotated_frame)

        created_at_utc7 = get_utc_plus_7()

        for det in detections:
            db.add(DetectionResult(
                class_name=det["class"],
                confidence=det["confidence"],
                seed_count=seedCount,
                width_cm=det["width_cm"],
                height_cm=det["height_cm"],
                avgBseedWidth=avg_bseed_width,
                avgBseedHeight=avg_bseed_height,
                output_image=encoded_image,
                created_at=created_at_utc7
            ))
        db.commit()

        return ImageResponse(
            detections=detections,
            seedCount=seedCount,
            avgBseedWidth=avg_bseed_width,
            avgBseedHeight=avg_bseed_height,
            processedImage=encoded_image,
            createdAt=created_at_utc7.isoformat()
        )
    
    except Exception as e:
        db.rollback()
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/history")
def get_detection_history(db: Session = Depends(get_db)):
    try:
        records = db.query(DetectionResult).all()
        
        results = []
        for record in records:
            image_data = None
            
            if record.output_image:
                try:
                    img_bytes = base64.b64decode(record.output_image)
                    img = Image.open(BytesIO(img_bytes))
                    img = img.convert("RGB")
                    img.thumbnail((300, 300))
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=30)
                    image_data = base64.b64encode(buffer.getvalue()).decode()
                except Exception as e:
                    logger.error(f"Error processing image for record {record.id}: {e}")

            results.append({
                "id": record.id,
                "class_name": record.class_name,
                "confidence": record.confidence,
                "seed_count": record.seed_count,
                "width_cm": record.width_cm,
                "height_cm": record.height_cm,
                "avgBseedWidth": record.avgBseedWidth,
                "avgBseedHeight": record.avgBseedHeight,
                "created_at": (record.created_at).isoformat(),  # Already stored as UTC+7
                "output_image": image_data
            })

        return results
    except Exception as e:
        logger.error(f"Error fetching detection history: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("bitterbean_api:app", host="0.0.0.0", port=9000, reload=True)
