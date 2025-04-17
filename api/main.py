from fastapi import FastAPI
from .routes import upload, subtitle # Import routers
import sys
import os

# Add project root to sys.path to allow importing utils etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Luciper API",
    description="API for Whisper fine-tuning pipeline including video upload and transcription.",
    version="0.1.0",
)

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(subtitle.router, prefix="/api/v1", tags=["Subtitle"])

@app.get("/", tags=["Root"])
async def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Luciper API!"}

# Optional: Add startup/shutdown events if needed
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Application startup...")

# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("Application shutdown.")

# To run the server (from project root):
# uvicorn api.main:app --reload --port 8000
