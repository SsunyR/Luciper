from fastapi import APIRouter, UploadFile, File, HTTPException, status
import os
import shutil
from datetime import datetime
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.file_utils import ensure_dir_exists
from utils.logging_utils import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

# Define base directory for uploads relative to project root
UPLOAD_DIR = "data/videos"
ensure_dir_exists(UPLOAD_DIR) # Ensure the directory exists on startup

@router.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """
    Receives a video file, saves it to the server, and returns the file path.
    """
    if not file.content_type.startswith("video/"):
        logger.warning(f"Upload failed: Invalid file type '{file.content_type}' from {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid file type. Please upload a video file.",
        )

    try:
        # Create a unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize filename (optional but recommended)
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ('_', '.', '-')).strip()
        if not safe_filename:
            safe_filename = "uploaded_video"
        unique_filename = f"{timestamp}_{safe_filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        logger.info(f"Receiving file: {file.filename} (Size: {file.size}), saving as {unique_filename}")

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Successfully saved video file to: {file_path}")

        # Placeholder: Trigger audio extraction and transcription process here
        # For now, just return the path
        # job_id = start_transcription_job(file_path) # Example function call

        return {
            "message": "Video uploaded successfully",
            "file_path": file_path,
            # "job_id": job_id # Example
            }

    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not upload the file: {e}",
        )
    finally:
        # Ensure the file object is closed
        await file.close()
