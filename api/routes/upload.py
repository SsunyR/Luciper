from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
import os
import shutil
from datetime import datetime
import sys
import uuid

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.file_utils import ensure_dir_exists
from utils.logging_utils import setup_logger
from utils.audio_utils import extract_audio
from whisper.transcribe import run_transcription

router = APIRouter()
logger = setup_logger(__name__)

# Define base directories relative to project root
VIDEO_UPLOAD_DIR = "data/videos"
AUDIO_EXTRACT_DIR = "data/audios"
TRANSCRIPT_OUTPUT_DIR = "data/transcripts" # Directory for .txt files
SUBTITLE_OUTPUT_DIR = "data/subtitles"   # Directory for .srt files
# Path to the model to use for transcription (Absolute path within container)
WHISPER_MODEL_PATH = "/app/whisper-finetuned-model" # Or path/name from config

ensure_dir_exists(VIDEO_UPLOAD_DIR)
ensure_dir_exists(AUDIO_EXTRACT_DIR)
ensure_dir_exists(TRANSCRIPT_OUTPUT_DIR) # Ensure transcript dir exists
ensure_dir_exists(SUBTITLE_OUTPUT_DIR)

# --- Background Task Function ---
def process_video_and_transcribe(video_path: str, job_id: str):
    """
    Extracts audio, runs transcription, and handles results.
    Designed to be run in the background.
    """
    logger.info(f"[Job {job_id}] Starting background processing for {video_path}")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(AUDIO_EXTRACT_DIR, f"{base_name}.wav")

    # 1. Extract Audio
    if not extract_audio(video_path, audio_path):
        logger.error(f"[Job {job_id}] Failed to extract audio from {video_path}")
        return

    logger.info(f"[Job {job_id}] Audio extracted successfully to {audio_path}")

    # 2. Run Transcription
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcription_result = run_transcription(
        audio_path=audio_path,
        model_path=WHISPER_MODEL_PATH,
        output_txt_dir=TRANSCRIPT_OUTPUT_DIR, # Pass transcript dir
        output_srt_dir=SUBTITLE_OUTPUT_DIR,   # Pass subtitle dir
        device=device
    )

    if transcription_result:
        logger.info(f"[Job {job_id}] Transcription completed successfully.")
    else:
        logger.error(f"[Job {job_id}] Transcription failed for audio {audio_path}")

    logger.info(f"[Job {job_id}] Background processing finished.")


@router.post("/upload/video", status_code=status.HTTP_202_ACCEPTED)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Receives a video file, saves it, and schedules background tasks
    for audio extraction and transcription. Returns a job ID.
    """
    if not file.content_type.startswith("video/"):
        logger.warning(f"Upload failed: Invalid file type '{file.content_type}' from {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid file type. Please upload a video file.",
        )

    try:
        # Create a unique job ID and filename
        job_id = str(uuid.uuid4())
        original_filename = "".join(c for c in file.filename if c.isalnum() or c in ('_', '.', '-')).strip()
        if not original_filename:
            original_filename = "video"
        video_filename = f"{job_id}_{original_filename}"
        video_path = os.path.join(VIDEO_UPLOAD_DIR, video_filename)

        logger.info(f"[Job {job_id}] Receiving file: {file.filename} (Size: {file.size}), saving as {video_filename}")

        # Save the uploaded file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"[Job {job_id}] Successfully saved video file to: {video_path}")

        # Add the processing task to the background
        background_tasks.add_task(process_video_and_transcribe, video_path, job_id)
        logger.info(f"[Job {job_id}] Scheduled background task for audio extraction and transcription.")

        return {
            "message": "Video upload accepted. Processing started in background.",
            "job_id": job_id,
            "video_path": video_path,
        }

    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not upload the file: {e}",
        )
    finally:
        await file.close()

import torch
