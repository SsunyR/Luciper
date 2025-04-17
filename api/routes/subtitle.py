from fastapi import APIRouter, HTTPException, status, Body
from fastapi.responses import FileResponse  # To return files directly
from pydantic import BaseModel
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logging_utils import setup_logger
from utils.file_utils import ensure_dir_exists

router = APIRouter()
logger = setup_logger(__name__)

SUBTITLE_OUTPUT_DIR = "data/subtitles"  # Define consistently

# Placeholder for subtitle data model
class SubtitleData(BaseModel):
    content: str  # Corrected subtitle content (e.g., in SRT format)

@router.get("/subtitle/{job_id}")
async def get_subtitle(job_id: str):
    """
    Retrieves the transcription result (SRT file) for a given job ID.
    """
    logger.info(f"Request received for subtitle with job_id: {job_id}")

    # Construct expected SRT file path based on job_id
    potential_files = [f for f in os.listdir(SUBTITLE_OUTPUT_DIR) if f.startswith(job_id) and f.endswith(".srt")]

    if not potential_files:
        # Check if the original video exists to determine if job is valid but pending/failed
        potential_videos = [f for f in os.listdir("data/videos") if f.startswith(job_id)]
        if potential_videos:
            logger.warning(f"Subtitle file starting with {job_id} not found, but video exists. Job might be processing or failed.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcription is processing or failed.")
        else:
            logger.warning(f"No video or subtitle found for job_id: {job_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")

    # Assume the first match is the correct one
    subtitle_file_path = os.path.join(SUBTITLE_OUTPUT_DIR, potential_files[0])

    if os.path.exists(subtitle_file_path):
        logger.info(f"Found subtitle file: {subtitle_file_path}. Returning content.")
        # Return the SRT file directly
        return FileResponse(subtitle_file_path, media_type="text/plain", filename=os.path.basename(subtitle_file_path))
    else:
        # This case should ideally be caught by the check above, but as a fallback:
        logger.error(f"File path {subtitle_file_path} existed in listing but not found on read attempt.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subtitle file not found.")

@router.put("/subtitle/{job_id}")
async def update_subtitle(job_id: str, subtitle_data: SubtitleData = Body(...)):
    """
    Receives updated/corrected subtitles (in SRT format) from the user.
    Saves the corrected content, potentially overwriting or creating a new file.
    """
    logger.info(f"Received update for subtitle with job_id: {job_id}")

    # Determine the path for the corrected subtitle file
    base_filename = job_id  # Assuming the original SRT might be job_id.srt or similar
    # Find the original video filename base if needed for consistent naming
    potential_videos = [f for f in os.listdir("data/videos") if f.startswith(job_id)]
    if potential_videos:
        base_filename = os.path.splitext(potential_videos[0])[0]  # e.g., job_id_original_video_name

    corrected_subtitle_path = os.path.join(SUBTITLE_OUTPUT_DIR, f"{base_filename}_corrected.srt")  # Save as SRT

    try:
        ensure_dir_exists(SUBTITLE_OUTPUT_DIR)  # Ensure dir exists
        with open(corrected_subtitle_path, "w", encoding="utf-8") as f:
            f.write(subtitle_data.content)  # Save the corrected SRT content
        logger.info(f"Saved corrected subtitle for job_id {job_id} to {corrected_subtitle_path}")

        # Placeholder: Trigger feedback processing (e.g., update_dataset.py)
        # process_feedback(corrected_subtitle_path, job_id)

        return {"message": "Subtitle updated successfully", "job_id": job_id, "corrected_path": corrected_subtitle_path}
    except Exception as e:
        logger.error(f"Failed to save corrected subtitle for job_id {job_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save corrected subtitle.")

