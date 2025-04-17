import subprocess
import os
import ffmpeg # Import ffmpeg-python
from .logging_utils import setup_logger # Use relative import
from .file_utils import ensure_dir_exists # Ensure this import is present

logger = setup_logger(__name__)
TARGET_SR = 16000 # Define target sample rate consistent with Whisper

def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file using ffmpeg-python,
    converts it to mono WAV format at TARGET_SR.
    """
    logger.info(f"Attempting to extract audio from {video_path} to {audio_path}")
    ensure_dir_exists(os.path.dirname(audio_path)) # Ensure output directory exists

    try:
        (
            ffmpeg
            .input(video_path)
            # vn: disable video recording
            # acodec: set audio codec to pcm_s16le (standard WAV)
            # ar: set audio sample rate
            # ac: set number of audio channels to 1 (mono)
            .output(audio_path, vn=None, acodec='pcm_s16le', ar=str(TARGET_SR), ac=1)
            .overwrite_output() # Overwrite if exists
            .run(capture_stdout=True, capture_stderr=True) # Use capture_stdout/stderr
        )
        logger.info(f"Successfully extracted audio to {audio_path}")
        return True
    # Catch specific ffmpeg error
    except ffmpeg.Error as e:
        logger.error(f"Error during audio extraction using ffmpeg-python:")
        # Decode stderr for more detailed error message
        stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
        logger.error(f"FFmpeg stderr: {stderr_output}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during audio extraction: {e}", exc_info=True)
        return False

def convert_audio_format(input_path, output_path, target_format="wav"):
    """
    Converts audio file format using FFmpeg.
    """
    # Placeholder implementation
    print(f"Converting {input_path} to {output_path} (format: {target_format})")
    # Example using ffmpeg
    # command = f"ffmpeg -i {input_path} {output_path}"
    # try:
    #     subprocess.run(command, check=True, shell=True, capture_output=True)
    #     print("Audio conversion successful.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during audio conversion: {e.stderr.decode()}")
    #     return False
    return True # Placeholder return

# Add other audio utility functions as needed
