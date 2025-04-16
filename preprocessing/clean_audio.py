import os
import librosa
import soundfile as sf
import numpy as np
import shutil
from utils.logging_utils import setup_logger
from utils.file_utils import ensure_dir_exists

logger = setup_logger(__name__)

# Define desired audio properties
TARGET_SR = 16000 # Whisper expects 16kHz

def normalize_volume(input_audio_path, output_audio_path, target_db=-20.0):
    """
    Normalizes the volume of an audio file to a target level (RMS).
    Converts to mono and resamples to TARGET_SR.
    """
    try:
        # Load audio, convert to mono, resample
        y, sr = librosa.load(input_audio_path, sr=TARGET_SR, mono=True)

        # Calculate current RMS level
        rms = np.sqrt(np.mean(y**2))
        if rms == 0: # Avoid division by zero for silent files
             logger.warning(f"Audio file {input_audio_path} is silent. Skipping normalization.")
             # Save the resampled (potentially silent) audio
             sf.write(output_audio_path, y, TARGET_SR)
             return True

        # Calculate gain needed to reach target dB
        target_rms = 10**(target_db / 20.0)
        gain = target_rms / rms

        # Apply gain
        y_normalized = y * gain

        # Optional: Prevent clipping (though normalization usually lowers volume)
        y_normalized = np.clip(y_normalized, -1.0, 1.0)

        # Save normalized audio
        sf.write(output_audio_path, y_normalized, TARGET_SR)
        logger.debug(f"Normalized volume for {input_audio_path} to {output_audio_path}")
        return True
    except Exception as e:
        logger.error(f"Error during volume normalization for {input_audio_path}: {e}")
        # Fallback: copy original if processing fails? Or just return False?
        # shutil.copy(input_audio_path, output_audio_path)
        return False

def remove_silence(input_audio_path, output_audio_path, top_db=30):
    """
    Removes leading/trailing silence from an audio file using librosa.effects.trim.
    Assumes audio is already mono and at TARGET_SR from previous steps if chained.
    """
    try:
        # Load audio (should ideally be already processed by normalize_volume)
        y, sr = librosa.load(input_audio_path, sr=None) # Load with its current SR

        if sr != TARGET_SR:
             logger.warning(f"Audio {input_audio_path} has sr={sr}, expected {TARGET_SR}. Resampling.")
             y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
             sr = TARGET_SR # Update sample rate

        # Trim silence
        y_trimmed, index = librosa.effects.trim(y, top_db=top_db)

        if len(y_trimmed) == 0:
            logger.warning(f"Trimming removed all audio from {input_audio_path}. Saving original.")
            # If trimming results in empty audio, maybe save the original or a tiny bit of noise?
            # For now, let's save the original untrimmed version.
            shutil.copy(input_audio_path, output_audio_path)
            # Or save the trimmed (empty) version: sf.write(output_audio_path, y_trimmed, sr)
        else:
            # Save trimmed audio
            sf.write(output_audio_path, y_trimmed, sr)
            logger.debug(f"Removed silence from {input_audio_path} to {output_audio_path}")

        return True
    except Exception as e:
        logger.error(f"Error during silence removal for {input_audio_path}: {e}")
        # Fallback: copy original if processing fails?
        # shutil.copy(input_audio_path, output_audio_path)
        return False

def process_audio_directory(input_dir, output_dir, normalize=True, trim_silence=True, target_db=-20.0, top_db=30):
    """
    Applies cleaning functions (normalize, trim silence) to all audio files in a directory.
    Ensures output is mono and at TARGET_SR (16kHz).
    """
    ensure_dir_exists(output_dir)
    logger.info(f"Starting audio cleaning process for directory: {input_dir}")
    processed_files = 0
    failed_files = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')): # Add more formats if needed
            input_path = os.path.join(input_dir, filename)
            # Use a temporary path for intermediate steps if chaining multiple operations
            temp_output_path = os.path.join(output_dir, f"temp_{filename}")
            final_output_path = os.path.join(output_dir, filename) # Keep original name in the end

            current_input = input_path
            processing_ok = True

            if normalize:
                if not normalize_volume(current_input, temp_output_path, target_db=target_db):
                    logger.warning(f"Skipping normalization for {filename}")
                    processing_ok = False
                else:
                    current_input = temp_output_path # Output of normalize becomes input for next step

            if processing_ok and trim_silence:
                 # If normalization was skipped, current_input is still input_path
                 # If normalization succeeded, current_input is temp_output_path
                if not remove_silence(current_input, final_output_path, top_db=top_db):
                    logger.warning(f"Skipping silence removal for {filename}")
                    # If silence removal fails, decide what to do. Copy the input to final?
                    if current_input != final_output_path: # Avoid copying if paths are same
                        try:
                            shutil.copy(current_input, final_output_path)
                            logger.debug(f"Copied '{current_input}' to '{final_output_path}' after silence removal failure.")
                        except Exception as copy_e:
                             logger.error(f"Failed to copy '{current_input}' to '{final_output_path}': {copy_e}")
                             processing_ok = False # Mark as failed if copy fails too
                # If silence removal succeeded, the result is already in final_output_path

            elif processing_ok and not trim_silence:
                 # If only normalization was done, move the temp file to final name
                 if current_input == temp_output_path: # Check if normalization actually ran
                     try:
                         os.rename(temp_output_path, final_output_path)
                         logger.debug(f"Renamed '{temp_output_path}' to '{final_output_path}' (only normalization applied).")
                     except Exception as rename_e:
                         logger.error(f"Failed to rename temp file '{temp_output_path}' to '{final_output_path}': {rename_e}")
                         processing_ok = False
                 elif current_input == input_path: # No processing was done
                     try:
                         # Ensure the output is in the target format/sr even if no processing done
                         y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
                         sf.write(final_output_path, y, TARGET_SR)
                         logger.debug(f"Converted/copied '{input_path}' to '{final_output_path}' (no processing requested).")
                     except Exception as load_save_e:
                         logger.error(f"Failed to load/save '{input_path}' to '{final_output_path}': {load_save_e}")
                         processing_ok = False


            # Clean up temporary file if it exists and wasn't renamed
            if os.path.exists(temp_output_path) and temp_output_path != final_output_path:
                try:
                    os.remove(temp_output_path)
                except OSError as e:
                    logger.warning(f"Could not remove temporary file {temp_output_path}: {e}")

            if processing_ok:
                logger.info(f"Successfully processed {filename}")
                processed_files += 1
            else:
                logger.error(f"Failed to process {filename}")
                failed_files += 1
                # Optionally remove the potentially corrupted final_output_path if processing failed
                if os.path.exists(final_output_path):
                    try:
                        # Be cautious removing files, maybe move to a 'failed' directory instead
                        # os.remove(final_output_path)
                        pass
                    except OSError as e:
                        logger.warning(f"Could not remove failed output file {final_output_path}: {e}")

        else:
            logger.debug(f"Skipping non-audio file: {filename}")

    logger.info(f"Audio cleaning finished. Processed: {processed_files}, Failed: {failed_files}")

# Example usage
if __name__ == "__main__":
    import sys
    # Ensure utils are importable
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.logging_utils import setup_logger # Re-import might be needed if run directly
    from utils.file_utils import ensure_dir_exists

    # Example directories (adjust as needed)
    input_audio_dir = "../data/tts_test_output" # Use output from TTS test
    output_cleaned_audio_dir = "../data/dataset/audio_cleaned"

    # Create dummy input if it doesn't exist
    ensure_dir_exists(input_audio_dir)
    if not os.listdir(input_audio_dir):
         logger.warning(f"Input directory {input_audio_dir} is empty. Creating a dummy silent file.")
         try:
             # Create a short silent wav file for testing
             sr = 16000; duration = 1; silence = np.zeros(int(sr*duration))
             sf.write(os.path.join(input_audio_dir, "dummy_silent.wav"), silence, sr)
         except Exception as e:
             logger.error(f"Failed to create dummy silent file: {e}")


    logger.info("Running audio cleaning process...")
    process_audio_directory(input_audio_dir, output_cleaned_audio_dir)
    logger.info("Audio cleaning process finished.")
