import os
import pandas as pd
import argparse
from utils.logging_utils import setup_logger
from utils.file_utils import ensure_dir_exists

logger = setup_logger(__name__)

def create_metadata_file(audio_dir, text_dir, output_metadata_path):
    """
    Creates a metadata file (e.g., CSV) mapping audio files to their transcriptions.
    Assumes audio and text files have corresponding names (e.g., line_0001.mp3 and line_0001.txt).
    Adjust logic based on your actual data structure.
    """
    logger.info(f"Creating metadata file for audio in '{audio_dir}' and text in '{text_dir}'")
    ensure_dir_exists(os.path.dirname(output_metadata_path))
    metadata = []

    try:
        for audio_filename in os.listdir(audio_dir):
            if not audio_filename.lower().endswith(('.wav', '.mp3', '.flac')):
                continue

            base_name = os.path.splitext(audio_filename)[0]
            text_filename = f"{base_name}.txt" # Assumes corresponding .txt file exists
            text_path = os.path.join(text_dir, text_filename)
            audio_path = os.path.join(audio_dir, audio_filename) # Store relative or absolute path as needed

            if os.path.exists(text_path):
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()
                    if transcription:
                        # Get absolute path for audio file for easier loading later
                        absolute_audio_path = os.path.abspath(audio_path)
                        metadata.append({'audio_filepath': absolute_audio_path, 'text': transcription})
                    else:
                        logger.warning(f"Skipping empty transcription file: {text_path}")
                except Exception as e:
                    logger.error(f"Error reading transcription file {text_path}: {e}")
            else:
                logger.warning(f"Transcription file not found for audio: {audio_filename}. Expected at: {text_path}")

        if not metadata:
            logger.warning("No matching audio-text pairs found. Metadata file will be empty.")
            # Create an empty DataFrame with columns to avoid errors downstream
            df = pd.DataFrame(columns=['audio_filepath', 'text'])
        else:
            df = pd.DataFrame(metadata)

        # Save as CSV (common format for Hugging Face datasets)
        df.to_csv(output_metadata_path, index=False, encoding='utf-8')
        logger.info(f"Metadata file created successfully at: {output_metadata_path} with {len(df)} entries.")
        return True

    except FileNotFoundError:
        logger.error(f"Audio directory not found: {audio_dir}")
        return False
    except Exception as e:
        logger.error(f"Failed to create metadata file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Format dataset for Whisper fine-tuning.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing the audio files (e.g., cleaned audio).")
    parser.add_argument("--text_dir", type=str, required=True, help="Directory containing the corresponding transcription files (.txt).")
    parser.add_argument("--output_metadata", type=str, required=True, help="Path to save the output metadata file (e.g., data/dataset/metadata.csv).")
    args = parser.parse_args()

    logger.info("Starting dataset formatting...")
    create_metadata_file(args.audio_dir, args.text_dir, args.output_metadata)
    logger.info("Dataset formatting finished.")

if __name__ == "__main__":
    # Add necessary imports to utils if you haven't already
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.logging_utils import setup_logger
    from utils.file_utils import ensure_dir_exists
    # Need pandas for this script
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas library is required. Please install it: pip install pandas")
        sys.exit(1)

    main()
