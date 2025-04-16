import argparse
import os
from gtts import gTTS
from utils.file_utils import read_yaml, ensure_dir_exists
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def generate_with_gtts(text, lang, output_path):
    """Generates speech using Google Text-to-Speech (gTTS)."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        logger.info(f"Successfully generated TTS audio at {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate TTS for text: '{text[:50]}...' - Error: {e}")
        return False

def process_text_file(input_txt_path, output_dir, config):
    """Processes a text file line by line to generate TTS audio."""
    ensure_dir_exists(output_dir)
    lang = config.get('language', 'en') # Default to English if not specified
    output_format = config.get('output_format', 'mp3')

    try:
        with open(input_txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue

                # Create a unique filename for each line
                output_filename = f"line_{i+1:04d}.{output_format}"
                output_path = os.path.join(output_dir, output_filename)

                # Generate TTS based on the selected engine
                if config.get('engine', 'gtts') == 'gtts':
                    if output_format != 'mp3':
                        logger.warning("gTTS primarily outputs MP3. Saving as MP3.")
                        output_filename = f"line_{i+1:04d}.mp3"
                        output_path = os.path.join(output_dir, output_filename)

                    if not generate_with_gtts(text, lang, output_path):
                        logger.warning(f"Skipping line {i+1} due to generation error.")
                else:
                    logger.error(f"Unsupported TTS engine: {config.get('engine')}")
                    # Add logic for other TTS engines here if needed
                    break # Stop processing if engine is unsupported

    except FileNotFoundError:
        logger.error(f"Input text file not found: {input_txt_path}")
    except Exception as e:
        logger.error(f"An error occurred while processing {input_txt_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio from text file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the TTS config YAML file.")
    parser.add_argument("--input_text", type=str, required=True, help="Path to the input text file (one sentence per line).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated audio files.")
    args = parser.parse_args()

    logger.info("Starting TTS generation process...")
    config = read_yaml(args.config)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    logger.info(f"Loaded configuration: {config}")
    logger.info(f"Input text file: {args.input_text}")
    logger.info(f"Output directory: {args.output_dir}")

    process_text_file(args.input_text, args.output_dir, config)

    logger.info("TTS generation process finished.")

if __name__ == "__main__":
    # Add necessary imports to utils if you haven't already
    # Make sure utils are accessible (e.g., by setting PYTHONPATH or running from the project root)
    import sys
    # Assuming utils directory is one level up from tts directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.file_utils import read_yaml, ensure_dir_exists
    from utils.logging_utils import setup_logger

    main()
