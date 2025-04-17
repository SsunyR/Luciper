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

def save_text_file(text, output_path):
    """Saves the given text to the specified path."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.debug(f"Saved text to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save text file {output_path}: {e}")
        return False

def process_text_file(input_txt_path, output_audio_dir, output_text_dir, config):
    """Processes a text file line by line to generate TTS audio and save text."""
    ensure_dir_exists(output_audio_dir)
    ensure_dir_exists(output_text_dir) # Ensure text output directory exists
    lang = config.get('language', 'en') # Default to English if not specified
    output_format = config.get('output_format', 'mp3')
    engine = config.get('engine', 'gtts')

    try:
        with open(input_txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue

                # Determine base filename (without extension)
                base_filename = f"line_{i+1:04d}"

                # Determine audio output path
                audio_output_filename = f"{base_filename}.{output_format}"
                if engine == 'gtts' and output_format != 'mp3':
                    logger.warning("gTTS primarily outputs MP3. Saving as MP3.")
                    audio_output_filename = f"{base_filename}.mp3"
                audio_output_path = os.path.join(output_audio_dir, audio_output_filename)

                # Determine text output path
                text_output_filename = f"{base_filename}.txt"
                text_output_path = os.path.join(output_text_dir, text_output_filename)

                # Generate TTS
                success = False
                if engine == 'gtts':
                    success = generate_with_gtts(text, lang, audio_output_path)
                else:
                    logger.error(f"Unsupported TTS engine: {engine}")
                    break # Stop if engine is unsupported

                # Save corresponding text file if TTS was successful
                if success:
                    if not save_text_file(text, text_output_path):
                        logger.warning(f"Failed to save text file for line {i+1}. Audio was generated.")
                else:
                    logger.warning(f"Skipping line {i+1} due to TTS generation error.")

    except FileNotFoundError:
        logger.error(f"Input text file not found: {input_txt_path}")
    except Exception as e:
        logger.error(f"An error occurred while processing {input_txt_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio and corresponding text files.")
    parser.add_argument("--config", type=str, default="tts/tts_config.yaml", help="Path to the TTS config YAML file. Default: tts/tts_config.yaml")
    parser.add_argument("--input_text", type=str, default="data/raw/input.txt", help="Path to the input text file (one sentence per line). Default: data/raw/input.txt")
    parser.add_argument("--output_audio_dir", type=str, default="data/tts", help="Directory to save the generated audio files. Default: data/tts")
    parser.add_argument("--output_text_dir", type=str, default="data/tts_text", help="Directory to save the corresponding text files (.txt). Default: data/tts_text")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found at default path: {args.config}. Please specify with --config.")
        return
    if not os.path.exists(args.input_text):
        logger.error(f"Input text file not found at default path: {args.input_text}. Please specify with --input_text or create the file.")
        return

    logger.info("Starting TTS generation process...")
    config = read_yaml(args.config)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    logger.info(f"Loaded configuration: {config}")
    logger.info(f"Input text file: {args.input_text}")
    logger.info(f"Output audio directory: {args.output_audio_dir}")
    logger.info(f"Output text directory: {args.output_text_dir}") # Log the text output dir

    process_text_file(args.input_text, args.output_audio_dir, args.output_text_dir, config) # Pass text dir

    logger.info("TTS generation process finished.")

if __name__ == "__main__":
    import sys
    # Assuming utils directory is one level up from tts directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.file_utils import read_yaml, ensure_dir_exists
    from utils.logging_utils import setup_logger

    main()
