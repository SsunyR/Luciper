import os
import sys
import argparse
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from datetime import timedelta
import warnings

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_utils import setup_logger
from utils.file_utils import ensure_dir_exists

logger = setup_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.pipelines.base')


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    """Formats seconds to SRT timestamp format."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

def save_as_srt(transcription_result: list, output_srt_path: str):
    """Saves the transcription result with timestamps as an SRT file."""
    ensure_dir_exists(os.path.dirname(output_srt_path))
    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        for i, chunk in enumerate(transcription_result):
            start_time = chunk["timestamp"][0]
            end_time = chunk["timestamp"][1]
            text = chunk["text"].strip()

            # Skip empty chunks or very short timestamps if needed
            if not text or end_time is None or start_time is None: #or (end_time - start_time < 0.1):
                continue

            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
            srt_file.write(f"{text}\n\n")
    logger.info(f"Transcription saved as SRT file: {output_srt_path}")


def run_transcription(audio_path: str, model_path: str, output_dir: str, device: str = "cpu"):
    """
    Transcribes an audio file using the specified Whisper model and saves the result.
    """
    logger.info(f"Starting transcription for: {audio_path}")
    logger.info(f"Using model: {model_path}")
    logger.info(f"Using device: {device}")

    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None

    try:
        # Determine if the model path is a local fine-tuned model or from Hugging Face Hub
        if os.path.isdir(model_path):
            logger.info("Loading local fine-tuned model...")
            # Load model and processor explicitly for local fine-tuned models
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, device_map=device) # Use device_map
            processor = AutoProcessor.from_pretrained(model_path)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=0 if device == "cuda" else -1, # pipeline expects device index
                # Consider chunk_length_s and stride_length_s for long files
                # chunk_length_s=30,
                # stride_length_s=[6, 0] # Example stride
            )
        else:
            logger.info("Loading model from Hugging Face Hub...")
            # Use pipeline directly for models from the Hub
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                device=0 if device == "cuda" else -1,
                # chunk_length_s=30,
            )

        logger.info("Model loaded. Starting transcription process...")
        # Perform transcription with timestamps
        # Set return_timestamps=True and potentially generate_kwargs for language
        # generate_kwargs = {"language": "korean"} # Adjust language if needed based on model/config
        outputs = pipe(audio_path, return_timestamps=True)#, generate_kwargs=generate_kwargs)
        transcription_chunks = outputs["chunks"]
        full_text = outputs["text"]

        logger.info("Transcription finished.")
        # logger.debug(f"Full Text: {full_text}")
        # logger.debug(f"Chunks: {transcription_chunks}")

        # Save results
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        ensure_dir_exists(output_dir)

        # Save full text
        txt_output_path = os.path.join(output_dir, f"{base_filename}.txt")
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info(f"Full transcription saved to: {txt_output_path}")

        # Save SRT
        srt_output_path = os.path.join(output_dir, f"{base_filename}.srt")
        save_as_srt(transcription_chunks, srt_output_path)

        return {"text_path": txt_output_path, "srt_path": srt_output_path, "full_text": full_text}

    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file to transcribe.")
    # Default to the fine-tuned model directory specified in train.py config
    parser.add_argument("--model_path", type=str, default="./whisper-finetuned-model", help="Path to the fine-tuned model directory or Hugging Face model name.")
    parser.add_argument("--output_dir", type=str, default="data/subtitles", help="Directory to save the transcription results (.txt, .srt).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference (cuda or cpu).")
    args = parser.parse_args()

    run_transcription(args.audio_path, args.model_path, args.output_dir, args.device)
