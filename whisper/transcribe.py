import os
import sys
import argparse
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, AutomaticSpeechRecognitionPipeline
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


def run_transcription(audio_path: str, model_path: str, output_txt_dir: str, output_srt_dir: str, device: str = "cpu"):
    """
    Transcribes an audio file using the specified Whisper model and saves the result
    into separate directories for TXT and SRT files.
    """
    resolved_model_path = os.path.abspath(model_path) # Resolve to absolute path
    logger.info(f"Starting transcription for: {audio_path}")
    logger.info(f"Attempting to load model from: {resolved_model_path}") # Log resolved path
    logger.info(f"Using device: {device}")

    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None

    try:
        # Determine if the model path is a local fine-tuned model or from Hugging Face Hub
        if not os.path.isdir(resolved_model_path):
            is_likely_hub_id = '/' not in resolved_model_path and '\\' not in resolved_model_path and resolved_model_path
            if is_likely_hub_id:
                logger.info(f"'{resolved_model_path}' is not a local directory. Assuming it's a Hugging Face Hub ID.")
                # Proceed to load from Hub using pipeline factory
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model_path, # Use original model_path which is the Hub ID
                    device=0 if device == "cuda" else -1,
                    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
                )
                logger.info("Hub model loaded via pipeline factory.")
            else:
                logger.error(f"Model path is neither a valid directory nor seems like a Hub ID: {resolved_model_path}")
                return None
        else:
            # It is a local directory, proceed with local loading
            logger.info("Loading local fine-tuned model components...")

            # 직접 model과 processor 로드
            device_map = "auto" if device == "cuda" else None
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            # 모델 로드
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                resolved_model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                local_files_only=True
            )

            # 프로세서 로드
            processor = AutoProcessor.from_pretrained(
                resolved_model_path,
                local_files_only=True
            )

            # --- Generation Config 수정 ---
            logger.info("Adjusting generation config settings...")
            if hasattr(model, "generation_config"):
                # 1. forced_decoder_ids 및 suppress_tokens 제거 (이전 단계에서 추가됨)
                model.generation_config.forced_decoder_ids = None
                model.generation_config.suppress_tokens = None
                logger.info("Cleared forced_decoder_ids and suppress_tokens.")

                # 2. pad_token_id 설정 (Attention Mask 경고 해결)
                if hasattr(model.config, 'eos_token_id'):
                     model.generation_config.pad_token_id = model.config.eos_token_id
                     logger.info(f"Set generation_config.pad_token_id to model.config.eos_token_id ({model.config.eos_token_id}).")
                else:
                     logger.warning("Model config does not have 'eos_token_id'. Cannot set pad_token_id automatically.")

            else:
                logger.warning("Model does not have 'generation_config' attribute. Skipping config adjustments.")
            # --- End of config 수정 ---

            # 파이프라인 수동 생성
            pipeline_kwargs = {
                "model": model,
                "tokenizer": processor.tokenizer,
                "feature_extractor": processor.feature_extractor,
                "max_new_tokens": 128,
                "torch_dtype": torch_dtype,
            }
            if device_map is None: # Only add device if not using device_map (i.e., CPU)
                pipeline_kwargs["device"] = -1 # device index for CPU

            pipe = AutomaticSpeechRecognitionPipeline(**pipeline_kwargs)
            logger.info("Local model components loaded and pipeline created.")

        logger.info("Starting transcription process...")
        # 트랜스크립션 수행
        # Define generate_kwargs for language if needed, otherwise pipeline might detect
        # generate_kwargs = {"language": "korean"} # Example
        outputs = pipe(
            audio_path,
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5
            # generate_kwargs=generate_kwargs # Pass if defined
        )
        transcription_chunks = outputs["chunks"]
        full_text = outputs["text"]

        logger.info("Transcription finished.")
        # logger.debug(f"Full Text: {full_text}")
        # logger.debug(f"Chunks: {transcription_chunks}")

        # Save results
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        ensure_dir_exists(output_txt_dir) # Ensure TXT directory exists
        ensure_dir_exists(output_srt_dir) # Ensure SRT directory exists

        # Save full text to output_txt_dir
        txt_output_path = os.path.join(output_txt_dir, f"{base_filename}.txt")
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info(f"Full transcription saved to: {txt_output_path}")

        # Save SRT to output_srt_dir
        srt_output_path = os.path.join(output_srt_dir, f"{base_filename}.srt")
        save_as_srt(transcription_chunks, srt_output_path)

        return {"text_path": txt_output_path, "srt_path": srt_output_path, "full_text": full_text}

    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file to transcribe.")
    # Default to the fine-tuned model directory (absolute path within container/project)
    default_model_path = os.path.abspath("./whisper-finetuned-model")
    parser.add_argument("--model_path", type=str, default=default_model_path, help=f"Path to the fine-tuned model directory or Hugging Face model name. Default: {default_model_path}")
    # Separate output directories
    parser.add_argument("--output_txt_dir", type=str, default="data/transcripts", help="Directory to save the full text transcription (.txt).")
    parser.add_argument("--output_srt_dir", type=str, default="data/subtitles", help="Directory to save the timestamped subtitle file (.srt).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference (cuda or cpu).")
    args = parser.parse_args()

    run_transcription(args.audio_path, args.model_path, args.output_txt_dir, args.output_srt_dir, args.device)
