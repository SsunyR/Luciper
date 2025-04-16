import os
import sys
import argparse
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate # HF evaluate library
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_utils import setup_logger
from utils.file_utils import read_yaml

logger = setup_logger(__name__)

# --- Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# --- Main Training Function ---
def run_training(config_path: str):
    logger.info("Starting Whisper fine-tuning process...")

    # 1. Load Configuration
    config = read_yaml(config_path)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return
    logger.info(f"Loaded configuration: {config}")

    # Extract relevant config values
    model_name = config['model_name_or_path']
    language = config['language']
    task = config['task']
    metadata_file = config['metadata_file_path']
    audio_col = config['audio_column_name']
    text_col = config['text_column_name']
    output_dir = config['output_dir']
    max_duration_s = config.get('max_duration_in_seconds', 30)
    min_duration_s = config.get('min_duration_in_seconds', 1) # Added min duration
    max_label_len = config.get('max_label_length', 448)

    # 2. Load Processor (Feature Extractor and Tokenizer)
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
        processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    except Exception as e:
        logger.error(f"Failed to load processor components for {model_name}: {e}")
        return

    # 3. Load and Prepare Dataset
    logger.info(f"Loading dataset from metadata file: {metadata_file}")
    try:
        # Load dataset from the CSV/JSON metadata file
        # Ensure the audio column is interpreted correctly
        dataset = load_dataset("csv", data_files=metadata_file, split=config.get('train_split_name', 'train')) # Adjust split if needed

        # --- Basic dataset validation ---
        if audio_col not in dataset.column_names or text_col not in dataset.column_names:
             logger.error(f"Metadata file must contain columns named '{audio_col}' and '{text_col}'. Found: {dataset.column_names}")
             return
        if len(dataset) == 0:
            logger.error("Loaded dataset is empty. Check metadata file and paths.")
            return
        logger.info(f"Original dataset size: {len(dataset)}")
        logger.info(f"Dataset features: {dataset.features}")

        # Cast the audio column to Audio feature type
        logger.info("Casting audio column to Audio feature...")
        dataset = dataset.cast_column(audio_col, Audio(sampling_rate=16000))

        # --- Preprocessing function ---
        def prepare_dataset(batch):
            # compute log-Mel input features from input audio array
            audio = batch[audio_col]
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

            # encode target text to label ids
            batch["labels"] = tokenizer(batch[text_col]).input_ids
            return batch

        # --- Apply preprocessing ---
        logger.info("Applying preprocessing to the dataset...")
        dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1) # Adjust num_proc based on CPU cores

        # --- Filtering ---
        min_input_length = min_duration_s * feature_extractor.sampling_rate
        max_input_length = max_duration_s * feature_extractor.sampling_rate

        def is_audio_in_length_range(length):
            return length >= min_input_length and length <= max_input_length

        def is_label_in_length_range(labels):
             return len(labels) > 0 and len(labels) < max_label_len # Ensure label is not empty and within max length

        # Need input_features length, which isn't directly available post-map without loading.
        # A common approach is to filter based on audio file duration *before* loading/preprocessing,
        # or filter based on label length after preprocessing. Let's filter by label length here.
        # Filtering by audio duration before `load_dataset` or just after is more efficient.
        # Example pre-filtering (requires reading audio duration beforehand):
        # metadata_df = pd.read_csv(metadata_file)
        # metadata_df['duration'] = metadata_df[audio_col].apply(get_audio_duration) # Implement get_audio_duration
        # filtered_df = metadata_df[(metadata_df['duration'] >= min_duration_s) & (metadata_df['duration'] <= max_duration_s)]
        # filtered_df.to_csv("filtered_metadata.csv", index=False)
        # dataset = load_dataset("csv", data_files="filtered_metadata.csv", ...)

        logger.info(f"Filtering dataset by label length (0 < len < {max_label_len})...")
        initial_count = len(dataset)
        dataset = dataset.filter(is_label_in_length_range, input_columns=["labels"])
        filtered_count = len(dataset)
        logger.info(f"Filtered out {initial_count - filtered_count} samples based on label length.")

        if filtered_count == 0:
            logger.error("Dataset is empty after filtering. Check data or filtering criteria.")
            return

        # Optional: Split into train/validation if needed
        # if "validation_split_name" in config:
        #     logger.info("Splitting dataset into train and validation...")
        #     dataset = dataset.train_test_split(test_size=0.1) # Example split
        # else:
        #     # If no validation split, create a dummy DatasetDict
        #     dataset = DatasetDict({"train": dataset})


    except Exception as e:
        logger.error(f"Failed to load or process dataset: {e}", exc_info=True)
        return

    # 4. Load Pre-trained Model
    logger.info(f"Loading pre-trained model: {model_name}")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        # Configure model for language and task if needed (often handled by processor)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        model.config.suppress_tokens = []
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return

    # 5. Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get('num_train_epochs', 3),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        learning_rate=float(config.get('learning_rate', 1e-5)), # Ensure float
        warmup_steps=config.get('warmup_steps', 50),
        # evaluation_strategy=config.get('evaluation_strategy', "no"), # Default to no eval if not specified
        # eval_steps=config.get('eval_steps', 500),
        # save_strategy=config.get('save_strategy', "steps"),
        # save_steps=config.get('save_steps', 500),
        # logging_steps=config.get('logging_steps', 100),
        logging_dir=f"{output_dir}/logs",
        fp16=config.get('fp16', torch.cuda.is_available()), # Enable if available
        # gradient_checkpointing=config.get('gradient_checkpointing', False),
        # report_to=config.get('report_to', "tensorboard"),
        load_best_model_at_end=False, # Set to True if using evaluation_strategy
        # metric_for_best_model="wer", # Set if load_best_model_at_end=True
        # greater_is_better=False, # For WER
        push_to_hub=False, # Set to True to push to Hugging Face Hub
        remove_unused_columns=False, # Important for custom datasets
        label_names=["labels"], # Ensure labels are identified correctly
        # Add any other necessary arguments from config
    )

    # 6. Initialize Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 7. Define Compute Metrics (WER)
    logger.info("Setting up evaluation metric (WER)...")
    try:
        wer_metric = evaluate.load("wer")
    except Exception as e:
        logger.error(f"Failed to load WER metric: {e}. Evaluation will not compute WER.")
        wer_metric = None

    def compute_metrics(pred):
        if wer_metric is None:
            return {}
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 8. Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset, # Use the processed dataset directly if no split
        # eval_dataset=dataset["validation"] if "validation" in dataset else None, # Use validation split if available
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None, # Only compute metrics if generating predictions
        tokenizer=processor.feature_extractor, # Pass feature extractor for generation config
    )

    # 9. Start Training
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        logger.info("Training finished.")

        # Save model and training stats
        trainer.save_model() # Saves the tokenizer too
        logger.info(f"Model saved to {output_dir}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Training metrics saved.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

    # 10. Optional: Evaluation on a test set
    # if "test" in dataset:
    #     logger.info("Starting evaluation on the test set...")
    #     try:
    #         eval_metrics = trainer.evaluate(eval_dataset=dataset["test"])
    #         logger.info(f"Test set evaluation metrics: {eval_metrics}")
    #         trainer.log_metrics("eval", eval_metrics)
    #         trainer.save_metrics("eval", eval_metrics)
    #     except Exception as e:
    #         logger.error(f"An error occurred during evaluation: {e}", exc_info=True)

    logger.info("Whisper fine-tuning process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model.")
    parser.add_argument("--config", type=str, default="whisper/config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Ensure CUDA is available if fp16 is requested
    config_check = read_yaml(args.config)
    if config_check and config_check.get('fp16', False) and not torch.cuda.is_available():
        logger.warning("FP16 requested in config, but CUDA is not available. Disabling FP16.")
        # You might want to update the config dict here or handle it in TrainingArguments
        # For simplicity, Seq2SeqTrainingArguments handles fp16=True gracefully if cuda is not available

    run_training(args.config)
