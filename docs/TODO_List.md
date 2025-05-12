# Luciper Project TODO List (Agent Friendly)

## Phase 1: Environment Setup & Basic Function Implementation

*   [x] **Task 1.1: Setup Project Structure**
    *   Create directories: `src`, `tests`, `notebooks`, `data`, `models`, `config`.
    *   Initialize `git` repository.
*   [x] **Task 1.2: Setup Virtual Environment & Core Dependencies**
    *   Initialize and activate a virtual environment using `uv`.
    *   Command: `uv venv`
    *   Command: `uv pip install python transformers torch torchaudio jupyterlab ipywidgets pandas` (or `openai-whisper` if chosen over `transformers` initially for basic tests)
    *   Command: `uv pip install --system uv` (if `uv` is to be managed within the project's dependencies, though typically it's a global tool)
*   [ ] **Task 1.3: Basic Whisper Model Test**
    *   In a `Jupyter Notebook` (`notebooks/01_basic_whisper_test.ipynb`):
        *   Load a pre-trained Whisper base model (e.g., `tiny` or `base`) using `transformers`.
        *   Perform inference on a sample audio file to generate subtitles.
        *   Document the process and results.
*   [ ] **Task 1.4: Implement Basic Inference Module (`src/inference_module.py`)**
    *   Create a function `generate_subtitles(audio_path: str, model_name_or_path: str) -> str`.
    *   This function should load the specified Whisper model and generate text from the given audio file.
    *   Test this module using a `Jupyter Notebook` (`notebooks/02_test_inference_module.ipynb`) with various simple audio files.
*   [ ] **Task 1.5: Initial `.gitignore`**
    *   Create a `.gitignore` file.
    *   Add common Python ignores (`__pycache__/`, `*.pyc`, `*.egg-info/`, `dist/`, `build/`, `venv/`, `.env`).
    *   Add project-specific ignores (`data/raw_audio/` (if large), `models/` (if not versioning large models directly), `*.ipynb_checkpoints`).

## Phase 2: Fine-tuning Pipeline Construction

*   [ ] **Task 2.1: Build Basic Fine-tuning Pipeline (`notebooks/03_basic_finetuning_pipeline.ipynb`)**
    *   Using Hugging Face `transformers` (`Trainer`, `TrainingArguments`) and `datasets`.
    *   Use a small, readily available sample dataset for speech-to-text (e.g., a subset of LibriSpeech or a dummy dataset).
    *   Implement data loading and preprocessing steps suitable for Whisper.
    *   Run a basic fine-tuning loop on a small Whisper model (e.g., `tiny`).
*   [ ] **Task 2.2: Implement Fine-tuning Module (`src/finetuning_module.py`)**
    *   Create a function `fine_tune_model(base_model_name_or_path: str, dataset_path: str, output_dir: str, training_args: dict) -> str` (returns path to fine-tuned model).
    *   Encapsulate the fine-tuning logic from the notebook.
    *   Parameterize hyperparameters and model/data paths.
    *   Test this module using a `Jupyter Notebook` (`notebooks/04_test_finetuning_module.ipynb`).
*   [ ] **Task 2.3: GPU Environment Configuration and Test**
    *   Ensure PyTorch is correctly configured to use available GPUs.
    *   Command: `python -c "import torch; print(torch.cuda.is_available())"`
    *   Test the fine-tuning module on a GPU and monitor resource usage.
    *   Document any specific driver or CUDA version requirements.

## Phase 3: Data Preparation Function Implementation

*   [ ] **Task 3.1: Implement TTS Data Generation (`src/data_preparation_module.py`)**
    *   Function: `generate_tts_audio(text_input: str, tts_engine: str, voice: str, output_audio_path: str) -> str`.
    *   Initial integration: `gTTS` for simplicity.
    *   Later integration: `Piper TTS`.
*   [ ] **Task 3.2: Integrate Piper TTS**
    *   Install `piper-tts`.
    *   Download a `Piper TTS` voice model.
    *   Test `Piper TTS` generation in a `Jupyter Notebook` (`notebooks/05_test_piper_tts.ipynb`).
    *   Update `generate_tts_audio` to support `Piper TTS`.
*   [ ] **Task 3.3: Create Fine-tuning Dataset from TTS (`src/data_preparation_module.py`)**
    *   Function: `create_dataset_from_tts(texts: list[str], tts_engine: str, voice: str, dataset_output_dir: str) -> str` (returns path to dataset).
    *   This function will use `generate_tts_audio` to create audio files and then structure them with the corresponding texts into a Hugging Face `Dataset` object (e.g., with "audio" and "text" columns).
    *   The dataset should be saved in a format loadable by the `Fine-tuning Module`.
    *   Test in `Jupyter Notebook` (`notebooks/06_test_tts_dataset_creation.ipynb`).
*   [ ] **Task 3.4: Implement Corrected Subtitle Processing (`src/data_preparation_module.py`)**
    *   Function: `create_dataset_from_corrected_subtitles(audio_files: list[str], corrected_texts: list[str], dataset_output_dir: str) -> str`.
    *   This function takes audio files and their corresponding user-corrected transcriptions and structures them into a Hugging Face `Dataset`.
    *   Test in `Jupyter Notebook`.

## Phase 4: Model Management Function Implementation

*   [ ] **Task 4.1: Implement Model Saving/Loading (`src/model_management_module.py`)**
    *   Fine-tuned models from `transformers` are typically saved as directories.
    *   Function: `save_model(model_object, path: str, metadata: dict)`. (The `transformers` Trainer already saves models, this might be more about managing paths and metadata).
    *   Function: `load_model(path: str) -> AutoModelForSpeechSeq2Seq`.
    *   Function: `save_metadata(model_name: str, metadata: dict, metadata_file_path: str)`.
    *   Function: `load_metadata(model_name: str, metadata_file_path: str) -> dict`.
    *   Function: `update_metadata(model_name: str, updates: dict, metadata_file_path: str)`.
    *   Function: `list_fine_tuned_models(models_base_dir: str, metadata_file_path: str) -> list[dict]`.
    *   Store metadata in `JSON` files initially (e.g., `models_metadata.json`). Each model entry should have `name`, `alias`, `domain_description`, `base_model`, `path_to_model_files`, `creation_date`, `last_finetuned_date`.
    *   Test module functions in `Jupyter Notebook` (`notebooks/07_test_model_management.ipynb`).

## Phase 5: Basic Gradio Web Interface Construction

*   [ ] **Task 5.1: Install Gradio**
    *   Command: `uv pip install gradio`
*   [ ] **Task 5.2: Create Basic Gradio App Structure (`main.py` or `app.py`)**
    *   Import necessary modules.
    *   Set up a basic `gr.Blocks()` interface.
*   [ ] **Task 5.3: Implement Model Selection UI (Gradio)**
    *   Dropdown to select a base Whisper model (e.g., `tiny`, `base`, `small`).
    *   Dropdown/List to display and select previously saved fine-tuned models (using `list_fine_tuned_models` from `Model Management Module`).
    *   Interface to create a new fine-tuned model (input alias, domain description).
*   [ ] **Task 5.4: Implement Subtitle Generation UI (Gradio)**
    *   File upload component for audio/video.
    *   Button "Generate Subtitles".
    *   Connect this to the `Inference Module` (`generate_subtitles`).
    *   Display generated subtitles in a textbox.
*   [ ] **Task 5.5: Implement TTS-based Training UI (Gradio)**
    *   Textbox for pasting large text or file upload for text file.
    *   Option to select TTS engine (if multiple are supported) and voice.
    *   Button "Start Fine-tuning with TTS Data".
    *   Connect this to `Data Preparation Module` (for TTS dataset creation) and `Fine-tuning Module`.
    *   **Crucial:** Implement this as a background/asynchronous task in Gradio to avoid UI freezing. Display logs or progress.

## Phase 6: Complete Subtitle Editing and Continuous Learning Pipeline

*   [ ] **Task 6.1: Add Subtitle Editing UI (Gradio)**
    *   Editable textbox to modify the generated subtitles.
*   [ ] **Task 6.2: Implement Evaluation Module (`src/evaluation_module.py`)**
    *   Function: `calculate_wer(reference: str, hypothesis: str) -> float`.
    *   Install `jiwer` or similar library: `uv pip install jiwer`.
    *   Test in `Jupyter Notebook` (`notebooks/08_test_evaluation_module.ipynb`).
*   [ ] **Task 6.3: Integrate Evaluation UI (Gradio)**
    *   Button "Check Accuracy" that takes original (generated) and edited subtitles.
    *   Display WER and potentially other metrics.
    *   Interface to show a diff between generated and corrected subtitles.
*   [ ] **Task 6.4: Implement "Train with Corrections" UI (Gradio)**
    *   Button "Train with Corrections".
    *   This button should trigger a pipeline:
        1.  Take the uploaded audio and the *edited* subtitles.
        2.  Use `Data Preparation Module` (`create_dataset_from_corrected_subtitles`) to create/append to a training dataset for the selected fine-tuned model.
        3.  Re-run the `Fine-tuning Module` using the selected model as a base and the new/updated dataset.
        4.  Handle this as a background/asynchronous task.

## Phase 7: Testing, Refactoring, and Documentation

*   [ ] **Task 7.1: Integration Testing**
    *   Test the full workflow via the Gradio interface:
        *   Create new model -> TTS train -> Generate subs -> Edit subs -> Check accuracy -> Train with corrections.
    *   Test with various audio types (clean, noisy) and text data.
*   [ ] **Task 7.2: Code Refactoring**
    *   Review all modules for clarity, efficiency, and adherence to Python best practices (type hints, docstrings).
    *   Ensure modularity and well-defined inputs/outputs.
    *   Incorporate feedback from pair programming.
*   [ ] **Task 7.3: Write/Update `README.md`**
    *   Project introduction.
    *   Installation guide (including `uv` usage, Python version, GPU requirements).
    *   Usage instructions for the Gradio app.
    *   Contribution guide (if applicable).
    *   License information (MIT).
*   [ ] **Task 7.4: Improve Code Documentation**
    *   Ensure all functions, classes, and modules have comprehensive Google-style docstrings.
    *   Add inline comments where necessary for complex logic.
*   [ ] **Task 7.5: Finalize `.gitignore`**
    *   Review and add any other necessary files/directories to ignore (e.g., specific IDE files, OS-specific files).
*   [ ] **Task 7.6: Add `LICENSE` file**
    *   Create a `LICENSE` file with the MIT License text.

## General/Ongoing Tasks

*   [ ] **Task G1: Error Handling**
    *   Implement robust error handling (try-except blocks, user-friendly messages in Gradio) for:
        *   File I/O errors.
        *   Model loading failures.
        *   Training interruptions.
        *   API errors (if any external APIs are used).
        *   Invalid user inputs.
*   [ ] **Task G2: Asynchronous Operations**
    *   Ensure all long-running tasks in Gradio (fine-tuning, extensive subtitle generation) are handled asynchronously to prevent UI freezes. Use `gr.Request` or background task patterns.
*   [ ] **Task G3: Hardware Requirements Documentation**
    *   Clearly document minimum and recommended GPU specifications (VRAM, model type) in the `README.md`.
*   [ ] **Task G4: Modular Design Adherence**
    *   Continuously ensure that the architecture remains modular with clearly defined inputs and outputs for each function and module.
*   [ ] **Task G5: Package Management with `uv`**
    *   Consistently use `uv` for installing and managing all Python packages.
    *   Maintain `pyproject.toml` (if `uv` uses it for project definition, or a `requirements.txt` generated by `uv pip freeze > requirements.txt`).
*   [ ] **Task G6: Jupyter Notebook for Module Testing**
    *   For each new significant function or module, create/update a corresponding Jupyter Notebook in the `notebooks/` directory to demonstrate and test its functionality.
*   [ ] **Task G7: Type Annotations and Docstrings**
    *   Ensure all new code includes type annotations and Google-style docstrings.