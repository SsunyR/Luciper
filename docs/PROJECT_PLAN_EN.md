# Project Plan: Luciper

## 1. Project Overview

* **Objective:** Develop an open-source service enabling users to easily create and utilize OpenAI Whisper models specialized for specific domains (topics, environments, pronunciation, etc.).
* **Core Features:**
    * Selection and Fine-tuning of various Whisper-based models (including `tiny`, `base`, `small`, `medium`, `large`, and latest versions).
    * Initial Fine-tuning and continuous learning based on TTS (Text-to-Speech) using text data.
    * Generating subtitles for audio files using the fine-tuned model.
    * Building a pipeline for continuous model performance improvement by editing generated subtitles and using them for further training.
    * Functionality to save, manage, and modify fine-tuned models and associated metadata (alias, domain description, etc.).
    * Feature to check accuracy (e.g., WER - Word Error Rate) by comparing generated and corrected subtitles.
* **Development Language:** `Python`
* **Development Method:** Pair programming using an AI agent (Vibe coding).
* **Package Management:** Use `uv`.
* **Architecture:** Modular structure with clearly defined Inputs/Outputs for each function.
* **Testing:** Functional verification using `Jupyter Notebook` during the development of each module.

## 2. Target Users

* Developers or researchers needing speech recognition for specific domains.
* Individuals or teams wanting to build their own high-quality speech recognition models.
* Users aiming to incrementally improve model performance through iterative subtitle generation and correction.
* Users with limited coding experience who want to experience Whisper Fine-tuning through a GUI.

## 3. Technology Stack (Proposed Minimum Optimal Configuration)

* **Core Language:** `Python 3.10+`
* **ASR Model:** OpenAI Whisper (Using the Whisper implementation within the Hugging Face `transformers` library - `transformers` might be more convenient for fine-tuning and pipeline construction)
* **Fine-tuning Framework:** Hugging Face `transformers` + `accelerate` + `datasets` (Efficient for fine-tuning pipelines, dataset processing, and distributed training support)
* **TTS Engine:**
    * `Piper TTS`: Fast, high-quality, locally running TTS engine developed by Mozilla. Supports various languages and voices. (Suitable for offline fine-tuning data generation)
    * `gTTS`: Based on Google Translate API, simple but requires an internet connection and has limitations. (Can be used for initial testing)
* **Web Framework & UI:** `Gradio`
    * Python-native, very easy for building ML model demos and interfaces.
    * Easily implements UI components like audio input, text input/output, buttons, etc.
    * Meets the "minimum tech stack" requirement and allows for rapid prototyping.
* **Package Manager:** `uv`
* **Data Handling:** `Pandas` (Metadata management), Hugging Face `datasets` (Audio/text dataset structuring and processing)
* **Audio Processing:** `librosa` or `soundfile` (Audio file loading, preprocessing)
* **Notebook Environment:** `Jupyter Notebook` / `JupyterLab` (Module testing and experimentation)
* **Storage:**
    * Model Files: Local file system or Cloud Storage (Optional)
    * Metadata: `JSON` files or `SQLite` (Start simply with JSON initially, consider SQLite for future scalability)

### Rationale:

* The `transformers` library offers many convenient features like the `Trainer` API for Whisper fine-tuning and dataset processing, increasing development productivity.
* `Piper TTS` enables high-quality TTS generation in a local environment, allowing dataset construction for fine-tuning without external API dependencies.
* `Gradio` allows rapid development of interactive web UIs using only Python code, without complex frontend development, making it suitable for a minimal tech stack and AI pair programming environment.
* `uv` is a fast, next-generation Python package manager reflecting current trends.

## 4. System Architecture (Modular Structure)
```
+-------------------------+      +-------------------------+      +----------------------------+
|   Web Interface (Gradio)|----->| Model Management Module |----->| Fine-tuning Module         |
| - Create/Select Model   |<-----| - Select Base Model     |<-----| - Load/Preprocess Data     |
| - Manage Metadata       |      | - Save/Load Fine-tuned |      | - Whisper Fine-tuning      |
| - Select Task (Train/Infer)|   | - Metadata CRUD         |      | (transformers.Trainer)     |
| - Gen/Edit/Compare Subs |      +-------------------------+      | - Save Model               |
| - Generate TTS Train Data|               |                      +------------+---------------+
+-----------+-------------+                |                                     |
|                                          | (Model Path/Object)                 | (Fine-tuned Model)
|                                          V                                     V
+-----------+-------------+      +-------------------------+      +----------------------------+
| Data Preparation Module |<-----| Inference Module        |----->| Evaluation Module          |
| - Text -> TTS Audio     |      | - Load Audio            |      | - Generated vs Corrected Subs|
| - Align Audio + Text    |      | - Load Selected Model   |      | - Calculate WER, etc.      |
| - Process Edited Subs   |----->| - Generate Subtitles    |----->| - Visualize/Display Results|
+-------------------------+      |   (Whisper)             |      +----------------------------+
                                 +-------------------------+
         (Training Data)         (Generated Subtitles)              (Accuracy Metrics)
```

### Module Descriptions:

* **Web Interface (Gradio):** Provides the user interface. Receives user input, calls other modules, and displays results.
    * `Input:` User actions (file uploads, text input, button clicks, etc.)
    * `Output:` UI updates (model list, subtitle text, accuracy metrics, etc.)
* **Model Management Module:** Selects Whisper Base models, saves/loads/manages fine-tuned models and metadata.
    * `Input:` Model name, path, metadata information
    * `Output:` Model object, model file path, saved/modified metadata
* **Data Preparation Module:** Creates and prepares datasets for fine-tuning.
    * `Input:` Text (for TTS), Audio file + corrected subtitle text
    * `Output:` Dataset formatted for Whisper Fine-tuning (audio + text pairs)
* **Fine-tuning Module:** Performs the actual model fine-tuning (Utilizing `transformers.Trainer`).
    * `Input:` Prepared dataset, model path (Base or previously fine-tuned model), training configuration (Hyperparameters)
    * `Output:` Fine-tuned model file path
* **Inference Module:** Generates subtitles from an audio file using the selected model.
    * `Input:` Audio file path, path to the model to use
    * `Output:` Generated subtitle text (or file format like SRT)
* **Evaluation Module:** Compares generated subtitles with user-corrected subtitles to calculate accuracy (`WER`, etc.).
    * `Input:` Generated subtitle text, corrected subtitle text
    * `Output:` Accuracy metrics (numeric or visualization data)

## 5. Web Interface Detailed Features

* **Model Management:**
    * `[Create]` Start a new Fine-tuning model session (Select Base model, input metadata like alias/domain).
    * `[Select]` Display list of previously saved Fine-tuned models and allow selection.
    * `[Modify]` Edit metadata (alias, domain, etc.) of the selected model.
    * `[Delete]` Delete the selected Fine-tuned model (Caution needed).
* **Task Selection:**
    * After selecting a model, choose the task to perform:
        * `[TTS-based Training]`: Perform Fine-tuning by converting input text to TTS.
        * `[Generate Subtitles]`: Upload audio/video file to generate subtitles and then edit the result.
* **TTS-based Training:**
    * Text input area (Paste large text or upload text file).
    * Option to select TTS engine (`Piper`, `Coqui`, etc.) and voice (if available).
    * Start Fine-tuning button.
    * Display training progress (Logs or Progress bar).
* **Subtitle Generation:**
    * Audio/Video file upload interface.
    * Start Subtitle Generation button.
    * Display area for generated subtitles.
* **Subtitle Editing and Evaluation:**
    * Text area to edit the generated subtitles.
    * Feature to display comparison between original (generated) and edited subtitles.
    * `[Check Accuracy]` button: Display calculation results like `WER`.
    * `[Train with Corrections]` button: Execute the pipeline to use the corrected subtitle data for additional Fine-tuning of the currently selected model.

## 6. Development Plan (Phased Approach - Utilizing Vibe Coding)

* **Phase 1: Environment Setup & Basic Function Implementation**
    * Set up project structure (`src`, `tests`, `notebooks`, `data`, etc.).
    * Set up virtual environment using `uv` and install basic libraries (`python`, `openai-whisper` or `transformers`, `torch`, `jupyterlab`, `uv`).
    * Test loading a base Whisper model and generating subtitles in `Jupyter Notebook`.
    * Implement and test the basic subtitle generation module (`Inference Module` basics) with simple audio files.
* **Phase 2: Fine-tuning Pipeline Construction**
    * Build the basic fine-tuning pipeline using Hugging Face `transformers` and `datasets` (using sample datasets).
    * Implement the `Fine-tuning Module` and test based on Notebooks.
    * Configure and test the GPU environment (GPU is essential for fine-tuning).
* **Phase 3: Data Preparation Function Implementation**
    * Implement Text input -> TTS conversion -> Audio file generation (`Data Preparation Module` - TTS part).
    * Integrate `Piper TTS` and test in Notebook.
    * Implement logic to connect the generated TTS dataset to the fine-tuning pipeline.
* **Phase 4: Model Management Function Implementation**
    * Implement saving and loading functionality for fine-tuned models (`Model Management Module` core function).
    * Implement save/load/modify functionality for model metadata (`JSON` or `SQLite`).
    * Test based on Notebooks.
* **Phase 5: Basic Gradio Web Interface Construction**
    * Install `Gradio` and create the basic app structure (`Web Interface Module`).
    * Implement model selection feature (Base, list of Fine-tuned).
    * Implement Audio file upload -> Subtitle generation request -> `Inference Module` integration -> Display results.
    * Implement Text input -> TTS-based training request -> `Data Preparation` and `Fine-tuning Module` integration (Requires background execution).
* **Phase 6: Complete Subtitle Editing and Continuous Learning Pipeline**
    * Add subtitle editing functionality to the `Gradio` interface.
    * Integrate the `Evaluation Module` for comparing generated vs. corrected subtitles and displaying accuracy results.
    * Complete the continuous learning pipeline: pass corrected subtitle data to `Data Preparation Module` and re-run `Fine-tuning Module`.
* **Phase 7: Testing, Refactoring, and Documentation**
    * Perform integration testing for each module and the entire workflow.
    * Refactor code and incorporate feedback from the Vibe coding process.
    * Write `README` (Project introduction, installation guide, usage, contribution guide, etc.).
    * Improve comments and code documentation.

## 7. Test Strategy

* **Module Unit Testing:** Verify the core functionality of each module using `Jupyter Notebook` with various inputs to ensure expected behavior (Write and execute test cases with the AI pair programming partner).
* **Integration Testing:** Verify that the entire workflow (Data input -> Training/Inference -> Result check -> Correction -> Re-training) operates correctly through the `Gradio` web interface.
* **Data Testing:** Test the system's robustness using various types of audio files (clean speech, noisy environments, different accents, etc.) and text data.

## 8. Additional Considerations

* **Hardware Requirements:** Whisper Fine-tuning requires significant computing resources (especially GPU memory). Users should be informed about minimum/recommended GPU specifications (e.g., NVIDIA GPU with 8GB+ VRAM recommended, `Large` model requires more).
* **Error Handling:** Need to handle potential exceptions like file I/O errors, model loading failures, training interruptions, etc.
* **Asynchronous Processing:** Long-running tasks like Fine-tuning in the web interface should be handled asynchronously to improve user experience (`Gradio` might handle this inherently, or use `asyncio`).

## 9. License

* MIT