# Full Guide: Pub Review Sentiment Analyzer - Flan-T5 Fine-tuning & Deployment

This comprehensive guide walks through fine-tuning the `google/flan-t5-small` model to classify pub review sentiment. It covers Hugging Face setup, fine-tuning in a new Google Colab notebook using your ~100 review `pub_reviews.txt` dataset, and deploying the trained LoRA adapter to a Hugging Face Space.

## Phase 0: Hugging Face Account & Space Setup

### 1. Create a Hugging Face Account:

* If you don't have one, go to [huggingface.co](https://huggingface.co/) and click "Sign Up".
* Follow the instructions to create your account.

### 2. Create a Hugging Face Access Token (API Token):

This token allows your Colab notebook to interact with the Hugging Face Hub.

* Log in to [huggingface.co](https://huggingface.co/).
* Click on your profile picture (top right) -> "Settings".
* In the left sidebar, click on "Access Tokens".
* Click "New token".
* Give your token a descriptive name (e.g., `colab-flan-t5-sentiment`).
* Assign it a role. For this project, "read" is sufficient if you're only downloading models. If you plan to push models/adapters to the Hub, choose "**write**".
* Click "Generate a token".
* **Important:** Copy the generated token immediately and save it securely. You won't see it again.

### 3. Create a New Hugging Face Space (for Deployment):

* On Hugging Face, click on your profile picture -> "New Space".
* Owner: Select your username.
* Space name: Choose a unique name (e.g., `flan-t5-pub-sentiment-analyzer`).
* License: Select an appropriate license (e.g., `mit`).
* Select Space SDK: Choose `Gradio`.
* Space hardware: Select the `CPU basic - FREE` tier.
* Visibility: Choose "Public" or "Private".
* Click "Create Space". This will initialize your Space with default files.

## Phase 1: Setting Up Your NEW Google Colab Notebook for Fine-tuning

### 1. New Notebook & GPU Runtime:

* Go to [colab.research.google.com](https://colab.research.google.com/).
* Click on "File" -> "New notebook".
* **Set Runtime to GPU:** In the Colab menu, go to "Runtime" -> "Change runtime type". Under "Hardware accelerator," select `GPU` (e.g., T4). Click "Save".

### 2. Add Hugging Face Token as a Secret in Colab:

* In your Colab notebook, click on the "Key" icon (Secrets) in the left sidebar.
* Click "Add a new secret".
* Name: Enter `HF_TOKEN`.
* Value: Paste your Hugging Face Access Token from Phase 0, Step 2.
* Enable the "Notebook access" toggle for this secret.

### 3. Critical First Cell: Library Installation & Hugging Face Login:

This will be the very first code cell you run in your new notebook.

```python
# Critical First Cell for Colab Setup
# Run this cell first, then RESTART THE RUNTIME before proceeding.

# Attempt to fix any lingering sympy issues
!pip uninstall -y sympy
!pip install --upgrade sympy

# Ensure pip is up-to-date
!pip install --upgrade pip

# Install necessary libraries
!pip install transformers[sentencepiece] datasets peft accelerate bitsandbytes torch evaluate rouge_score nltk

print("\n--- Library installation/upgrade and HF login attempt complete. ---")
print("IMPORTANT: Please RESTART THE RUNTIME now before running any other cells.")
print("Go to 'Runtime' -> 'Restart session' (or 'Restart runtime') in the Colab menu.")
````

  * **VERY IMPORTANT:** After this cell finishes executing, you **MUST** restart the Colab runtime. Go to the Colab menu: "Runtime" -\> "Restart session" (or "Restart runtime"). Click "Yes".

## Phase 2: Preparing Your Pub Review Data for Flan-T5 in Colab

### 1\. Upload Your `pub_reviews.txt` (with \~100 reviews):

  * After the Colab runtime has restarted, use the "Files" tab (folder icon) in the left sidebar to upload your `pub_reviews.txt` file (the one with \~100 examples).

### 2\. Create and Run Preprocessing Logic for T5:

  * Create a new code cell in your Colab notebook.
  * Paste the entire script below into this cell and run it. This script defines the `create_t5_training_data` function and then calls it.

<!-- end list -->

```python
# preprocess_data_for_t5.py
# This entire block should be pasted into a single Colab code cell.

import json
import os

def create_t5_training_data(input_file="pub_reviews.txt", output_file="train_t5.jsonl", instruction_prefix="As a helpful pirate chatbot from The Pirate's Forge pub, answer the question: "):
    """
    Processes a text file with Q&A pairs into a JSONL file suitable for T5 fine-tuning.
    Each line in the output JSONL file will be a dictionary with "input_text" and "target_text".
    The "input_text" will be prefixed with an instruction.
    This version is adapted for sentiment analysis where the "question" is the review
    and the "answer" is the sentiment label.
    """
    processed_records = 0
    malformed_pairs = 0
    skipped_lines = 0

    current_review_text = None # Will store the review text

    print(f"Starting T5 sentiment data processing of {input_file}...")

    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found in the current Colab session directory.")
        print(f"Please ensure '{input_file}' has been uploaded to your Colab session.")
        return # Stop if input file is missing

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for i, line in enumerate(infile):
            line_stripped = line.strip()

            if not line_stripped: # Skip blank lines
                skipped_lines +=1
                continue

            if line_stripped.startswith("Q: What is the sentiment of this review:"):
                if current_review_text: # If a Q (review) was pending without an A (sentiment)
                    print(f"Warning: Line {i+1} - New review found, but previous review '{current_review_text}' had no matching sentiment label. Skipping previous.")
                    malformed_pairs += 1
                current_review_text = line_stripped.replace("Q: What is the sentiment of this review:", "").strip()

            elif line_stripped.startswith("A:"):
                if not current_review_text: # If an A (sentiment) appears without a preceding Q (review)
                    print(f"Warning: Line {i+1} - Sentiment label found without a preceding review: '{line_stripped}'. Skipping this label.")
                    malformed_pairs += 1
                    continue

                sentiment_label = line_stripped[3:].strip() # "Positive", "Negative", or "Neutral"

                input_text_for_model = f"{instruction_prefix}{current_review_text}"
                target_text_for_model = sentiment_label # Just the label

                json_record = {
                    "input_text": input_text_for_model,
                    "target_text": target_text_for_model
                }
                outfile.write(json.dumps(json_record) + "\n")
                processed_records += 1
                current_review_text = None # Reset for the next pair

            else: # Line is not blank, not Q:, not A:
                if current_review_text: # If we were expecting an A for a pending Q
                    print(f"Warning: Line {i+1} - Expected A: (sentiment label) for review '{current_review_text}', but found other text: '{line_stripped[:50]}...'. Skipping this review-sentiment pair.")
                    malformed_pairs += 1
                    current_review_text = None
                else: # Just an extra line not part of Q/A structure
                    print(f"Info: Line {i+1} - Skipping non-Q/A line: '{line_stripped[:50]}...'")
                skipped_lines +=1

        if current_review_text:
            print(f"Warning: End of file - Review '{current_review_text}' has no matching sentiment label. Skipping.")
            malformed_pairs += 1

    print(f"\nT5 Sentiment Data Processing complete.")
    print(f"Successfully processed {processed_records} review-sentiment pairs into {output_file}.")
    if malformed_pairs > 0:
        print(f"Skipped {malformed_pairs} incomplete pairs.")
    if skipped_lines > 0:
        print(f"Skipped {skipped_lines} other lines.")

# --- This is the part that actually RUNS the preprocessing ---
print("\nRunning T5 preprocessing for sentiment analysis directly in Colab cell...")
custom_instruction_prefix_for_sentiment = "What is the sentiment of this review: "
create_t5_training_data(
    input_file="pub_reviews.txt",
    output_file="train_t5.jsonl",
    instruction_prefix=custom_instruction_prefix_for_sentiment
)
print("\nT5 Preprocessing for sentiment complete! 'train_t5.jsonl' should now exist in your Colab session files.")
print("Each line should look like: {\"input_text\": \"What is the sentiment of this review: [Review Text]\", \"target_text\": \"[Sentiment Label]\"}")
```

  * Verify that `train_t5.jsonl` is created and the output indicates \~100 records processed.

## Phase 3: Fine-tuning Flan-T5-Small in Colab

### 1\. Create and Run Fine-tuning Logic for T5:

  * Create a new code cell.
  * Paste the entire content of the Flan-T5 training script below into this cell and run it. This version includes the fix for `TrainingArguments`.

<!-- end list -->

```python
# train_flan_t5_lora.py
# This entire block should be pasted into a single Colab code cell.
# Includes manual data loading and the fix for TrainingArguments.

import os
import torch
from datasets import load_dataset, Dataset # Ensure Dataset is imported
from transformers import (
    AutoModelForSeq2SeqLM, # For T5 models
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq, # Specific data collator for T5
)
from peft import LoraConfig, get_peft_model, TaskType
import math
import json # For manually loading JSONL

def train_model():
    # --- Model and Tokenizer Configuration ---
    base_model_name = "google/flan-t5-small"
    adapter_output_dir = "./pirate_flan_t5_lora_adapter" # Default output, can be changed like _attempt2
    data_file = "train_t5.jsonl"

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # --- Load Base Model ---
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        device_map="auto"
    )
    model.config.use_cache = False

    # --- LoRA Configuration ---
    print("Configuring LoRA for Flan-T5...")
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q", "v"]

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Load Dataset (Manual JSONL Loading) ---
    print(f"Attempting to load dataset from: {data_file}")
    if not os.path.exists(data_file):
        print(f"ERROR: Data file '{data_file}' not found. Please run the T5 preprocessing script first.")
        return False

    try:
        print(f"Manually reading and parsing JSONL file: {data_file}")
        data_list = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))

        if not data_list:
            print(f"ERROR: No data loaded from {data_file}.")
            return False

        raw_dataset = Dataset.from_list(data_list)
        dataset_size = len(raw_dataset)
        print(f"Successfully loaded and created Dataset with {dataset_size} records.")

        if dataset_size == 0:
            print("ERROR: The dataset is empty after loading.")
            return False

    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return False

    # --- Tokenize Dataset ---
    print("Tokenizing dataset for T5...")
    max_source_length = 512
    max_target_length = 10

    def t5_tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_source_length,
            truncation=True,
            padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"],
                max_length=max_target_length,
                truncation=True,
                padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    columns_to_remove = ["input_text", "target_text"]
    actual_columns_to_remove = [col for col in columns_to_remove if col in raw_dataset.column_names]
    tokenized_dataset = raw_dataset.map(
        t5_tokenize_function,
        batched=True,
        remove_columns=actual_columns_to_remove
    )
    print("Dataset tokenized.")

    # --- Training Arguments ---
    print("Setting up training arguments for Flan-T5 LoRA fine-tuning (Sentiment Analysis)...")
    num_train_epochs = 10
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim_choice = "adamw_torch"
    print(f"Using optimizer: {optim_choice}")
    fp16_enabled = False
    print(f"FP16 training enabled: {fp16_enabled}")
    learning_rate = 3e-5
    weight_decay = 0.01
    max_grad_norm = 1.0
    warmup_ratio = 0.1
    steps_per_epoch = math.ceil(dataset_size / (per_device_train_batch_size * gradient_accumulation_steps))
    logging_steps = max(1, steps_per_epoch // 2)
    print(f"Dataset size: {dataset_size}, Steps per epoch: {steps_per_epoch}, Logging every: {logging_steps} steps")

    training_arguments = TrainingArguments(
        output_dir="./results_flan_t5_sentiment_colab",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim_choice,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16_enabled,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none"
    )

    # --- Initialize Trainer ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if fp16_enabled else None
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Train the Model ---
    print("Starting Flan-T5 LoRA fine-tuning for Sentiment Analysis...")
    print("IMPORTANT: Monitor the 'loss' value. It MUST decrease consistently for effective training.")
    try:
        trainer.train()
        print("Flan-T5 LoRA fine-tuning for Sentiment Analysis complete!")
    except Exception as e:
        print(f"ERROR during trainer.train(): {e}")
        return False

    # --- Save LoRA Adapter ---
    print(f"Saving LoRA adapter to {adapter_output_dir}...")
    try:
        model.save_pretrained(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)
        print(f"Adapter saved to {adapter_output_dir}. You can now zip and download.")
        return True
    except Exception as e:
        print(f"ERROR saving adapter: {e}")
        return False

# --- This is the part that actually RUNS the training ---
print("\nAttempting to run Flan-T5 train_model() for Sentiment Analysis directly in Colab cell...")
if train_model():
    print("\n✅ Flan-T5 train_model() for Sentiment Analysis executed successfully in Colab.")
else:
    print("\n❌ Flan-T5 train_model() for Sentiment Analysis FAILED in Colab. Check output for errors.")
```

  * Monitor the output for decreasing training loss. It should save the adapter to `./pirate_flan_t5_lora_adapter`.

## Phase 4: Testing Your Fine-tuned Flan-T5 in Colab

### 1\. Create and Run Colab Testing Script for T5:

  * Once Phase 3 completes successfully, create a new code cell.
  * Paste the entire content of the Flan-T5 testing script below into this cell and run it. This version includes the fix for the `test_sentiment_model` function call.

<!-- end list -->

```python
# test_flan_t5_colab.py
# Test script to run in a new Colab cell AFTER the Flan-T5 train_model() has completed.
# Corrected to ensure test_sentiment_model function is properly defined and called.

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from peft import PeftModel # For loading the LoRA adapter
import torch
import os

def test_sentiment_model_logic(): # Main logic wrapped in a function
    # --- Configuration ---
    base_model_name_colab_test = "google/flan-t5-small"
    # This should match the adapter_output_dir from your Flan-T5 training script
    adapter_path_colab_test = "./pirate_flan_t5_lora_adapter" # Default from training script
    # This MUST match the instruction_prefix used during T5 preprocessing
    instruction_prefix_colab_test = "What is the sentiment of this review: "

    if not os.path.exists(adapter_path_colab_test):
        print(f"ERROR: Adapter path '{adapter_path_colab_test}' not found!")
        print(f"Ensure training was successful and adapter was saved to '{adapter_path_colab_test}'.")
        return
    else:
        print(f"Adapter path '{adapter_path_colab_test}' found. Proceeding with loading...")

    colab_base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name_colab_test, device_map="auto")
    print("Base Flan-T5 model loaded.")
    colab_tokenizer = AutoTokenizer.from_pretrained(base_model_name_colab_test)
    print("Tokenizer loaded.")

    colab_fine_tuned_model = None
    try:
        colab_fine_tuned_model = PeftModel.from_pretrained(colab_base_model, adapter_path_colab_test)
        colab_fine_tuned_model = colab_fine_tuned_model.merge_and_unload()
        print("LoRA adapter loaded and merged successfully for Colab test.")
    except Exception as e:
        print(f"Error loading or merging LoRA adapter in Colab: {e}")
        colab_fine_tuned_model = colab_base_model

    colab_pirate_pipeline = None
    if colab_fine_tuned_model and colab_tokenizer:
        colab_pirate_pipeline = pipeline(
            task="text2text-generation",
            model=colab_fine_tuned_model,
            tokenizer=colab_tokenizer,
            max_length=10 # For short sentiment labels
        )
        print("Pipeline created.")
    else:
        print("ERROR: Model or tokenizer not loaded correctly. Pipeline not created.")
        return

    if colab_pirate_pipeline:
        sample_reviews_for_testing = [
            "The ale was magnificent and the crew was friendly! A true gem!",
            "Absolutely dreadful experience. The ship was leaky and the captain was a fool.",
            "It was a standard pub, nothing special to note either way.",
            "Best pirate grub I've had on the seven seas! The Kraken Calamari is legendary!",
            "Waited an eternity for a simple mug of grog. Service needs to improve."
        ]
        for review_text in sample_reviews_for_testing:
            input_text_for_test = f"{instruction_prefix_colab_test}{review_text}"
            print(f"\nTesting in Colab with input: \"{input_text_for_test}\"")
            try:
                responses = colab_pirate_pipeline(input_text_for_test)
                if responses and len(responses) > 0:
                    predicted_sentiment_label = responses[0]['generated_text'].strip()
                    print(f"  Colab Test - Predicted Sentiment: {predicted_sentiment_label}")
                else:
                    print("  Colab Test - No response generated.")
            except Exception as e:
                print(f"  Error during Colab test generation for review '{review_text}': {e}")
    else:
        print("\nSkipping tests as the pipeline was not created.")

# --- Call the main testing logic ---
if __name__ == '__main__': # Ensures this runs only when script is executed directly
    print("\nAttempting to run Flan-T5 Sentiment Analyzer test logic directly in Colab cell...")
    test_sentiment_model_logic()
    print("\nFlan-T5 Sentiment Analyzer testing complete.")

# For direct execution in a Colab cell after defining the function:
print("\nRunning Flan-T5 Sentiment Analyzer test_sentiment_model_logic() directly in Colab cell...")
test_sentiment_model_logic()
print("\nFlan-T5 Sentiment Analyzer testing logic execution finished.")
```

  * Examine the "Predicted Sentiment" outputs. They should be accurate ("Positive", "Negative", "Neutral").

## Phase 5: Downloading Your Trained Adapter from Colab

### 1\. Zip and Download:

  * If Colab testing (Phase 4) shows good results, create a new code cell and run:

<!-- end list -->

```python
import os
from google.colab import files

# This MUST match the adapter_output_dir from your Flan-T5 training script
adapter_directory = "./pirate_flan_t5_lora_adapter"
zip_filename = "pirate_flan_t5_lora_adapter.zip"

if os.path.exists(adapter_directory):
    print(f"Zipping directory: {adapter_directory}")
    # The ! prefix runs shell commands
    !zip -r {zip_filename} {adapter_directory}

    print(f"\nAttempting to download {zip_filename}...")
    files.download(zip_filename)
    print(f"\nIf download doesn't start automatically, check your browser's download permissions for Colab.")
else:
    print(f"ERROR: Directory '{adapter_directory}' not found. Cannot zip and download. Was training successful?")
```

  * Download `pirate_flan_t5_lora_adapter.zip` to your local computer.

## Phase 6: Deploying to Your Hugging Face Space

### 1\. Prepare Hugging Face Space Files:

  * `requirements.txt`: In your Space ("Files and versions" tab), ensure this file exists and contains:

    ```txt
    transformers
    datasets
    peft
    accelerate
    bitsandbytes
    torch
    gradio
    sentencepiece
    ```

    Commit changes if you edit it.

  * `app.py`: You will use the script below. This version is for Flan-T5 and includes the fix for the `device` argument in the pipeline.

      * In your Space, edit `app.py`. Delete any existing content.
      * Paste the content of the script below into it.
      * Crucially, ensure the `ADAPTER_FOLDER_NAME` variable in this `app.py` script matches the name of the folder you will upload (e.g., `pirate_flan_t5_lora_adapter`).
      * Commit changes.

<!-- end list -->

```python
# app.py for Flan-T5 Sentiment Analyzer on Hugging Face Spaces
# Fixed: Removed device argument from pipeline creation when device_map="auto" is used.

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from peft import PeftModel
import os

# --- Configuration ---
BASE_MODEL_NAME = "google/flan-t5-small"
# This MUST match the name of the folder you upload to Hugging Face Spaces
ADAPTER_FOLDER_NAME = "pirate_flan_t5_lora_adapter" # Ensure this matches your uploaded folder
ADAPTER_PATH = f"./{ADAPTER_FOLDER_NAME}"
INSTRUCTION_PREFIX = "What is the sentiment of this review: "

model = None
tokenizer = None
sentiment_pipeline = None
adapter_loaded_successfully = False

def load_model_and_pipeline():
    global model, tokenizer, sentiment_pipeline, adapter_loaded_successfully
    print("Loading Flan-T5 model and tokenizer for Gradio app...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        print(f"Loading base model: {BASE_MODEL_NAME}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto"
        )
        base_model.config.use_cache = False

        if os.path.exists(ADAPTER_PATH) and \
           (os.path.exists(os.path.join(ADAPTER_PATH, "adapter_model.safetensors")) or \
            os.path.exists(os.path.join(ADAPTER_PATH, "adapter_model.bin"))):
            print(f"LoRA adapter found at {ADAPTER_PATH}. Attempting to load...")
            try:
                peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
                model = peft_model.merge_and_unload()
                adapter_loaded_successfully = True
                print("LoRA adapter loaded and merged successfully.")
            except Exception as e:
                print(f"Error loading LoRA adapter: {e}")
                model = base_model
        else:
            print(f"LoRA adapter not found at '{ADAPTER_PATH}'. Using base model.")
            model = base_model

        sentiment_pipeline = pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=10
        )
        status_message = "Flan-T5 Sentiment Analyzer Ready (Adapter Loaded)!" if adapter_loaded_successfully else "Base Flan-T5 Ready (Adapter NOT Loaded/Found)!"
        print(status_message)
        return status_message
    except Exception as e:
        print(f"CRITICAL ERROR during model loading or pipeline creation: {e}")
        sentiment_pipeline = None
        return f"Error loading model: {str(e)}"

def classify_sentiment(review_text):
    if not sentiment_pipeline:
        return "ERROR: Sentiment analysis pipeline not initialized. Check Space logs."
    if not review_text or not review_text.strip():
        return "Please enter a review to analyze."
    prompt = f"{INSTRUCTION_PREFIX}{review_text.strip()}"
    print(f"Classifying sentiment for review. Prompt: \"{prompt}\"")
    try:
        outputs = sentiment_pipeline(prompt)
        if outputs and len(outputs) > 0:
            predicted_label = outputs[0]['generated_text'].strip()
            print(f"Predicted sentiment: {predicted_label}")
            if predicted_label.lower() in ["positive", "negative", "neutral"]:
                return predicted_label.capitalize()
            else:
                print(f"Warning: Model output unexpected label: '{predicted_label}'")
                return f"Model Output: {predicted_label} (Expected Positive/Negative/Neutral)"
        else:
            return "No sentiment predicted by the model."
    except Exception as e:
        print(f"Error during sentiment prediction: {e}")
        return f"Error during prediction: {str(e)}"

initial_status = load_model_and_pipeline()

iface = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter a pub review here, matey...", label="Pub Review"),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="🏴‍☠️ Pub Review Sentiment Analyzer 🦜",
    description=f"Ahoy! Enter a review for 'The Pirate's Forge' (or any pub) and I'll tell ye if it's good, bad, or just so-so. Powered by Flan-T5-Small!\nModel Status: {initial_status}",
    examples=[
        ["The grog was excellent and the shanties were lively! Best pub in port!"],
        ["Service was terribly slow, and my fish was cold."],
        ["It was a standard pub, nothing special to note either way."]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Launching Gradio app for Flan-T5 Sentiment Analyzer...")
    iface.launch()
```

  * **Clean the Space:** Delete any old adapter folders from previous attempts in your Space.

### 2\. Upload Fine-tuned Adapter to Space:

  * On your local computer, unzip the `pirate_flan_t5_lora_adapter.zip` file.
  * In your Hugging Face Space ("Files and versions" tab), click "Add file" -\> "Upload folder."
  * Select and upload the entire unzipped adapter folder to the root of your Space.

### 3\. Restart and Test Space:

  * Go to the "Settings" tab of your Space (or the "..." menu) and "Restart this Space."
  * Monitor the "Logs" tab. Look for messages confirming the base model and your LoRA adapter are loaded successfully.
  * Go to the "App" tab. Your "Pub Review Sentiment Analyzer" should be live\! Test it with various reviews.

<!-- end list -->

```
```
