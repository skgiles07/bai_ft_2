Here is the content of your file formatted using Markdown, which should render correctly on GitHub.

### Presentation Treasure Map: Pub Review Sentiment Analyzer - Flan-T5 Fine-tuning & Deployment

This guide provides a complete walkthrough for fine-tuning the `google/flan-t5-small` model to classify pub review sentiment[cite: 1]. We'll cover setting up Hugging Face, fine-tuning in a new Google Colab notebook using your \~100 review `pub_reviews.txt` dataset, and then deploying the trained LoRA adapter to a Hugging Face Space[cite: 2].

#### Phase 0: Hugging Face Account & Space Setup

1.  **Create a Hugging Face Account:**
    If you don't have one, go to huggingface.co and click "Sign Up"[cite: 3]. Follow the instructions to create your account[cite: 4].

2.  **Create a Hugging Face Access Token (API Token):**
    This token allows your Colab notebook to interact with the Hugging Face Hub (e.g., for logging in, pushing models/adapters if needed, though not strictly required for downloading public models)[cite: 4].

      * Log in to huggingface.co[cite: 5].
      * Click on your profile picture (top right) -\> "Settings"[cite: 5].
      * In the left sidebar, click on "Access Tokens"[cite: 6].
      * Click "New token"[cite: 6].
      * Give your token a descriptive name (e.g., "colab-fine-tuning")[cite: 6].
      * Assign it a role. For most fine-tuning tasks where you might want to save your model, "write" access is appropriate. For just downloading, "read" is fine. Choose "write" to be safe for future projects[cite: 7, 8].
      * Click "Generate a token"[cite: 8].
      * **Important:** Copy the generated token immediately and save it somewhere secure (like a password manager). You won't be able to see it again after you navigate away from the page[cite: 9, 10].

3.  **Create a New Hugging Face Space (for Deployment):**
    On Hugging Face, click on your profile picture -\> "New Space"[cite: 11].

      * **Owner:** Select your username[cite: 12].
      * **Space name:** Choose a unique name for your project (e.g., `pub-sentiment-analyzer-demo`)[cite: 12].
      * **License:** Select an appropriate license (e.g., `mit` or `apache-2.0`)[cite: 13].
      * **Select Space SDK:** Choose `Gradio`. This will provide a Python environment with Gradio pre-installed[cite: 13, 14].
      * **Space hardware:** Select the `CPU basic - FREE` tier for this demo[cite: 14].
      * **Visibility:** Choose "Public" if you want anyone to be able to see it, or "Private" if not[cite: 15].
      * Click "Create Space"[cite: 16].
        Your new Space will be initialized with some default files (`app.py`, `README.md`, `requirements.txt`). We will modify these later[cite: 16, 17].

#### Phase 1: Setting Up Your NEW Google Colab Notebook for Fine-tuning

1.  **New Notebook & GPU Runtime:**
    Go to colab.research.google.com[cite: 17]. Click on `"File"` -\> `"New notebook"`[cite: 18].
    Set Runtime to GPU: In the Colab menu, go to `"Runtime"` -\> `"Change runtime type"`[cite: 18]. Under "Hardware accelerator," select `GPU` (e.g., T4). Click "Save"[cite: 19].

2.  **Add Hugging Face Token as a Secret in Colab:**
    In your Colab notebook, click on the "Key" icon (Secrets) in the left sidebar[cite: 20]. Click "Add a new secret"[cite: 21].

      * **Name:** Enter `HF_TOKEN`[cite: 21].
      * **Value:** Paste your Hugging Face Access Token that you copied in Phase 0, Step 2[cite: 21].
      * Enable the "Notebook access" toggle for this secret[cite: 22]. This keeps your token secure and allows your notebook to access it[cite: 22].

3.  **Critical First Cell: Library Installation & Hugging Face Login:**
    This will be the **very first code cell** you run in your new notebook[cite: 23]. Paste and run:

    ```python
    # Install/Upgrade libraries
    !pip uninstall -y sympy
    !pip install --upgrade sympy
    !pip install --upgrade pip
    !pip install transformers[sentencepiece] datasets peft accelerate bitsandbytes torch evaluate rouge_score nltk

    # Login to Hugging Face Hub using the secret token
    from google.colab import userdata
    from huggingface_hub import login

    try:
        hf_token = userdata.get('HF_TOKEN')
        login(token=hf_token)
        print("Successfully logged into Hugging Face Hub!")
    except userdata.SecretNotFoundError:
        print("HF_TOKEN secret not found. Please add it to Colab secrets.")
    except Exception as e:
        print(f"An error occurred during Hugging Face login: {e}")

    print("\n--- Library installation/upgrade and HF login attempt complete. ---")
    print("IMPORTANT: Please RESTART THE RUNTIME now before running any other cells.")
    print("Go to 'Runtime' -> 'Restart session' (or 'Restart runtime') in the Colab menu.")
    ```

    **VERY IMPORTANT:** After this cell finishes executing, you **MUST restart the Colab runtime** for the changes to take full effect[cite: 25]. Go to the Colab menu: `"Runtime"` -\> `"Restart session"` (or `"Restart runtime"`). Click "Yes" if prompted[cite: 26].

#### Phase 2: Preparing Your Pub Review Data for Flan-T5 in Colab

1.  **Upload Your `pub_reviews.txt` (with 100 reviews):**
    After the runtime has restarted, upload your `pub_reviews.txt` file (the one with 100 examples from `pub_reviews_sentiment_data_v1`) to Colab using the file browser on the left[cite: 27].

2.  **Create and Run Preprocessing Logic for T5:**
    Create a new code cell in your Colab notebook[cite: 28]. You will use the script from artifact `preprocess_data_t5_with_call`. This script takes your `pub_reviews.txt` and creates `train_t5.jsonl`[cite: 29]. Paste the content of `preprocess_data_t5_with_call` into this cell and run it[cite: 30].
    Verify that `train_t5.jsonl` is created and contains \~100 records in the format: `{"input_text": "What is the sentiment of this review: [Review Text]", "target_text": "[Sentiment Label]"}`[cite: 31].

#### Phase 3: Fine-tuning Flan-T5-Small in Colab

1.  **Create and Run Fine-tuning Logic for T5:**
    Create a new code cell[cite: 31]. You will use the training script from artifact `train_flan_t5_lora_colab_fixed` (this is the version that fixed the `TrainingArguments` issue and was used for the successful training run)[cite: 32]. Paste its content into this cell and run it[cite: 33].
    Monitor the output: Look for successful loading of `train_t5.jsonl` (\~100 records)[cite: 33]. Crucially, watch the training loss. It should start relatively high and show a consistent downward trend over the epochs[cite: 34]. (Your successful run showed it decreasing from \~30s down significantly)[cite: 35]. The script will save the adapter to `./pirate_flan_t5_lora_adapter_attempt2` (or the folder name specified in that script)[cite: 36].

#### Phase 4: Testing Your Fine-tuned Flan-T5 in Colab

1.  **Create and Run Colab Testing Script for T5:**
    Once Phase 3 completes successfully (loss decreased, adapter saved), create a new code cell[cite: 37]. You will use the testing script from artifact `test_flan_t5_colab` (the version with the corrected `test_sentiment_model` function definition)[cite: 38].
    **Important:** Before running, ensure the `adapter_path_colab_test` variable inside this testing script correctly points to the output folder from your successful training run (e.g., `./pirate_flan_t5_lora_adapter_attempt2`)[cite: 39]. Run this cell[cite: 40].
    Examine the "Predicted Sentiment" outputs. They should now be accurate ("Positive", "Negative", "Neutral")[cite: 40].

#### Phase 5: Downloading Your Trained Adapter from Colab

1.  **Zip and Download:**
    If Colab testing (Phase 4) shows good results, create a new code cell and run (adjust `adapter_directory` if your training script saved it to a different name):

    ```python
    import os
    adapter_directory = "./pirate_flan_t5_lora_adapter_attempt2" # Or your actual adapter folder
    zip_filename = "pirate_flan_t5_lora_adapter.zip" # Generic zip name

    if os.path.exists(adapter_directory):
        print(f"Zipping directory: {adapter_directory}")
        !zip -r {zip_filename} {adapter_directory}

        from google.colab import files
        files.download(zip_filename)
    else:
        print(f"ERROR: Directory {adapter_directory} not found.")
    ```

    Download `pirate_flan_t5_lora_adapter.zip` to your local computer[cite: 41].

#### Phase 6: Deploying to Your Hugging Face Space

1.  **Prepare Hugging Face Space Files:**

      * `requirements.txt`: In your Space ("Files and versions" tab), ensure this file exists and contains:
        ```
        transformers
        datasets
        peft
        accelerate
        bitsandbytes
        torch
        gradio
        sentencepiece
        ```
        Commit changes if you edit it[cite: 42].
      * `app.py`: You will use the script from artifact `app_py_flan_t5_hf_space` (the one that fixed the device argument in the pipeline)[cite: 43]. In your Space, edit `app.py`. Delete any existing content[cite: 44]. Paste the content of `app_py_flan_t5_hf_space` into it[cite: 44]. Crucially, ensure the `ADAPTER_FOLDER_NAME` variable in this `app.py` script matches the name of the folder you will upload (e.g., `pirate_flan_t5_lora_adapter_attempt2`)[cite: 45]. Commit changes[cite: 46].
      * Clean the Space: Delete any old adapter folders from previous attempts in your Space[cite: 46].

2.  **Upload Fine-tuned Adapter to Space:**
    On your local computer, **unzip** the `pirate_flan_t5_lora_adapter.zip` file[cite: 47]. This will give you the adapter folder (e.g., `pirate_flan_t5_lora_adapter_attempt2`)[cite: 48]. In your Hugging Face Space ("Files and versions" tab), click "Add file" -\> "Upload folder"[cite: 48]. Select and upload the entire unzipped adapter folder to the **root** of your Space[cite: 49].

3.  **Restart and Test Space:**
    Go to the "Settings" tab of your Space (or the "..." menu) and "Restart this Space"[cite: 50]. Monitor the "Logs" tab. Look for messages confirming the base model and your LoRA adapter are loaded successfully[cite: 51]. Go to the "App" tab. Your "Pub Review Sentiment Analyzer" should be live\! Test it with various reviews[cite: 52].

This comprehensive plan should guide you through creating a successful fine-tuning demo for your workshop\! [cite: 53]

Let me know if you have any other questions\!
