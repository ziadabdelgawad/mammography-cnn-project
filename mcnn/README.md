Mammography CNN Project – Starting Steps

Step 1: Download the Dataset

Open this link:

[https://huggingface.co/datasets/dpetrini/cbis_ddsm_rev](https://huggingface.co/datasets/dpetrini/cbis_ddsm_rev)

Download cbis_ddsm_rev.tar.bz2 and extract it.

Find the inner cbis_ddsm_rev folder. It must contain:

train
val
test
readme.txt

Step 2: Install Python

Install Python 3.12 on your computer.

Step 3: Open PowerShell

Open the mammography_cnn_project folder.

Click the folder address bar, type powershell, and press Enter.

Step 4: Set Up the Project

Run:

.\SETUP_WINDOWS.bat

Wait until the setup finishes.

Step 5: Prepare the Dataset

Run:

.\STEP1_PREPARE_DATASET.bat

Paste the complete path to the inner cbis_ddsm_rev folder.

Example:

C:\Users\YOUR_NAME\Downloads\cbis_ddsm_rev\cbis_ddsm_rev

Wait until this message appears:

STEP 1 COMPLETE. All patient-overlap counts are 0.

Do not move or delete the dataset folder.

Step 6: Train the Baseline Model

Run:

.\TRAIN_BASELINE.bat

Wait until it finishes.

Step 7: Train the Regularised Model

Run:

.\TRAIN_REGULARIZED.bat

Wait until it finishes.

Step 8: Train the Transfer-Learning Model

Run:

.\TRAIN_TRANSFER.bat

Wait until it finishes.

Step 9: Evaluate All Models

Run:

.\EVALUATE_ALL.bat

Wait until the evaluation finishes.

Step 10: Select the Best Model

Run:

.\SELECT_BEST_MODEL.bat

The selected model will be saved as:

models\best_model.keras

Step 11: Start the Website

Run:

.\RUN_WEB.bat

Open this link in your browser:

[http://127.0.0.1:8000](http://127.0.0.1:8000)
