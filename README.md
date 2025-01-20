# Flight Delay Prediction Project

This repository provides source code, data preprocessing scripts, and machine learning pipelines for predicting US domestic flight delays using the 2008 flights dataset. The project demonstrates how to perform data cleaning, feature engineering, and model training with **CatBoost**, emphasizing GPU acceleration, high accuracy, and robust evaluation.

---

## Table of Contents

1. [Overview](#overview)  
2. [Requirements](#requirements)  
3. [Installation and Setup](#installation-and-setup)  
4. [Project Structure](#project-structure)  
5. [Running the Project](#running-the-project)  
6. [Reproducing the Results](#reproducing-the-results)  


---

## Overview

Domestic flight delays are a significant concern for travelers and airlines alike. By leveraging Python’s powerful data manipulation and machine learning libraries such as pandas, numpy, matplotlib, scikit-learn and catboost, this project aims to classify flights as “delayed” (arrival delay > 15 minutes) or “on-time.” The data preprocessing pipeline involves filtering irrelevant or missing data, validating timestamps, and creating relevant features. The model uses **CatBoost** for its native handling of categorical data and GPU support.

Key highlights:
- Comprehensive data cleaning, noise handling, including invalid time removal.
- Feature engineering (e.g., route creation, average delay per route/carrier).
- Gradient boosting on GPU to handle large datasets efficiently.
- Evaluation using accuracy, F1-score, and ROC AUC.

---

## Requirements

1. **Operating System**: Windows, macOS, or Linux  
2. **Python**: Version 3.7+  
3. **VS Code**: Recommended for an integrated development environment ([Download VS Code](https://code.visualstudio.com/)).  
4. **GPU Drivers** (Optional but recommended for large datasets):  
   - NVIDIA GPU with CUDA drivers if you plan to use GPU training with CatBoost.
5. **Jupyter Notebook**

### Required Python Libraries

All dependencies are listed in `requirements.txt`:


---

## Installation and Setup

1. **Install Python**  
   - Visit the [Python Downloads page](https://www.python.org/downloads/) and follow the instructions for your operating system.
   - Make sure Python is added to your system’s PATH if you are on Windows (this is an option in the installer).

2. **Install VS Code**  
   - Go to the [Visual Studio Code website](https://code.visualstudio.com/) and download the installer for your OS.
   - During installation, you can select additional options like “Add to PATH” to make the `code` command available in your terminal.

3. **Clone or Download the Project**  
   - Clone this repository using Git:
     ```bash
     git clone https://github.com/Fabelouzz/Predicting-US-Flight-2008-Delays-Using-CatBoost
     ```
   - Or download the repository as a ZIP file from GitHub and extract it.

4. **Set Up a Virtual Environment (Recommended)**  
   - Create a virtual environment to isolate project dependencies:
     ```bash
     python -m venv env
     ```
   - Activate the environment:
     - **Windows**: `env\Scripts\activate`
     - **macOS/Linux**: `source env/bin/activate`

5. **Install Dependencies**  
   - Inside the project folder, install all required libraries:
     ```bash
     pip install -r requirements.txt
     ```
   - This will fetch and install **pandas**, **numpy**, **matplotlib**, **seaborn**, **scikit-learn**, **xgboost**, **imbalanced-learn**, **notebook**, **ipykernel**, and **catboost**.

---

## Project Structure

```plaintext
Predicting-US-Flight-2008-Delays-Using-CatBoost/
├── data/
│   └── 2008.csv               # Original dataset (downloaded separately)
├── results/
│   ├── feature_importance.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── f1_score_plot.png
│   ├── catboost_training_log.png
│   ├── final_metrics.png
├── catBoost.ipynb             # Main Jupyter Notebook for training & evaluation
├── outputs.txt                # Example logs or results saved during runs
├── requirements.txt           # List of required Python packages
├── specs.txt                  # Notes or project specifications
└── README.md                  # Project README file
```
## Running the Project

1. **Obtain the Dataset**  
   - Download the 2008 US domestic flights CSV file from [BTS](https://www.kaggle.com/datasets/artomas/us-flights/data) or an alternative source.
   - Place the file in the `Predicting-US-Flight-2008-Delays-Using-CatBoost` directory as `2008.csv`. If you name the file differently, update the code references accordingly.

2. **Set Up the Environment**  
   - Ensure you have Python 3.7+ installed.  
   - Optionally, create and activate a virtual environment:
     ```bash
     python -m venv env
     source env/bin/activate  # On Windows: env\Scripts\activate
     ```
   - Install the required libraries via:
     ```bash
     pip install -r requirements.txt
     ```

3. **Preprocessing**  
   - Open or run the preprocessing steps within `catBoost.ipynb` or a separate script (if provided). This will:
     1. Clean the dataset.
     2. Filter out invalid time entries.
     3. Create features such as Route, IsWeekday, and average delay columns.
     4. Split the dataset into train, validation, and test sets without data leakage.

4. **Training and Evaluation**  
   - Still in `catBoost.ipynb`, execute the cells that:
     1. Initialize the CatBoost model (with GPU if available).
     2. Train on the training set and validate using the validation set.
     3. Display final metrics on the test set.
     4. Generate results such as ROC curves, confusion matrices, and feature importance plots, which will be saved in the current folder.

5. **Check Log Outputs**  
   - Monitor the console or the notebook output for iterations, best scores, and final performance metrics. If the GPU is detected, logs will confirm GPU usage.  
   - Metrics such as accuracy, F1-score, and ROC AUC will be printed at the end of training.

---

## Reproducing the Results

1. **Verify Configuration**  
   - Confirm that your environment matches the project specifications. Ensure the dataset path, hyperparameters, and random seeds (if any) match those in `catBoost.ipynb`.

2. **Run the Jupyter Notebook**  
   - Launch Jupyter Notebook and open `catBoost.ipynb`:
     ```bash
     jupyter notebook catBoost.ipynb
     ```
   - Follow the notebook cells in sequence. The code will perform data preprocessing, splitting, training, and evaluation in the same manner it was originally executed.

3. **Inspect Generated Outputs**  
   - After training completes, you should see final metrics (accuracy, F1-score, ROC AUC) that closely match previously reported results.  
   - Plots such as `feature_importance.png`, `roc_curve.png`, `confusion_matrix.png`, and `f1_score_plot.png` will appear in the `results/` folder or inline, depending on the notebook settings.

4. **Review Changes**  
   - If your results vary significantly, confirm:
     1. The dataset is the same version used in the original run.
     2. No changes to hyperparameters or code logic were introduced.
     3. The random seed and environment details (e.g., GPU usage) align with the original setup.

By following these steps precisely, you should reproduce the near-perfect performance metrics presented in the project’s documentation, including the generation of images and logs stored in the `results/` and related output files.

---
Thank you for your time, enjoy the repository! 

// Fabian



