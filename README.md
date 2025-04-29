# credit_card_fraud_detection_model
# 💳 Credit Card Fraud Detection

## Objective

The objective of this project is to develop a robust machine learning pipeline capable of detecting fraudulent credit card transactions in highly imbalanced datasets. By leveraging techniques like feature engineering, class balancing, and ensemble learning (Random Forest), the project aims to uncover subtle patterns that distinguish fraudulent behavior from legitimate activity, ensuring early detection, minimizing financial losses, and strengthening digital transaction security.

---

## Project Structure

```bash
credit-card-fraud-detection/
├── data/                          # Raw dataset (fraudTest.csv, fraudTrain.csv)
├── notebooks/                     # Jupyter Notebooks for EDA and model building
│   ├── fraud_detection.ipynb
│   └── model_building.ipynb
├── src/                           # Source code for data processing and modeling
│   ├── preprocessing.py           # Data cleaning and feature engineering
│   ├── train_model.py             # Training pipeline script
│   └── predict.py                 # Model loading and prediction
├── outputs/                       # Trained models, evaluation metrics, plots
│   ├── random_forest_fraud_model.pkl
│   └── classification_report.txt
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview and instructions
```


## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
### 2. Install Dependencies
Ensure Python 3.8+ is installed, then run:

```bash
pip install -r requirements.txt
```
### 3. Explore the Data
Launch Jupyter Notebook and open the EDA notebook:

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```
### 4. Train the Model
You can train the Random Forest model by running:

```bash
python src/train_model.py
```
### 5. Evaluate the Model
Check model performance results:

```bash
outputs/classification_report.txt
```
### 6. Make Predictions
Use the trained model to make new predictions:
```bash
python src/predict.py
```
## Techniques Used
- Feature Engineering (e.g., extracting hour, age, weekday/month)

- Categorical Encoding (Label/One-Hot Encoding)

- Handling Class Imbalance

- Random Forest Classification

- Evaluation Metrics:

- Precision

- Recall

- F1-Score

- Confusion Matrix

## Dataset
### Dataset: Synthetic Credit Card Fraud Dataset (available on[ Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection))

### Files: Make sure the following are placed in the /data directory:

- fraudTrain.csv

- fraudTest.csv

