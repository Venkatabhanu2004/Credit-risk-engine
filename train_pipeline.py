import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from credit_risk.logger import logger
from credit_risk.data_ingestion import load_data, clean_data
from credit_risk.data_eda import run_all_eda
from credit_risk.model_trainer import prepare_features, train_and_evaluate, save_model

def run_pipeline():
    logger.info("==========================================")
    logger.info("Loan Pipeline Started")
    logger.info("==========================================")

    # 1. Load Data
    data_path = "artifacts/data/raw/loan_guard_dataset.csv"
    df = load_data(data_path)

    # 2. Clean Data (Duplicates & Missing Values)
    df = clean_data(df)

    # 3. Target Definition
    target = "Loan_Default"

    # 4. Run Complete EDA (Runs and saves all 4 charts automatically)
    run_all_eda(df, target_col=target)

    # 5. Feature Engineering & Split
    X, y = prepare_features(df, target_col=target, drop_cols=["Customer_ID", "Customer_I"])

    # 6. Model Training & Evaluation
    model, feature_names = train_and_evaluate(X, y)

    # 7. Save Model Artifact
    save_model(model, feature_names, "artifacts/models/loan_rf_model.joblib")

    logger.info("Pipeline completed successfully")
    print("\nTraining and artifact generation complete.")

if __name__ == "__main__":
    run_pipeline()