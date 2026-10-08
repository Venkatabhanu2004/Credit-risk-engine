import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from credit_risk.logger import logger

def prepare_features(df, target_col, drop_cols=None):
    logger.info("Splitting features and target")
    
    if drop_cols is None:
        drop_cols = []
    
    # Drop target and unnecessary ID columns
    cols_to_drop = [target_col] + drop_cols
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    logger.info("Encoding categorical variables")
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def train_and_evaluate(X, y):
    logger.info("Train-test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    logger.info("Training Random Forest model")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    logger.info("Making predictions and evaluating")
    y_pred = model.predict(X_test)

    print("\n==============================")
    print("CLASSIFICATION REPORT")
    print("==============================\n")
    print(classification_report(y_test, y_pred))

    return model, X_train.columns.tolist()

def save_model(model, feature_names, path="artifacts/models/loan_rf_model.joblib"):
    logger.info(f"Saving model artifact to: {path}")
    # Saving both model and feature column order for prediction consistency
    artifact = {
        "model": model,
        "features": feature_names
    }
    joblib.dump(artifact, path)
    print(f"\nModel saved successfully at: {path}")