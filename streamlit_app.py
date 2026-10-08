import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Batch Loan Default Scoring", layout="wide")
st.title("💳 Batch Credit Risk Prediction")
st.write("Upload a CSV file containing applicant records to score them in bulk.")

# 1. Load Model Artifact
artifact = joblib.load("artifacts/models/loan_rf_model.joblib")
model = artifact["model"]
model_features = artifact["features"]

# Optional: Add a risk threshold slider in sidebar
st.sidebar.header("Risk Settings")
threshold = st.sidebar.slider("Default Probability Cutoff", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

# 2. File Uploader
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.subheader(f"Uploaded Data Preview ({test_df.shape[0]} rows, {test_df.shape[1]} columns)")
    st.dataframe(test_df.head(), use_container_width=True)

    if st.button("🚀 Run Batch Prediction", type="primary"):
        # Exclude Target, ID, or prior prediction columns if re-uploading
        ignore_cols = ["Loan_Default", "Customer_ID", "Default_Probability", "Predicted_Status"]
        features_df = test_df.drop(columns=[c for c in ignore_cols if c in test_df.columns])

        # Handle any null values present in test set
        for col in features_df.columns:
            if pd.api.types.is_numeric_dtype(features_df[col]):
                features_df[col] = features_df[col].fillna(features_df[col].median())
            else:
                mode_val = features_df[col].mode()
                features_df[col] = features_df[col].fillna(mode_val[0] if not mode_val.empty else "Missing")

        # Encode and align columns with training features
        encoded_df = pd.get_dummies(features_df)
        encoded_df = encoded_df.reindex(columns=model_features, fill_value=0)

        # Predict probability for the Default class (index 1)
        probabilities = model.predict_proba(encoded_df)[:, 1]

        # Attach results using the probability threshold
        result_df = test_df.copy()
        result_df["Default_Probability"] = probabilities.round(4)
        result_df["Predicted_Status"] = [
            "High Risk (Default)" if p >= threshold else "Approved" 
            for p in probabilities
        ]

        st.success("Batch Prediction Complete!")

        # Summary KPIs
        total = len(result_df)
        defaults = (result_df["Predicted_Status"] == "High Risk (Default)").sum()
        approved = total - defaults

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Applicants", total)
        col2.metric("Approved", approved)
        col3.metric("High Risk / Rejected", defaults)

        # Scored table preview
        st.subheader("Scored Results Preview")
        cols_to_show = ["Predicted_Status", "Default_Probability"] + [
            c for c in result_df.columns if c not in ["Predicted_Status", "Default_Probability"]
        ]
        st.dataframe(result_df[cols_to_show], use_container_width=True)

        # Download button
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Scored Predictions CSV",
            data=csv_data,
            file_name="loan_predictions_scored.csv",
            mime="text/csv",
            use_container_width=True
        )