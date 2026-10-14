
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_fraud_detection_model.pkl"
DATASET_PATH = BASE_DIR / "data" / "creditcard.csv"


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()


# --------------------------------------------------
# 3. Application introduction
# --------------------------------------------------
st.title("🔍 Machine Learning Fraud Detection System")
st.subheader("Project Overview")

st.write("""
This project uses supervised machine learning to detect
fraudulent credit card transactions.

The model analyzes transaction features and predicts
whether a transaction is legitimate or fraudulent.

The application displays fraud predictions, model
performance metrics, and downloadable results.
""")
st.write(
    "This application uses a trained Random Forest model "
    "to identify potentially fraudulent credit card transactions."
)

st.info(
    "Upload a CSV containing the same feature columns used "
    "during training. Do not include the Class target column."
)


# --------------------------------------------------
# 4. Upload transaction data
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.warning("The uploaded CSV contains no transactions.")
        else:
            st.subheader("Uploaded Transaction Data")
            st.dataframe(data.head(10), use_container_width=True)

            # Check required feature columns when available
            expected_columns = getattr(
                model, "feature_names_in_", None
            )

            if expected_columns is not None:
                missing_columns = [
                    col for col in expected_columns
                    if col not in data.columns
                ]
                extra_columns = [
                    col for col in data.columns
                    if col not in expected_columns
                ]

                if missing_columns:
                    st.error(
                        "The uploaded CSV is missing required columns: "
                        + ", ".join(missing_columns)
                    )
                    st.stop()

                if extra_columns:
                    st.warning(
                        "Extra columns will be ignored: "
                        + ", ".join(extra_columns)
                    )

                # Ensure features are supplied in training order
                data = data[list(expected_columns)]

            if st.button("Detect Fraud", type="primary"):
                predictions = model.predict(data)

                # ------------------------------------------
                # 5. Fraud detection summary
                # ------------------------------------------
                total_transactions = len(predictions)
                fraud_count = int((predictions == 1).sum())
                legitimate_count = int((predictions == 0).sum())

                fraud_percentage = (
                    fraud_count / total_transactions * 100
                    if total_transactions else 0
                )

                st.subheader("Fraud Detection Summary")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Transactions", total_transactions
                )
                col2.metric(
                    "Legitimate Transactions", legitimate_count
                )
                col3.metric(
                    "Fraudulent Transactions", fraud_count
                )
                col4.metric(
                    "Predicted Fraud Rate",
                    f"{fraud_percentage:.2f}%",
                )

                # ------------------------------------------
                # 6. Transaction distribution chart
                # ------------------------------------------
                st.subheader("Transaction Distribution")

                chart_data = pd.DataFrame(
                    {
                        "Transaction Type": [
                            "Legitimate", "Fraudulent"
                        ],
                        "Count": [
                            legitimate_count, fraud_count
                        ],
                    }
                )

                st.bar_chart(
                    chart_data,
                    x="Transaction Type",
                    y="Count",
                )

                # ------------------------------------------
                # 7. Prepare prediction results
                # ------------------------------------------
                output = data.copy()

                output["Prediction"] = [
                    "Fraudulent" if p == 1 else "Legitimate"
                    for p in predictions
                ]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(data)[:, 1]
                    output["Fraud Probability (%)"] = (
                        probabilities * 100
                    ).round(2)

                # ------------------------------------------
                # 8. Display results and download
                # ------------------------------------------
                st.subheader("Fraud Detection Results")
                st.dataframe(output, use_container_width=True)

                csv_data = output.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Results CSV",
                    data=csv_data,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Check the uploaded file's column names, feature types, "
            "and compatibility with the trained model."
        )


# --------------------------------------------------
# 9. Model performance evaluation
# --------------------------------------------------
st.divider()
st.header("Model Performance")

st.write(
    "These metrics evaluate the saved model on labelled test data. "
    "They are calculated independently of any uploaded CSV."
)

try:
    if not DATASET_PATH.exists():
        st.warning(
            "The original labelled dataset was not found at "
            f"{DATASET_PATH}. Model performance cannot be calculated."
        )
    else:
        df = pd.read_csv(DATASET_PATH)

        if "Class" not in df.columns:
            st.error("The original dataset must contain a 'Class' column.")
        else:
            X = df.drop(columns=["Class"])
            y = df["Class"]

            # Recreate the test split used in the project
            _, X_test, _, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            y_pred = model.predict(X_test)

            # Calculate metrics for the fraud class (Class = 1)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, pos_label=1, zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, pos_label=1, zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, pos_label=1, zero_division=0
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
            col2.metric("Fraud Precision", f"{precision * 100:.2f}%")
            col3.metric("Fraud Recall", f"{recall * 100:.2f}%")
            col4.metric("Fraud F1-Score", f"{f1 * 100:.2f}%")

            # ------------------------------------------
            # 10. Confusion matrix
            # ------------------------------------------
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1],
            )

            st.write(
                pd.DataFrame(
                    cm,
                    index=["Actual Legitimate", "Actual Fraud"],
                    columns=[
                        "Predicted Legitimate",
                        "Predicted Fraud",
                    ],
                )
            )

            st.caption(
                "Class 0 = legitimate transaction; Class 1 = fraud. "
                "Evaluation is valid only if these test rows were not "
                "used to train the saved model."
            )

except Exception as e:
    st.error(f"Could not calculate model performance: {e}")
