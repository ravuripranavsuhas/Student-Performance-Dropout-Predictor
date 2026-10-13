import io
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CONFIGURATION
# ============================================================
APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "student_history.db"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
TARGET_COLUMN = "Target"

st.set_page_config(
    page_title="Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25);
    padding: 14px;
    border-radius: 12px;
}
div[data-testid="stAlert"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET AND MODEL TRAINING
# ============================================================
@st.cache_data(show_spinner=False)
def load_dataset():
    """Download the UCI dataset and return the main CSV."""
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_names = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]
        if not csv_names:
            raise ValueError("No CSV file was found inside the UCI ZIP.")
        # The UCI package contains data.csv.
        preferred = next(
            (name for name in csv_names if Path(name).name.lower() == "data.csv"),
            csv_names[0]
        )
        with archive.open(preferred) as file:
            # This dataset uses semicolon separators.
            df = pd.read_csv(file, sep=";")

    df.columns = [str(c).strip() for c in df.columns]
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}', but found: "
            + ", ".join(df.columns[-5:])
        )
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    return df


@st.cache_resource(show_spinner=False)
def train_models():
    df = load_dataset()
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y_original = df[TARGET_COLUMN].astype(str).str.strip()

    # Genuine dropout target: Dropout=1; Graduate or Enrolled=0.
    y = (y_original.str.lower() == "dropout").astype(int)

    # Dataset features are numeric-coded, including categorical fields.
    # Coerce all feature columns to numeric and impute any missing values.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), feature_names)
        ],
        remainder="drop"
    )

    algorithms = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
    }

    models = {}
    metric_rows = []
    test_predictions = {}

    for name, algorithm in algorithms.items():
        model = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", algorithm)
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        models[name] = model
        test_predictions[name] = pred
        metric_rows.append({
            "Model": name,
            "Accuracy (%)": round(accuracy_score(y_test, pred) * 100, 2),
            "Precision (%)": round(
                precision_score(y_test, pred, zero_division=0) * 100, 2
            ),
            "Recall (%)": round(
                recall_score(y_test, pred, zero_division=0) * 100, 2
            ),
            "F1 Score (%)": round(
                f1_score(y_test, pred, zero_division=0) * 100, 2
            ),
        })

    return (
        df, feature_names, models, pd.DataFrame(metric_rows),
        y_test, test_predictions, y_original
    )


# ============================================================
# LOCAL SQLITE HISTORY
# ============================================================
def init_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                student_id TEXT,
                student_name TEXT,
                model TEXT NOT NULL,
                risk_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                prediction TEXT NOT NULL,
                input_json TEXT NOT NULL
            )
        """)


def save_prediction(record, input_values):
    import json
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                created_at, student_id, student_name, model,
                risk_probability, risk_level, prediction, input_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["created_at"], record["student_id"], record["student_name"],
            record["model"], record["risk_probability"], record["risk_level"],
            record["prediction"], json.dumps(input_values)
        ))


def load_history():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT id, created_at, student_id, student_name, model, "
            "risk_probability, risk_level, prediction "
            "FROM predictions ORDER BY id DESC", conn
        )


def classify_risk(probability):
    # Adjustable demonstration bands, not validated intervention thresholds.
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


def create_pdf_report(record, input_values):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 55

    pdf.setTitle("Student Dropout Risk Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(45, y, "Student Dropout Risk Report")
    y -= 32
    pdf.setFont("Helvetica", 10)

    lines = [
        f"Generated: {record['created_at']}",
        f"Student ID: {record['student_id'] or 'Not provided'}",
        f"Student name: {record['student_name'] or 'Not provided'}",
        f"Model: {record['model']}",
        f"Prediction: {record['prediction']}",
        f"Risk band: {record['risk_level']}",
        f"Estimated dropout probability: {record['risk_probability']:.2f}%",
        "",
        "Input features:"
    ]
    for line in lines:
        pdf.drawString(45, y, str(line)[:110])
        y -= 18

    for key, value in input_values.items():
        if y < 80:
            pdf.showPage()
            y = height - 55
            pdf.setFont("Helvetica", 10)
        pdf.drawString(55, y, f"{key}: {value}"[:110])
        y -= 15

    y -= 10
    if y < 75:
        pdf.showPage()
        y = height - 55
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(45, y, "Important limitations")
    y -= 16
    pdf.setFont("Helvetica", 9)
    notes = [
        "This is a machine-learning estimate, not a guaranteed outcome.",
        "The model is trained on a public UCI dataset and may not generalize",
        "to your institution. Use it only to support human review and student support."
    ]
    for note in notes:
        pdf.drawString(45, y, note)
        y -= 13

    pdf.save()
    return buffer.getvalue()


# ============================================================
# LOAD APP
# ============================================================
init_database()
st.title("🎓 Student Dropout Prediction System")
st.caption("Machine learning • Student analytics • Academic support")

try:
    with st.spinner("Downloading the UCI dataset and training models..."):
        (
            df, feature_names, models, metrics, y_test,
            test_predictions, original_targets
        ) = train_models()
except Exception as exc:
    st.error(
        "The app could not load the UCI dataset. Check your internet "
        "connection, then restart Streamlit."
    )
    st.code(str(exc))
    st.info(
        "The first launch requires internet access to download the public dataset."
    )
    st.stop()

# Helpful descriptions for the feature inputs.
FEATURE_HELP = {
    "Marital status": "Numeric code used by the UCI dataset.",
    "Application mode": "Numeric code for the admission application route.",
    "Application order": "Order of application preference.",
    "Course": "Numeric code for the enrolled course.",
    "Daytime/evening attendance": "1=daytime, 0=evening in the dataset.",
    "Previous qualification": "Numeric code for prior qualification.",
    "Previous qualification (grade)": "Grade in the prior qualification.",
    "Nacionality": "Numeric code for nationality in the dataset.",
    "Mother's qualification": "Numeric code for mother's qualification.",
    "Father's qualification": "Numeric code for father's qualification.",
    "Mother's occupation": "Numeric code for mother's occupation.",
    "Father's occupation": "Numeric code for father's occupation.",
    "Admission grade": "Admission grade recorded by the institution.",
    "Displaced": "Whether the student is displaced (dataset code).",
    "Educational special needs": "Dataset indicator for special educational needs.",
    "Debtor": "Whether the student is recorded as a debtor.",
    "Tuition fees up to date": "Whether tuition fees are up to date.",
    "Gender": "Dataset gender code.",
    "Scholarship holder": "Whether the student holds a scholarship.",
    "Age at enrollment": "Age when enrolling.",
    "International": "Whether the student is international.",
    "Curricular units 1st sem (credited)": "First-semester credited units.",
    "Curricular units 1st sem (enrolled)": "First-semester enrolled units.",
    "Curricular units 1st sem (evaluations)": "First-semester evaluations.",
    "Curricular units 1st sem (approved)": "First-semester approved units.",
    "Curricular units 1st sem (grade)": "First-semester average grade.",
    "Curricular units 1st sem (without evaluations)": "First-semester units without evaluations.",
    "Curricular units 2nd sem (credited)": "Second-semester credited units.",
    "Curricular units 2nd sem (enrolled)": "Second-semester enrolled units.",
    "Curricular units 2nd sem (evaluations)": "Second-semester evaluations.",
    "Curricular units 2nd sem (approved)": "Second-semester approved units.",
    "Curricular units 2nd sem (grade)": "Second-semester average grade.",
    "Curricular units 2nd sem (without evaluations)": "Second-semester units without evaluations.",
    "Unemployment rate": "Macroeconomic unemployment rate in the dataset.",
    "Inflation rate": "Macroeconomic inflation rate in the dataset.",
    "GDP": "Macroeconomic GDP value in the dataset."
}


def render_input_form():
    st.write(
        "Enter the features using the numeric coding and definitions in the "
        "UCI dataset. Defaults are based on median values from the dataset."
    )
    with st.form("student_form"):
        student_id = st.text_input("Student ID", placeholder="e.g. STU001")
        student_name = st.text_input("Student name", placeholder="e.g. Student A")
        values = {}
        medians = df[feature_names].apply(pd.to_numeric, errors="coerce").median()
        st.subheader("Admission and background")
        first_group = feature_names[:20]
        cols = st.columns(2)
        for i, feature in enumerate(first_group):
            with cols[i % 2]:
                default = float(medians.get(feature, 0))
                if not np.isfinite(default):
                    default = 0.0
                values[feature] = st.number_input(
                    feature,
                    value=float(default),
                    help=FEATURE_HELP.get(feature, "Use the dataset's numeric coding."),
                    key=f"single_{feature}"
                )
        st.subheader("Semester performance and economic indicators")
        rest = feature_names[20:]
        cols = st.columns(2)
        for i, feature in enumerate(rest):
            with cols[i % 2]:
                default = float(medians.get(feature, 0))
                if not np.isfinite(default):
                    default = 0.0
                values[feature] = st.number_input(
                    feature,
                    value=float(default),
                    help=FEATURE_HELP.get(feature, "Use the dataset's numeric coding."),
                    key=f"single_{feature}"
                )
        chosen_model = st.selectbox(
            "Machine learning model", list(models.keys()), index=2
        )
        submitted = st.form_submit_button(
            "Predict Dropout Risk", type="primary", use_container_width=True
        )
    return submitted, student_id, student_name, values, chosen_model


page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Predict Student", "Bulk Prediction", "Prediction History", "Model Insights"]
)
selected_model = st.sidebar.selectbox(
    "Model for charts and batch predictions",
    list(models.keys()), index=2
)
st.sidebar.divider()
st.sidebar.write("**Dataset**")
st.sidebar.write(f"{len(df):,} records")
st.sidebar.write(f"{len(feature_names)} input features")
st.sidebar.caption("Target: Dropout vs Graduate/Enrolled")


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.header("Dashboard Overview")
    labels = original_targets.astype(str).str.strip()
    dropout_count = int((labels.str.lower() == "dropout").sum())
    graduate_count = int((labels.str.lower() == "graduate").sum())
    enrolled_count = int((labels.str.lower() == "enrolled").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Dropout records", f"{dropout_count:,}")
    c3.metric("Graduate records", f"{graduate_count:,}")
    c4.metric("Enrolled records", f"{enrolled_count:,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Original dataset outcomes")
        counts = labels.value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_ylabel("Number of records")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Model performance")
        st.dataframe(metrics, hide_index=True, use_container_width=True)
        st.caption("Metrics are from one stratified 20% hold-out test split.")

    st.info(
        "The prediction target is genuine dropout status from the UCI dataset. "
        "Graduate and Enrolled are grouped as not-dropout for this binary model."
    )


# ============================================================
# INDIVIDUAL PREDICTION
# ============================================================
elif page == "Predict Student":
    st.header("Individual Student Prediction")
    submitted, student_id, student_name, values, chosen_model = render_input_form()

    if submitted:
        student_frame = pd.DataFrame([values], columns=feature_names)
        model = models[chosen_model]
        prediction = int(model.predict(student_frame)[0])
        probability = float(model.predict_proba(student_frame)[0][1]) * 100
        risk_level = classify_risk(probability / 100)
        label = "Predicted Dropout Risk" if prediction == 1 else "Predicted Not Dropout"

        left, mid, right = st.columns(3)
        left.metric("Prediction", label)
        mid.metric("Estimated dropout probability", f"{probability:.1f}%")
        right.metric("Risk band", risk_level)

        if risk_level == "High":
            st.error("High estimated risk. Review the student's situation and support needs.")
        elif risk_level == "Medium":
            st.warning("Medium estimated risk. Consider monitoring academic progress.")
        else:
            st.success("Low estimated risk according to this model.")

        record = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_id": student_id.strip(),
            "student_name": student_name.strip(),
            "model": chosen_model,
            "risk_probability": round(probability, 2),
            "risk_level": risk_level,
            "prediction": label
        }
        save_prediction(record, values)
        st.success("Prediction saved in the local history database.")

        pdf = create_pdf_report(record, values)
        safe_id = "".join(ch for ch in student_id if ch.isalnum() or ch in "-_") or "student"
        st.download_button(
            "Download PDF report",
            data=pdf,
            file_name=f"dropout_report_{safe_id}.pdf",
            mime="application/pdf"
        )


# ============================================================
# BULK PREDICTION
# ============================================================
elif page == "Bulk Prediction":
    st.header("Bulk Student Prediction")
    st.write(
        "Upload a CSV containing the same feature columns used by the UCI "
        "dataset. You may also include optional Student_ID and Student_Name columns."
    )

    template = df[feature_names].head(1).copy()
    template.insert(0, "Student_Name", "Sample Student")
    template.insert(0, "Student_ID", "STU001")
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="student_input_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload student CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            optional = ["Student_ID", "Student_Name"]
            missing = [c for c in feature_names if c not in batch.columns]
            if missing:
                st.error(
                    "CSV is missing required columns: "
                    + ", ".join(missing[:12])
                    + (" ..." if len(missing) > 12 else "")
                )
            elif batch.empty:
                st.warning("The CSV contains no student records.")
            else:
                X_batch = batch[feature_names].copy()
                for col in feature_names:
                    X_batch[col] = pd.to_numeric(X_batch[col], errors="coerce")
                if X_batch.isna().all(axis=1).any():
                    st.warning(
                        "At least one row has no usable numeric features. "
                        "Check the uploaded CSV."
                    )
                pred_model = models[selected_model]
                pred = pred_model.predict(X_batch)
                probs = pred_model.predict_proba(X_batch)[:, 1] * 100
                output = batch.copy()
                output["Predicted outcome"] = np.where(
                    pred == 1, "Dropout risk", "Not predicted dropout"
                )
                output["Dropout probability (%)"] = np.round(probs, 2)
                output["Risk band"] = [
                    classify_risk(p / 100) for p in probs
                ]

                st.success(f"Predictions generated for {len(output)} rows.")
                st.dataframe(output, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download predictions CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="student_dropout_predictions.csv",
                    mime="text/csv"
                )
        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")


# ============================================================
# PREDICTION HISTORY
# ============================================================
elif page == "Prediction History":
    st.header("Prediction History")
    history = load_history()
    if history.empty:
        st.info("No saved predictions yet. Make a prediction first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            search_id = st.text_input("Search Student ID")
        with c2:
            risk_filter = st.selectbox("Risk band", ["All", "Low", "Medium", "High"])
        filtered = history.copy()
        if search_id.strip():
            filtered = filtered[
                filtered["student_id"].fillna("").str.contains(
                    search_id.strip(), case=False, regex=False
                )
            ]
        if risk_filter != "All":
            filtered = filtered[filtered["risk_level"] == risk_filter]
        st.metric("Matching records", len(filtered))
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "Download history CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="student_prediction_history.csv",
            mime="text/csv"
        )
        st.caption(
            f"Database file: {DB_PATH.name}. Keep it private because it contains student records."
        )


# ============================================================
# MODEL INSIGHTS
# ============================================================
elif page == "Model Insights":
    st.header("Model Insights and Evaluation")
    st.subheader("Compare algorithms")
    st.dataframe(metrics, hide_index=True, use_container_width=True)

    metric_choice = st.selectbox(
        "Metric to visualize",
        ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(metrics["Model"], metrics[metric_choice])
    ax.set_ylabel(metric_choice)
    ax.set_title("Hold-out test performance")
    ax.tick_params(axis="x", rotation=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, test_predictions[selected_model], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Not dropout", "Dropout"])
    ax.set_yticks([0, 1], labels=["Not dropout", "Dropout"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(selected_model)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Most influential features")
    model = models[selected_model]
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        scores = classifier.feature_importances_
        method_note = "Tree feature importance reflects how the fitted trees use features."
    else:
        scores = np.abs(classifier.coef_[0])
        method_note = (
            "For Logistic Regression, absolute standardized coefficient magnitudes "
            "are shown as a rough influence measure; they are not causal effects."
        )

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": scores
    }).sort_values("Importance", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    ax.set_xlabel("Relative importance")
    ax.set_title("Top features")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(method_note)

    st.subheader("Classification report")
    report = classification_report(
        y_test, test_predictions[selected_model],
        labels=[0, 1],
        target_names=["Not dropout", "Dropout"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.divider()
st.caption(
    "Educational prototype using the UCI Predict Students' Dropout and Academic "
    "Success dataset. Model output should support human review, not determine a "
    "student's future or access to education."
)
