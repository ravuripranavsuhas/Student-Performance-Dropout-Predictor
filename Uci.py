import streamlit as st
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance & Dropout Predictor")
st.write(
    "Analyze student information, predict academic risk, "
    "compare ML models, and identify important factors."
)

st.info(
    "Academic risk is defined here as a final grade below 10/20. "
    "This is an academic-risk demonstration, not a true dropout prediction."
)

# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_data():
    dataset = fetch_ucirepo(id=320)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Use the mathematics dataset for this demonstration.
    # The UCI package provides student-mat.csv and student-por.csv.
    if "G3" not in y.columns:
        raise ValueError("Final grade G3 was not found.")

    data = X.copy()
    data["G3"] = y["G3"].values

    return data


try:
    data = load_data()
except Exception as e:
    st.error(
        "Could not download the UCI dataset. "
        "Check your internet connection and try again."
    )
    st.caption(f"Technical details: {e}")
    st.stop()

st.success(f"Dataset loaded successfully: {len(data)} student records")

# ---------------- SELECT FEATURES ----------------

features = [
    "age",
    "Medu",
    "Fedu",
    "studytime",
    "failures",
    "absences",
    "famsup",
    "schoolsup",
    "G1",
    "G2"
]

# Ensure all required columns exist
missing_columns = [c for c in features + ["G3"] if c not in data.columns]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

X = data[features].copy()

# Final grade below 10 is treated as academic risk
y = (data["G3"] < 10).astype(int)

# ---------------- PREPROCESS DATA ----------------

numeric_features = [
    "age",
    "Medu",
    "Fedu",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]

categorical_features = [
    "famsup",
    "schoolsup"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# ---------------- TRAIN AND TEST DATA ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------- MACHINE LEARNING MODELS ----------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42
    )
}

trained_models = {}
results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    trained_models[name] = pipeline

    results.append({
        "Model": name,
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "Precision (%)": round(
            precision_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "Recall (%)": round(
            recall_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "F1 Score (%)": round(
            f1_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        )
    })

results_df = pd.DataFrame(results)

# ---------------- DASHBOARD ----------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(data))
col2.metric("At-Risk Records", int(y.sum()))
col3.metric(
    "Not At-Risk Records",
    int((y == 0).sum())
)

with st.expander("View dataset sample"):
    st.dataframe(data.head(10), use_container_width=True)

# ---------------- MODEL COMPARISON ----------------

st.header("🤖 Model Performance Comparison")

st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True
)

best_model_name = results_df.sort_values(
    "F1 Score (%)", ascending=False
).iloc[0]["Model"]

st.write(f"**Highest F1 score on this test split:** {best_model_name}")

st.caption(
    "Metrics are calculated on a held-out test set. "
    "Results may vary with different splits and data."
)

# ---------------- STUDENT INPUT FORM ----------------

st.header("🧑‍🎓 Predict Academic Risk")

st.write("Enter the student's details below.")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Student age",
            min_value=15,
            max_value=22,
            value=17
        )

        mother_education = st.selectbox(
            "Mother's education level",
            options=[0, 1, 2, 3, 4],
            index=2,
            help="0 = none, 4 = higher education"
        )

        father_education = st.selectbox(
            "Father's education level",
            options=[0, 1, 2, 3, 4],
            index=2
        )

        studytime = st.selectbox(
            "Weekly study-time category",
            options=[1, 2, 3, 4],
            index=1,
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2 to 5 hours",
                3: "5 to 10 hours",
                4: "More than 10 hours"
            }[x]
        )

        failures = st.number_input(
            "Previous class failures",
            min_value=0,
            max_value=4,
            value=0
        )

    with col2:
        absences = st.number_input(
            "Number of absences",
            min_value=0,
            max_value=100,
            value=3
        )

        family_support = st.selectbox(
            "Family educational support",
            ["yes", "no"]
        )

        school_support = st.selectbox(
            "Extra school support",
            ["yes", "no"]
        )

        g1 = st.slider(
            "First-period grade (0–20)",
            0, 20, 12
        )

        g2 = st.slider(
            "Second-period grade (0–20)",
            0, 20, 12
        )

    selected_model = st.selectbox(
        "Choose machine learning algorithm",
        list(models.keys())
    )

    submitted = st.form_submit_button(
        "Predict Academic Risk",
        type="primary"
    )

# ---------------- PREDICTION ----------------

if submitted:

    student = pd.DataFrame([{
        "age": age,
        "Medu": mother_education,
        "Fedu": father_education,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "famsup": family_support,
        "schoolsup": school_support,
        "G1": g1,
        "G2": g2
    }])

    model = trained_models[selected_model]

    prediction = model.predict(student)[0]

    probabilities = model.predict_proba(student)[0]

    risk_probability = probabilities[
        list(model.classes_).index(1)
    ] * 100

    st.subheader("Prediction Result")

    st.write(f"**Selected model:** {selected_model}")

    if prediction == 1:
        st.warning(
            "Potential academic risk detected. "
            "Consider offering additional academic support."
        )
    else:
        st.success(
            "The model did not classify this student as at risk."
        )

    st.metric(
        "Estimated Academic-Risk Probability",
        f"{risk_probability:.1f}%"
    )

    st.progress(int(round(risk_probability)))

    st.caption(
        "This probability is a model estimate, not a guarantee. "
        "It does not measure the probability of dropping out."
    )

    # ---------------- EXPLANATION ----------------

    st.subheader("🔍 Important Factors")

    classifier = model.named_steps["classifier"]
    transformed = model.named_steps["preprocessor"]

    feature_names = transformed.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):

        importance_values = classifier.feature_importances_

    elif hasattr(classifier, "coef_"):

        importance_values = np.abs(classifier.coef_[0])

    else:
        importance_values = None

    if importance_values is not None:

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_values
        })

        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )

        importance_df = importance_df.sort_values(
            "Importance", ascending=False
        ).head(10)

        st.bar_chart(
            importance_df.set_index("Feature")["Importance"]
        )

        st.caption(
            "Tree models show feature importance. "
            "Logistic Regression shows absolute coefficient magnitudes. "
            "These measures have different interpretations and "
            "do not prove causation."
        )

    # ---------------- RECOMMENDATIONS ----------------

    st.subheader("💡 Suggested Academic Support")

    suggestions = []

    if absences >= 10:
        suggestions.append(
            "Review attendance barriers and help the student attend regularly."
        )

    if g1 < 10 or g2 < 10:
        suggestions.append(
            "Offer extra practice, tutoring, or subject-specific guidance."
        )

    if studytime == 1:
        suggestions.append(
            "Help the student create a manageable weekly study schedule."
        )

    if failures > 0:
        suggestions.append(
            "Review topics that caused difficulty in previous classes."
        )

    if family_support == "no":
        suggestions.append(
            "Discuss available academic support with the student and school."
        )

    if not suggestions:
        suggestions.append(
            "Continue regular progress checks and provide support when needed."
        )

    for suggestion in suggestions:
        st.write(f"- {suggestion}")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Student Performance & Dropout Predictor | "
    "Machine Learning Mini Project | Python + Streamlit"
)