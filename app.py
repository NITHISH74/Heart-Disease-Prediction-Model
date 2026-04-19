"""
app.py
======
Streamlit front-end for the Heart Disease Prediction System.

Displays CV accuracy, confusion matrix, feature importance, live predictions,
and local explainability via SHAP (Random Forest component) plus global
permutation importance from training.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st

# -----------------------------------------------------------------------------
# Paths & theme (Medical Blue)
# -----------------------------------------------------------------------------

ARTIFACT_PATH = Path(__file__).resolve().parent / "model_artifacts.joblib"

MEDICAL_BLUE_CSS = """
<style>
    /* Page background */
    .stApp {
        background: linear-gradient(180deg, #e8f2fb 0%, #f5fafd 100%);
    }
    /* Primary headers */
    h1, h2, h3 {
        color: #0b4f8c !important;
        font-weight: 600 !important;
    }
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        color: #0b4f8c;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b4f8c 0%, #1565a8 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f0f7ff !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #e3f0ff !important;
    }
    /* Buttons */
    .stButton > button {
        background-color: #0b6fb8;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #095a96;
        color: white;
    }
    /* Info boxes */
    div[data-testid="stExpander"] {
        border: 1px solid #b8d4ee;
        border-radius: 8px;
    }
</style>
"""


@st.cache_resource(show_spinner="Loading trained model…")
def load_artifacts():
    """Load joblib bundle produced by model_train.py."""
    if not ARTIFACT_PATH.is_file():
        return None
    return joblib.load(ARTIFACT_PATH)


def apply_medical_theme():
    st.markdown(MEDICAL_BLUE_CSS, unsafe_allow_html=True)


def build_input_vector(bundle: dict) -> np.ndarray:
    """Assemble feature vector from sidebar widgets (order matches training)."""
    names = bundle["feature_names"]
    row = []
    for n in names:
        key = f"inp_{n}"
        row.append(float(st.session_state.get(key, 0)))
    return np.array(row, dtype=np.float64).reshape(1, -1)


def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_title("Confusion Matrix (out-of-fold CV)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.patch.set_facecolor("#f5fafd")
    return fig


def plot_feature_importance(names: list[str], values: np.ndarray):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6, 5))
    v_range = float(np.ptp(values)) + 1e-9
    colors = plt.cm.Blues(0.35 + 0.5 * (values[order] - values.min()) / v_range)
    ax.barh(np.array(names)[order], values[order], color=colors)
    ax.set_xlabel("Mean decrease in accuracy (permutation)")
    ax.set_title("Global Feature Importance")
    fig.patch.set_facecolor("#f5fafd")
    plt.tight_layout()
    return fig


def shap_waterfall_for_rf(bundle: dict, X_single: np.ndarray):
    """
    Local explanation: SHAP TreeExplainer on the Random Forest estimator inside
    the soft-voting ensemble (approximates non-linear contributions for that base learner).
    """
    model = bundle["model"]
    rf = model.named_estimators_["rf"]
    feature_names = bundle["feature_names"]
    background = bundle["X_background"]

    explainer = shap.TreeExplainer(rf, data=background, feature_perturbation="interventional")
    ex = explainer(X_single, check_additivity=False)
    # Binary RF: take explanation row for positive class (index 1) when SHAP returns multi-output.
    row = ex[0]
    if hasattr(row, "values") and np.ndim(row.values) > 1 and row.values.shape[-1] == 2:
        row = shap.Explanation(
            values=row.values[:, 1],
            base_values=row.base_values[1] if np.ndim(row.base_values) else row.base_values,
            data=row.data,
            feature_names=feature_names,
        )
    elif hasattr(row, "feature_names") and row.feature_names is None:
        row.feature_names = feature_names

    out = shap.plots.waterfall(row, show=False, max_display=14)
    # Newer SHAP returns Axes; Streamlit expects a Figure.
    fig = out.get_figure() if hasattr(out, "get_figure") else plt.gcf()
    fig.patch.set_facecolor("#f5fafd")
    plt.tight_layout()
    return fig


def sidebar_inputs(bundle: dict):
    """Collect patient features; defaults near population medians."""
    st.sidebar.title("Patient inputs")
    st.sidebar.caption("UCI Heart Disease features — adjust sliders to simulate a case.")

    names = bundle["feature_names"]
    defaults = {
        "age": 54.0,
        "sex": 1.0,
        "cp": 1.0,
        "trestbps": 131.0,
        "chol": 246.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 149.0,
        "exang": 0.0,
        "oldpeak": 1.0,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 2.0,
    }

    for n in names:
        key = f"inp_{n}"
        if key not in st.session_state:
            st.session_state[key] = defaults.get(n, 0.0)

    st.sidebar.slider("Age (years)", 18, 90, int(st.session_state["inp_age"]), key="inp_age")
    st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="inp_sex")
    st.sidebar.selectbox(
        "Chest pain type (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x],
        key="inp_cp",
    )
    st.sidebar.slider("Resting BP (trestbps)", 80, 200, int(st.session_state["inp_trestbps"]), key="inp_trestbps")
    st.sidebar.slider("Serum cholesterol (chol)", 100, 600, int(st.session_state["inp_chol"]), key="inp_chol")
    st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1], key="inp_fbs")
    st.sidebar.selectbox("Resting ECG (restecg)", [0, 1, 2], key="inp_restecg")
    st.sidebar.slider("Max heart rate achieved (thalach)", 60, 220, int(st.session_state["inp_thalach"]), key="inp_thalach")
    st.sidebar.selectbox("Exercise induced angina (exang)", [0, 1], key="inp_exang")
    st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.5, float(st.session_state["inp_oldpeak"]), 0.1, key="inp_oldpeak")
    st.sidebar.selectbox("Slope of peak ST (slope)", [0, 1, 2], key="inp_slope")
    st.sidebar.selectbox("Major vessels colored (ca)", [0, 1, 2, 3, 4], key="inp_ca")
    st.sidebar.selectbox("Thalassemia (thal)", [0, 1, 2, 3], key="inp_thal")


def main():
    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_medical_theme()

    bundle = load_artifacts()
    if bundle is None:
        st.error(
            f"Model artifacts not found at `{ARTIFACT_PATH}`. "
            "Run `python model_train.py` first to train and save the model."
        )
        st.stop()

    sidebar_inputs(bundle)

    st.title("Heart Disease Prediction Dashboard")
    st.markdown(
        "Soft **VotingClassifier** (Random Forest + Logistic Regression + SVM) with "
        "**Stratified K-Fold** evaluation and **permutation importance** for global explainability."
    )

    # --- Top metrics ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Model accuracy (CV mean)",
            f"{bundle['cv_accuracy_mean']:.2%}",
            help="Mean accuracy across stratified K-fold folds.",
        )
    with c2:
        st.metric(
            "CV std",
            f"{bundle['cv_accuracy_std']:.4f}",
        )
    with c3:
        st.metric(
            "OOF accuracy",
            f"{bundle['oof_accuracy']:.2%}",
            help="Accuracy of out-of-fold predictions (same folds as confusion matrix).",
        )

    st.subheader("Evaluation & global importance")
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(plot_confusion_matrix(bundle["confusion_matrix"]), clear_figure=True)
    with col_b:
        st.pyplot(
            plot_feature_importance(
                bundle["feature_names"],
                np.asarray(bundle["permutation_importance_mean"]),
            ),
            clear_figure=True,
        )

    st.divider()
    st.subheader("Risk prediction")

    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        predict = st.button("Predict", type="primary", use_container_width=True)

    X_input = build_input_vector(bundle)
    model = bundle["model"]
    proba = model.predict_proba(X_input)[0]
    p_disease = float(proba[1])

    if predict:
        if p_disease >= 0.5:
            st.error(f"**High risk** — estimated probability of heart disease: **{p_disease:.1%}**")
        else:
            st.success(f"**Low risk** — estimated probability of heart disease: **{p_disease:.1%}**")
    else:
        st.info("Adjust sidebar values and click **Predict** to see risk and explanations.")

    st.subheader("Why this prediction? (explainability)")
    exp1, exp2 = st.columns(2)
    with exp1:
        st.markdown("**SHAP waterfall (Random Forest base learner)**")
        st.caption(
            "Shows how each feature pushes the RF output toward class 1 vs baseline. "
            "The full app prediction combines RF + LR + SVM via soft voting."
        )
        try:
            fig = shap_waterfall_for_rf(bundle, X_input)
            st.pyplot(fig, clear_figure=True)
            plt.close("all")
        except Exception as e:
            st.warning(f"Could not render SHAP plot: {e}")

    with exp2:
        st.markdown("**Input snapshot**")
        df_in = pd.DataFrame(X_input, columns=bundle["feature_names"])
        st.dataframe(df_in, use_container_width=True)

    st.divider()
    with st.expander("How this system works (viva notes)"):
        st.markdown(
            """
            - **Data**: UCI Heart Disease CSV (binary target: presence of disease).
            - **Models**: `VotingClassifier` with **soft** voting — averages calibrated probabilities from RF, LR, and SVM.
            - **Scaling**: Each linear model uses its own `StandardScaler` inside a sub-pipeline; tree model uses raw features.
            - **Validation**: **StratifiedKFold** keeps class balance per fold; confusion matrix uses **out-of-fold** predictions.
            - **Global importance**: **Permutation importance** on the full ensemble (drop in accuracy when a feature is shuffled).
            - **Local explanation**: **SHAP TreeExplainer** on the **Random Forest** base estimator for an interpretable additive breakdown.
            """
        )


if __name__ == "__main__":
    main()
