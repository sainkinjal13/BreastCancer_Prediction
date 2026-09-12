import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# RESPONSIVE CSS
# ==========================================================

st.markdown("""
<style>

/* ----------------------------------------------------------
   MAIN APP
---------------------------------------------------------- */

.stApp {
    background-color: #0e1117;
}


/* ----------------------------------------------------------
   RESPONSIVE PAGE WIDTH
---------------------------------------------------------- */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5%;
    padding-right: 5%;
}


/* ----------------------------------------------------------
   HEADINGS
---------------------------------------------------------- */

h1 {
    text-align: center;
    font-weight: 700;
    margin-bottom: 0.5rem;
}


/* ----------------------------------------------------------
   SUBTITLE
---------------------------------------------------------- */

.subtitle {
    text-align: center;
    color: #a0a8b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}


/* ----------------------------------------------------------
   METRIC CARDS
---------------------------------------------------------- */

div[data-testid="stMetric"] {
    background-color: #171b26;
    border: 1px solid #2a3140;
    border-radius: 12px;
    padding: 15px;
}


/* ----------------------------------------------------------
   INPUT LABELS
---------------------------------------------------------- */

.stNumberInput label {
    font-size: 16px;
    font-weight: 600;
}


/* ----------------------------------------------------------
   INPUT BOXES
---------------------------------------------------------- */

.stNumberInput input {
    border-radius: 8px;
}


/* ----------------------------------------------------------
   BUTTON
---------------------------------------------------------- */

div.stButton > button {
    width: 100%;
    min-height: 52px;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 600;
}


/* ----------------------------------------------------------
   EXPANDER
---------------------------------------------------------- */

.streamlit-expanderHeader {
    font-size: 16px;
}


/* ----------------------------------------------------------
   MOBILE RESPONSIVENESS
---------------------------------------------------------- */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    h1 {
        font-size: 28px !important;
    }

    .subtitle {
        font-size: 15px;
    }

    div.stButton > button {
        min-height: 50px;
        font-size: 16px;
    }
}


/* ----------------------------------------------------------
   SMALL MOBILE SCREENS
---------------------------------------------------------- */

@media (max-width: 480px) {

    .block-container {
        padding-left: 0.7rem;
        padding-right: 0.7rem;
    }

    h1 {
        font-size: 24px !important;
    }

    .subtitle {
        font-size: 14px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD MACHINE LEARNING MODEL
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "breast_cancer_model.pkl"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ==========================================================
# HEADER
# ==========================================================

st.title("🩺 Breast Cancer Prediction System")

st.markdown(
    """
    <p class="subtitle">
    Enter the tumor measurement values below to analyze the data
    using a trained Machine Learning model.
    </p>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# INFORMATION CARDS
# ==========================================================

col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        label="🤖 Model",
        value="Logistic Regression"
    )


with col2:
    st.metric(
        label="📊 Input Features",
        value=f"{len(model.feature_names_in_)}"
    )


with col3:
    st.metric(
        label="⚡ Analysis",
        value="Instant Prediction"
    )


st.divider()


# ==========================================================
# INPUT SECTION
# ==========================================================

st.subheader("🔬 Enter Tumor Measurements")

st.caption(
    "Provide the required tumor measurement values for prediction."
)


# ==========================================================
# RESPONSIVE INPUT COLUMNS
# ==========================================================

left_column, right_column = st.columns(2)


with left_column:

    radius_mean = st.number_input(
        "Radius Mean",
        min_value=0.0,
        value=0.0,
        format="%.2f",
        help="Mean distance from the center to points on the perimeter."
    )

    texture_mean = st.number_input(
        "Texture Mean",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

    radius_worst = st.number_input(
        "Radius Worst",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )


with right_column:

    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

    area_mean = st.number_input(
        "Area Mean",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )


# ==========================================================
# CREATE FEATURE DATA
# ==========================================================

feature_data = {
    "radius_mean": radius_mean,
    "texture_mean": texture_mean,
    "perimeter_mean": perimeter_mean,
    "area_mean": area_mean,
    "radius_worst": radius_worst
}


# ==========================================================
# PREDICTION BUTTON
# ==========================================================

st.write("")

predict_button = st.button(
    "🔍 Analyze & Predict",
    use_container_width=True
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    # Create DataFrame
    input_data = pd.DataFrame([feature_data])

    # Arrange features in EXACT order required by model
    input_data = input_data[model.feature_names_in_]

    # Make prediction
    prediction = model.predict(input_data)

    # Get probability
    probability = model.predict_proba(input_data)


    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction[0] == 1:

        confidence = probability[0][1] * 100

        st.error(
            "### ⚠️ Model Prediction: Malignant"
        )

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

    else:

        confidence = probability[0][0] * 100

        st.success(
            "### ✅ Model Prediction: Benign"
        )

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )


    # ------------------------------------------------------
    # INPUT SUMMARY
    # ------------------------------------------------------

    with st.expander("📋 View Submitted Measurements"):

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Developed using Python • Streamlit • Scikit-learn"
)

st.caption(
    "⚠️ Educational machine-learning demonstration only. "
    "This application is not a substitute for professional medical evaluation."
)