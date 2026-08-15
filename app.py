
import streamlit as st
import os
import tempfile
import shutil
import gc
import pandas as pd
import numpy as np
import joblib
import altair as alt
import shap
from dashboard_views import (
    render_medium_risk_intelligence,
    render_low_risk_intelligence
)
from executive_dashboard import render_executive_dashboard


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Taxpayer Risk Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

LOGO_PATH = os.path.join(
    BASE_DIR,
    "nrs_logo.png"
)

HERO_PATH = os.path.join(
    BASE_DIR,
    "hero_image.png"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ----------------------------------------------------------
   MAIN APPLICATION
---------------------------------------------------------- */

.stApp {
    background-color: #F5F6F8;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}


/* ----------------------------------------------------------
   STREAMLIT HEADER
---------------------------------------------------------- */

[data-testid="stHeader"] {
    background: transparent;
}


/* ----------------------------------------------------------
   LEFT HERO CONTENT
---------------------------------------------------------- */

.hero-left {
    padding-top: 0.5rem;
    padding-right: 1rem;
}

.hero-title {
    font-size: 3.1rem;
    line-height: 1.05;
    font-weight: 800;
    color: #202833;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}

.hero-title-red {
    color: #D62828;
}

.hero-subtitle {
    font-size: 1.08rem;
    line-height: 1.6;
    color: #4E5661;
    max-width: 600px;
    margin-bottom: 1.2rem;
}

.red-line {
    width: 90px;
    height: 4px;
    background-color: #D62828;
    border-radius: 5px;
    margin-top: 0.8rem;
    margin-bottom: 1.5rem;
}


/* ----------------------------------------------------------
   FEATURE CARDS
---------------------------------------------------------- */

.feature-row {
    display: flex;
    gap: 12px;
    margin-top: 1.2rem;
    margin-bottom: 1rem;
}

.feature-card {
    flex: 1;
    background-color: #FFFFFF;
    border: 1px solid #E4E6EA;
    border-radius: 14px;
    padding: 16px 8px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
}

.feature-icon {
    width: 42px;
    height: 42px;
    margin: 0 auto 9px auto;
    border-radius: 10px;
    background-color: #FFF0F0;
    color: #D62828;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    font-weight: 700;
}

.feature-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #252C35;
    line-height: 1.25;
}


/* ----------------------------------------------------------
   IMAGES
---------------------------------------------------------- */

[data-testid="stImage"] img {
    border-radius: 22px;
}


/* ----------------------------------------------------------
   UPLOAD SECTION
---------------------------------------------------------- */

.upload-title {
    font-size: 1.55rem;
    font-weight: 750;
    color: #252C35;
    margin-bottom: 0.4rem;
}

.upload-description {
    color: #66707C;
    font-size: 0.98rem;
    line-height: 1.6;
}


/* ----------------------------------------------------------
   FILE UPLOADER
---------------------------------------------------------- */

[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border-radius: 16px;
    padding: 0.4rem;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E3E8;
    border-radius: 14px;
}

[data-testid="stFileUploader"] button {
    background-color: #D62828 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    padding-left: 1.3rem !important;
    padding-right: 1.3rem !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #B71C1C !important;
    color: #FFFFFF !important;
    border: none !important;
}


/* ----------------------------------------------------------
   STANDARD BUTTONS
---------------------------------------------------------- */

div.stButton > button {
    background-color: #D62828;
    color: #FFFFFF;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

div.stButton > button:hover {
    background-color: #B71C1C;
    color: #FFFFFF;
    border: none;
}


/* ----------------------------------------------------------
   MOBILE RESPONSIVENESS
---------------------------------------------------------- */

@media (max-width: 900px) {

    .hero-title {
        font-size: 2.3rem;
    }

    .feature-row {
        flex-direction: column;
    }

}

</style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

left_col, right_col = st.columns(
    [0.9, 1.25],
    gap="large"
)


# ============================================================
# LEFT HERO PANEL
# ============================================================

with left_col:

    st.image(
        LOGO_PATH,
        width=280
    )

    st.markdown(
        """
<div class="hero-left">
<div class="hero-title"><span class="hero-title-red">TAXPAYER RISK</span><br>PREDICTION SYSTEM</div>
<div class="red-line"></div>
<div class="hero-subtitle">
Machine-learning-powered risk classification and compliance intelligence
for smarter, data-driven taxpayer risk assessment.
</div>
<div class="feature-row">
<div class="feature-card">
<div class="feature-icon">✓</div>
<div class="feature-title">AI-Powered<br>Risk Analysis</div>
</div>
<div class="feature-card">
<div class="feature-icon">▥</div>
<div class="feature-title">Data-Driven<br>Insights</div>
</div>
<div class="feature-card">
<div class="feature-icon">◎</div>
<div class="feature-title">Smarter Compliance<br>Decisions</div>
</div>
</div>
</div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT HERO PANEL
# ============================================================

with right_col:

    st.image(
        HERO_PATH,
        use_container_width=True
    )


# ============================================================
# SEPARATOR
# ============================================================

st.markdown("---")


# ============================================================
# UPLOAD PANEL
# ============================================================

upload_left, upload_right = st.columns(
    [1.1, 2.4],
    gap="large"
)


with upload_left:

    st.markdown(
        """
<div class="upload-title">Upload Taxpayer Dataset</div>
<div class="upload-description">
Upload taxpayer financial and sector data for machine-learning
risk analysis.<br><br>
<strong>Supported formats:</strong> CSV and Excel
</div>
        """,
        unsafe_allow_html=True
    )


with upload_right:

    uploaded_file = st.file_uploader(
        "Upload taxpayer dataset",
        type=["csv", "xlsx"],
        label_visibility="collapsed"
    )



# ============================================================
# MODEL PATH
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)


# ============================================================
# LOAD SAVED MODEL COMPONENTS
# ============================================================

@st.cache_resource
def load_model_components():

    model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "clean_random_forest_model.joblib"
        )
    )

    encoder = joblib.load(
        os.path.join(
            MODEL_DIR,
            "label_encoder.joblib"
        )
    )

    model_features = joblib.load(
        os.path.join(
            MODEL_DIR,
            "model_feature_columns.joblib"
        )
    )

    numeric_medians = joblib.load(
        os.path.join(
            MODEL_DIR,
            "numeric_medians.joblib"
        )
    )

    raw_features = joblib.load(
        os.path.join(
            MODEL_DIR,
            "raw_feature_columns.joblib"
        )
    )

    valid_divisions = joblib.load(
        os.path.join(
            MODEL_DIR,
            "valid_divisions.joblib"
        )
    )

    valid_subdivisions = joblib.load(
        os.path.join(
            MODEL_DIR,
            "valid_subdivisions.joblib"
        )
    )

    return (
        model,
        encoder,
        model_features,
        numeric_medians,
        raw_features,
        valid_divisions,
        valid_subdivisions
    )


(
    loaded_model,
    loaded_encoder,
    loaded_features,
    loaded_medians,
    raw_feature_columns,
    valid_divisions,
    valid_subdivisions

) = load_model_components()

# ============================================================
# SHAP EXPLAINER
# ============================================================

#shap_explainer = shap.TreeExplainer(loaded_model)
shap_explainer = shap.TreeExplainer(loaded_model)




# ============================================================
# REUSABLE PREDICTION FUNCTION
# ============================================================

def predict_taxpayer_risk(input_df):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(input_df, pd.DataFrame):

        raise TypeError(
            "Input data must be a pandas DataFrame."
        )


    if input_df.empty:

        raise ValueError(
            "The uploaded dataset is empty."
        )


    missing_columns = [

        col

        for col in required_upload_columns

        if col not in input_df.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


    # --------------------------------------------------------
    # Preserve original uploaded dataset
    # --------------------------------------------------------

    results_df = input_df.copy()


    # --------------------------------------------------------
    # Select required input columns
    # --------------------------------------------------------

    X_raw = input_df[
        required_upload_columns
    ].copy()


    # --------------------------------------------------------
    # Create engineered features
    # --------------------------------------------------------

    X_raw["zero_profit_flag"] = (

        X_raw["Total Profit"] == 0

    ).astype(int)


    X_raw["high_rev_no_profit_flag"] = (

        (X_raw["Revenue"] >= 500_000_000)

        &

        (X_raw["Total Profit"] == 0)

    ).astype(int)


    # --------------------------------------------------------
    # Match raw training feature order
    # --------------------------------------------------------

    X_raw = X_raw[
        raw_feature_columns
    ]


    # --------------------------------------------------------
    # Define categorical columns
    # --------------------------------------------------------

    categorical_columns = [

        "Division_Description",

        "SubDivision_Description"
    ]


    # --------------------------------------------------------
    # Identify numerical columns
    # --------------------------------------------------------

    numerical_columns = [

        col

        for col in X_raw.columns

        if col not in categorical_columns
    ]


    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    for col in numerical_columns:

        X_raw[col] = pd.to_numeric(
            X_raw[col],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Fill missing numerical values
    # --------------------------------------------------------

    for col in numerical_columns:

        if col in loaded_medians:

            X_raw[col] = X_raw[col].fillna(
                loaded_medians[col]
            )


    # --------------------------------------------------------
    # Handle missing categories
    # --------------------------------------------------------

    X_raw["Division_Description"] = (

        X_raw["Division_Description"]
        .fillna("Unknown")
        .astype(str)
    )


    X_raw["SubDivision_Description"] = (

        X_raw["SubDivision_Description"]
        .fillna("Unknown")
        .astype(str)
    )


    # --------------------------------------------------------
    # One-hot encode categorical features
    # --------------------------------------------------------

    X_encoded = pd.get_dummies(

        X_raw,

        columns=categorical_columns
    )


    # --------------------------------------------------------
    # Align with 378 trained features
    # --------------------------------------------------------

    X_encoded = X_encoded.reindex(

        columns=loaded_features,

        fill_value=0
    )

    # --------------------------------------------------------
    # Store encoded features for Explainable AI
    # --------------------------------------------------------

    st.session_state["X_encoded"] = X_encoded.copy()

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    encoded_predictions = loaded_model.predict(
        X_encoded
    )


    predicted_classes = (
        loaded_encoder.inverse_transform(
            encoded_predictions
        )
    )


    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    probabilities = loaded_model.predict_proba(
        X_encoded
    )


    # --------------------------------------------------------
    # Add risk classes
    # --------------------------------------------------------

    results_df["Predicted_Risk_Class"] = (
        predicted_classes
    )


    # --------------------------------------------------------
    # Add class probabilities
    # --------------------------------------------------------

    model_class_labels = (

        loaded_encoder.inverse_transform(
            loaded_model.classes_
        )
    )


    for index, class_name in enumerate(
        model_class_labels
    ):

        results_df[
            f"Probability_{class_name}"
        ] = probabilities[:, index]


    # --------------------------------------------------------
    # Add prediction confidence
    # --------------------------------------------------------

    results_df["Prediction_Confidence"] = (

        probabilities.max(axis=1)
    )


    # --------------------------------------------------------
    # Add engineered features
    # --------------------------------------------------------

    results_df["zero_profit_flag"] = (

        X_raw["zero_profit_flag"].values
    )


    results_df["high_rev_no_profit_flag"] = (

        X_raw[
            "high_rev_no_profit_flag"
        ].values
    )


    return results_df



# ============================================================
# REQUIRED MODEL INPUT COLUMNS
# ============================================================

required_upload_columns = [

    "Revenue",

    "Total Profit",

    "Division_Description",

    "SubDivision_Description",

    "Total Profit to Turnover",

    "Cost of Sales to Turnover",

    "Operating Expenses to Turnover",

    "Operating Expenses to Net Profit",

    "Net Profit to Total Assets",

    "Gross Profit to Turnover",

    "Share Holders Equity to Total Assets",

    "Liquid Assets to Total Assets",

    "Current Assets to Current Liabilities"

]


# ============================================================
# READ AND VALIDATE UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # FILE INFORMATION
        # ----------------------------------------------------

        file_size_mb = (
            uploaded_file.size
            / (1024 * 1024)
        )


        st.caption(
            f"Selected file: {uploaded_file.name} "
            f"({file_size_mb:.2f} MB)"
        )


        if file_size_mb >= 50:

            st.warning(
                "This is a large file. Initial loading may take "
                "several minutes, particularly for Excel files. "
                "Do not refresh the page while processing."
            )


        # ----------------------------------------------------
        # CREATE A FILE SIGNATURE
        #
        # This allows the app to recognise whether the current
        # upload has already been loaded.
        # ----------------------------------------------------

        current_file_signature = (

            uploaded_file.name,

            uploaded_file.size
        )


        stored_file_signature = (

            st.session_state.get(
                "loaded_file_signature"
            )
        )


        # ----------------------------------------------------
        # READ FILE ONLY IF IT IS NEW
        # ----------------------------------------------------

        if (
            stored_file_signature
            != current_file_signature
        ):

            # Clear results belonging to an earlier file
            st.session_state.pop(
                "uploaded_dataframe",
                None
            )

            st.session_state.pop(
                "prediction_results",
                None
            )

            st.session_state.pop(
                "prediction_file_name",
                None
            )


            temporary_file_path = None


            with st.status(
                "Loading taxpayer dataset...",
                expanded=True
            ) as loading_status:

                st.write(
                    "Saving uploaded file temporarily..."
                )


                # --------------------------------------------
                # PRESERVE ORIGINAL FILE EXTENSION
                # --------------------------------------------

                file_extension = os.path.splitext(
                    uploaded_file.name
                )[1].lower()


                # --------------------------------------------
                # WRITE UPLOAD TO A TEMPORARY FILE
                # --------------------------------------------

                with tempfile.NamedTemporaryFile(

                    delete=False,

                    suffix=file_extension

                ) as temporary_file:

                    uploaded_file.seek(0)

                    shutil.copyfileobj(

                        uploaded_file,

                        temporary_file
                    )

                    temporary_file_path = (

                        temporary_file.name
                    )


                try:

                    # ----------------------------------------
                    # READ CSV
                    # ----------------------------------------

                    if file_extension == ".csv":

                        st.write(
                            "Reading CSV records..."
                        )


                        try:

                            uploaded_df = pd.read_csv(

                                temporary_file_path,

                                low_memory=False,

                                encoding="utf-8"
                            )


                        except UnicodeDecodeError:

                            st.write(
                                "UTF-8 encoding was not detected. "
                                "Trying an alternative encoding..."
                            )


                            uploaded_df = pd.read_csv(

                                temporary_file_path,

                                low_memory=False,

                                encoding="latin-1"
                            )


                    # ----------------------------------------
                    # READ EXCEL
                    # ----------------------------------------

                    elif file_extension == ".xlsx":

                        st.write(
                            "Reading Excel workbook. "
                            "Large workbooks can take several minutes..."
                        )


                        uploaded_df = pd.read_excel(

                            temporary_file_path,

                            engine="openpyxl"
                        )


                    else:

                        raise ValueError(

                            "Unsupported file format. "
                            "Please upload a CSV or XLSX file."
                        )


                    # ----------------------------------------
                    # VALIDATE THAT RECORDS WERE READ
                    # ----------------------------------------

                    if uploaded_df.empty:

                        raise ValueError(

                            "The uploaded file contains "
                            "no readable records."
                        )


                    # ----------------------------------------
                    # STORE LOADED DATA IN SESSION STATE
                    # ----------------------------------------

                    st.session_state[
                        "uploaded_dataframe"
                    ] = uploaded_df


                    st.session_state[
                        "loaded_file_signature"
                    ] = current_file_signature


                    loading_status.update(

                        label=(
                            "Dataset loaded successfully."
                        ),

                        state="complete",

                        expanded=False
                    )


                finally:

                    # ----------------------------------------
                    # DELETE TEMPORARY FILE
                    # ----------------------------------------

                    if (

                        temporary_file_path

                        and

                        os.path.exists(
                            temporary_file_path
                        )
                    ):

                        os.remove(
                            temporary_file_path
                        )


                    gc.collect()


        # ----------------------------------------------------
        # REUSE DATAFRAME ALREADY LOADED IN THIS SESSION
        # ----------------------------------------------------

        else:

            uploaded_df = st.session_state[

                "uploaded_dataframe"
            ]


            st.info(
                "The dataset is already loaded in memory. "
                "The app is reusing it without reading the "
                "workbook again."
            )


        # ----------------------------------------------------
        # CONFIRM FILE LOADED
        # ----------------------------------------------------

        st.success(
            f"File loaded successfully: {uploaded_file.name}"
        )


        # ----------------------------------------------------
        # DATASET SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Dataset Summary"
        )


        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.metric(
                "Number of Records",
                f"{len(uploaded_df):,}"
            )


        with summary_col2:

            st.metric(
                "Number of Columns",
                uploaded_df.shape[1]
            )


        # ----------------------------------------------------
        # CHECK REQUIRED COLUMNS
        # ----------------------------------------------------

        missing_columns = [

            column

            for column in required_upload_columns

            if column not in uploaded_df.columns

        ]


        # ----------------------------------------------------
        # INVALID DATASET
        # ----------------------------------------------------

        if missing_columns:

            st.error(
                "Dataset validation failed. "
                "The uploaded file is missing required columns."
            )


            st.write(
                "### Missing Required Columns"
            )


            for column in missing_columns:

                st.write(
                    f"- {column}"
                )


        # ----------------------------------------------------
        # VALID DATASET
        # ----------------------------------------------------

        else:

            st.success(
                "Dataset validation passed. "
                "All 13 required model input columns are present."
            )


            # ------------------------------------------------
            # DATA PREVIEW
            # ------------------------------------------------

            st.subheader(
                "Data Preview"
            )


            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True
            )


            st.info(
                "The dataset has passed structural validation "
                "and is ready for preprocessing and risk prediction."
            )


            # =================================================
            # RUN PREDICTION
            # =================================================

            if st.button(
                "Run Risk Prediction",
                type="primary"
            ):

                with st.spinner(
                    "Running taxpayer risk prediction..."
                ):

                    prediction_results = (
                        predict_taxpayer_risk(
                            uploaded_df
                        )
                    )


                # ---------------------------------------------
                # STORE RESULTS IN SESSION STATE
                # ---------------------------------------------

                st.session_state[
                    "prediction_results"
                ] = prediction_results


                st.session_state[
                    "prediction_file_name"
                ] = uploaded_file.name


                st.success(
                    "Risk prediction completed successfully."
                )


            # =================================================
            # DISPLAY STORED PREDICTION RESULTS
            # =================================================

            if (
                "prediction_results"
                in st.session_state
            ):

                prediction_results = (
                    st.session_state[
                        "prediction_results"
                    ]
                )


                # ---------------------------------------------
                # CHECK THAT RESULTS BELONG TO CURRENT FILE
                # ---------------------------------------------

                stored_file_name = (
                    st.session_state.get(
                        "prediction_file_name"
                    )
                )


                if stored_file_name == uploaded_file.name:


                    # =========================================
                    # DASHBOARD TITLE
                    # =========================================

                    st.markdown("---")

                    st.header(
                        "Risk Analysis Dashboard"
                    )


                    # =========================================
                    # CALCULATE KPI VALUES
                    # =========================================

                    total_records = len(
                        prediction_results
                    )


                    risk_counts = (

                        prediction_results[
                            "Predicted_Risk_Class"
                        ]
                        .value_counts()
                    )


                    high_count = int(
                        risk_counts.get(
                            "High",
                            0
                        )
                    )


                    medium_count = int(
                        risk_counts.get(
                            "Medium",
                            0
                        )
                    )


                    low_count = int(
                        risk_counts.get(
                            "Low",
                            0
                        )
                    )


                    # =========================================
                    # KPI CARDS
                    # =========================================

                    kpi1, kpi2, kpi3, kpi4 = (
                        st.columns(4)
                    )


                    with kpi1:

                        st.metric(
                            "Total Records",
                            f"{total_records:,}"
                        )


                    with kpi2:

                        st.metric(
                            "High Risk",
                            f"{high_count:,}",
                            f"{(
                                high_count
                                / total_records
                                * 100
                            ):.2f}%"
                        )


                    with kpi3:

                        st.metric(
                            "Medium Risk",
                            f"{medium_count:,}",
                            f"{(
                                medium_count
                                / total_records
                                * 100
                            ):.2f}%"
                        )


                    with kpi4:

                        st.metric(
                            "Low Risk",
                            f"{low_count:,}",
                            f"{(
                                low_count
                                / total_records
                                * 100
                            ):.2f}%"
                        )


                    # =========================================
                    # RISK DISTRIBUTION CHART
                    # =========================================

                    st.subheader(
                        "Risk Level Distribution"
                    )


                    chart_order = [
                        "High",
                        "Medium",
                        "Low"
                    ]


                    chart_data = pd.DataFrame({

                        "Risk Level":
                            chart_order,

                        "Number of Records": [

                            high_count,

                            medium_count,

                            low_count
                        ]
                    })


                    chart_data = (
                        chart_data.set_index(
                            "Risk Level"
                        )
                    )


                    # =========================================
                    # NRS RED RISK DISTRIBUTION CHART
                    # =========================================

                    chart_plot_data = (
                        chart_data
                        .reset_index()
                    )


                    risk_chart = (

                        alt.Chart(
                            chart_plot_data
                        )

                        .mark_bar(
                            color="#D62828",
                            cornerRadiusTopLeft=5,
                            cornerRadiusTopRight=5
                        )

                        .encode(

                            x=alt.X(
                                "Risk Level:N",
                                sort=[
                                    "High",
                                    "Medium",
                                    "Low"
                                ],
                                title="Risk Level"
                            ),

                            y=alt.Y(
                                "Number of Records:Q",
                                title="Number of Records"
                            ),

                            tooltip=[

                                alt.Tooltip(
                                    "Risk Level:N",
                                    title="Risk Level"
                                ),

                                alt.Tooltip(
                                    "Number of Records:Q",
                                    title="Records",
                                    format=","
                                )
                            ]
                        )

                        .properties(
                            height=380
                        )
                    )


                    st.altair_chart(
                        risk_chart,
                        use_container_width=True
                    )


                    # =========================================
                    # DETAILED DISTRIBUTION TABLE
                    # =========================================

                    st.subheader(
                        "Detailed Risk Distribution"
                    )


                    risk_distribution = pd.DataFrame({

                        "Risk Class": [
                            "High",
                            "Medium",
                            "Low"
                        ],

                        "Number of Records": [
                            high_count,
                            medium_count,
                            low_count
                        ],

                        "Percentage": [

                            round(
                                high_count
                                / total_records
                                * 100,
                                2
                            ),

                            round(
                                medium_count
                                / total_records
                                * 100,
                                2
                            ),

                            round(
                                low_count
                                / total_records
                                * 100,
                                2
                            )
                        ]
                    })


                    st.dataframe(
                        risk_distribution,
                        use_container_width=True,
                        hide_index=True
                    )

                    render_executive_dashboard(
                        prediction_results
                    )




                    # =========================================
                    # TAXPAYER RECORD INTELLIGENCE
                    # =========================================

                    st.markdown("---")

                    st.header(
                        "Taxpayer Record Intelligence"
                    )

                    st.write(
                        "This section separates assessment-level "
                        "records from unique taxpayers. Multiple "
                        "records for the same TIN are preserved "
                        "because they may represent different "
                        "assessment years or assessment types."
                    )


                    # -----------------------------------------
                    # CHECK FOR TIN COLUMN
                    # -----------------------------------------

                    if "TIN" not in prediction_results.columns:

                        st.warning(
                            "TIN is not available in this dataset, "
                            "so unique-taxpayer analysis cannot be "
                            "performed."
                        )


                    else:

                        taxpayer_data = (
                            prediction_results.copy()
                        )


                        # =====================================
                        # IDENTIFY ASSESSMENT YEAR COLUMN
                        # =====================================

                        possible_year_columns = [

                            "Assessment Year",
                            "Assessment_Year",
                            "assessment_year",
                            "Year of Assessment",
                            "Year"
                        ]


                        taxpayer_year_column = next(

                            (
                                column

                                for column
                                in possible_year_columns

                                if column
                                in taxpayer_data.columns
                            ),

                            None
                        )


                        # =====================================
                        # IDENTIFY ASSESSMENT TYPE COLUMN
                        # =====================================

                        possible_type_columns = [

                            "Assessment Type",
                            "Assessment_Type",
                            "assessment_type",
                            "AssessmentType"
                        ]


                        taxpayer_type_column = next(

                            (
                                column

                                for column
                                in possible_type_columns

                                if column
                                in taxpayer_data.columns
                            ),

                            None
                        )


                        # =====================================
                        # BASIC TAXPAYER COUNTS
                        # =====================================

                        total_records_taxpayer_view = len(
                            taxpayer_data
                        )


                        unique_taxpayers = (

                            taxpayer_data[
                                "TIN"
                            ]
                            .nunique(
                                dropna=True
                            )
                        )


                        tin_record_counts = (

                            taxpayer_data
                            .dropna(
                                subset=["TIN"]
                            )
                            .groupby(
                                "TIN"
                            )
                            .size()
                        )


                        repeated_taxpayers = int(

                            (
                                tin_record_counts > 1
                            )
                            .sum()
                        )


                        # =====================================
                        # CONFLICTING RISK CLASSIFICATIONS
                        # =====================================

                        risk_class_counts = (

                            taxpayer_data
                            .dropna(
                                subset=["TIN"]
                            )
                            .groupby(
                                "TIN"
                            )[
                                "Predicted_Risk_Class"
                            ]
                            .nunique()
                        )


                        conflicting_risk_taxpayers = int(

                            (
                                risk_class_counts > 1
                            )
                            .sum()
                        )


                        # =====================================
                        # SUMMARY KPI CARDS
                        # =====================================

                        tin_kpi1, tin_kpi2, tin_kpi3, tin_kpi4 = (
                            st.columns(4)
                        )


                        with tin_kpi1:

                            st.metric(
                                "Assessment Records",
                                f"{total_records_taxpayer_view:,}"
                            )


                        with tin_kpi2:

                            st.metric(
                                "Unique TINs",
                                f"{unique_taxpayers:,}"
                            )


                        with tin_kpi3:

                            st.metric(
                                "TINs with Multiple Records",
                                f"{repeated_taxpayers:,}"
                            )


                        with tin_kpi4:

                            st.metric(
                                "TINs with Different Risk Classes",
                                f"{conflicting_risk_taxpayers:,}"
                            )


                        # =====================================
                        # CREATE RISK SEVERITY RANK
                        # =====================================

                        risk_severity = {

                            "Low": 1,

                            "Medium": 2,

                            "High": 3
                        }


                        taxpayer_data[
                            "_Risk_Severity"
                        ] = (

                            taxpayer_data[
                                "Predicted_Risk_Class"
                            ]
                            .map(
                                risk_severity
                            )
                        )


                        # =====================================
                        # CREATE ONE SUMMARY ROW PER TIN
                        # =====================================

                        taxpayer_summary_rows = []


                        for tin, tin_group in (

                            taxpayer_data
                            .dropna(
                                subset=["TIN"]
                            )
                            .groupby(
                                "TIN",
                                sort=False
                            )
                        ):

                            # ---------------------------------
                            # HIGHEST OBSERVED RISK
                            # ---------------------------------

                            highest_risk_row = (

                                tin_group

                                .sort_values(

                                    by=[
                                        "_Risk_Severity",
                                        "Prediction_Confidence"
                                    ],

                                    ascending=[
                                        False,
                                        False
                                    ]
                                )

                                .iloc[0]
                            )


                            # ---------------------------------
                            # LATEST RECORD
                            #
                            # If assessment year exists, use
                            # the latest year. Otherwise use
                            # the last available record.
                            # ---------------------------------

                            if taxpayer_year_column:

                                year_numeric = pd.to_numeric(

                                    tin_group[
                                        taxpayer_year_column
                                    ],

                                    errors="coerce"
                                )


                                if year_numeric.notna().any():

                                    latest_index = (

                                        year_numeric.idxmax()
                                    )


                                    latest_record = (

                                        tin_group.loc[
                                            latest_index
                                        ]
                                    )


                                else:

                                    latest_record = (

                                        tin_group.iloc[-1]
                                    )


                            else:

                                latest_record = (

                                    tin_group.iloc[-1]
                                )


                            # ---------------------------------
                            # YEARS REPRESENTED
                            # ---------------------------------

                            if taxpayer_year_column:

                                years_represented = ", ".join(

                                    sorted(

                                        tin_group[
                                            taxpayer_year_column
                                        ]
                                        .dropna()
                                        .astype(str)
                                        .unique()
                                    )
                                )

                            else:

                                years_represented = "N/A"


                            # ---------------------------------
                            # ASSESSMENT TYPES REPRESENTED
                            # ---------------------------------

                            if taxpayer_type_column:

                                types_represented = ", ".join(

                                    sorted(

                                        tin_group[
                                            taxpayer_type_column
                                        ]
                                        .dropna()
                                        .astype(str)
                                        .unique()
                                    )
                                )

                            else:

                                types_represented = "N/A"


                            # ---------------------------------
                            # BUILD TAXPAYER SUMMARY ROW
                            # ---------------------------------

                            summary_row = {

                                "TIN":
                                    tin,

                                "Number of Records":
                                    len(tin_group),

                                "Years Represented":
                                    years_represented,

                                "Assessment Types":
                                    types_represented,

                                "Highest Observed Risk":
                                    highest_risk_row[
                                        "Predicted_Risk_Class"
                                    ],

                                "Maximum Confidence (%)":
                                    round(

                                        tin_group[
                                            "Prediction_Confidence"
                                        ]
                                        .max()
                                        * 100,

                                        2
                                    ),

                                "Risk Classes Observed":
                                    ", ".join(

                                        sorted(

                                            tin_group[
                                                "Predicted_Risk_Class"
                                            ]
                                            .dropna()
                                            .astype(str)
                                            .unique()
                                        )
                                    ),

                                "Different Risk Classes":
                                    (
                                        tin_group[
                                            "Predicted_Risk_Class"
                                        ]
                                        .nunique()
                                        > 1
                                    )
                            }


                            # ---------------------------------
                            # OPTIONAL TAXPAYER NAME
                            # ---------------------------------

                            if "custName" in tin_group.columns:

                                summary_row[
                                    "Taxpayer Name"
                                ] = latest_record.get(
                                    "custName"
                                )


                            # ---------------------------------
                            # OPTIONAL SECTOR
                            # ---------------------------------

                            if (
                                "Division_Description"
                                in tin_group.columns
                            ):

                                summary_row[
                                    "Latest Sector"
                                ] = latest_record.get(
                                    "Division_Description"
                                )


                            # ---------------------------------
                            # LATEST REVENUE
                            # ---------------------------------

                            if "Revenue" in tin_group.columns:

                                summary_row[
                                    "Latest Record Revenue"
                                ] = latest_record.get(
                                    "Revenue"
                                )


                            # ---------------------------------
                            # LATEST PROFIT
                            # ---------------------------------

                            if (
                                "Total Profit"
                                in tin_group.columns
                            ):

                                summary_row[
                                    "Latest Record Profit"
                                ] = latest_record.get(
                                    "Total Profit"
                                )


                            taxpayer_summary_rows.append(
                                summary_row
                            )


                        unique_taxpayer_summary = pd.DataFrame(

                            taxpayer_summary_rows
                        )


                        # =====================================
                        # REORDER SUMMARY COLUMNS
                        # =====================================

                        preferred_summary_columns = [

                            "TIN",
                            "Taxpayer Name",
                            "Number of Records",
                            "Years Represented",
                            "Assessment Types",
                            "Highest Observed Risk",
                            "Risk Classes Observed",
                            "Different Risk Classes",
                            "Maximum Confidence (%)",
                            "Latest Sector",
                            "Latest Record Revenue",
                            "Latest Record Profit"
                        ]


                        ordered_summary_columns = [

                            column

                            for column
                            in preferred_summary_columns

                            if column
                            in unique_taxpayer_summary.columns
                        ]


                        unique_taxpayer_summary = (

                            unique_taxpayer_summary[
                                ordered_summary_columns
                            ]
                        )


                        # =====================================
                        # VIEW SELECTOR
                        # =====================================

                        st.subheader(
                            "Record-Level and Taxpayer-Level Views"
                        )


                        intelligence_view = st.radio(

                            "Select analysis level",

                            [
                                "Assessment / Record View",

                                "Unique Taxpayer View"
                            ],

                            horizontal=True,

                            key="taxpayer_intelligence_view"
                        )


                        # =====================================
                        # RECORD-LEVEL VIEW
                        # =====================================

                        if (

                            intelligence_view

                            == "Assessment / Record View"
                        ):

                            st.caption(
                                "Each row represents one uploaded "
                                "assessment or filing record. "
                                "The same TIN may therefore appear "
                                "more than once."
                            )


                            record_view_columns = []


                            for column in [

                                "TIN",
                                "custName",
                                taxpayer_year_column,
                                taxpayer_type_column,
                                "Division_Description",
                                "Revenue",
                                "Total Profit",
                                "Predicted_Risk_Class",
                                "Prediction_Confidence"

                            ]:

                                if (

                                    column

                                    and

                                    column
                                    in taxpayer_data.columns

                                    and

                                    column
                                    not in record_view_columns
                                ):

                                    record_view_columns.append(
                                        column
                                    )


                            record_level_view = (

                                taxpayer_data[
                                    record_view_columns
                                ]
                                .copy()
                            )


                            if (
                                "Prediction_Confidence"
                                in record_level_view.columns
                            ):

                                record_level_view[
                                    "Prediction_Confidence"
                                ] = (

                                    record_level_view[
                                        "Prediction_Confidence"
                                    ]
                                    * 100
                                ).round(2)


                            record_level_view = (

                                record_level_view.rename(

                                    columns={

                                        "custName":
                                            "Taxpayer Name",

                                        "Division_Description":
                                            "Sector",

                                        "Prediction_Confidence":
                                            "Confidence (%)"
                                    }
                                )
                            )


                            st.dataframe(

                                record_level_view,

                                use_container_width=True,

                                hide_index=True
                            )


                        # =====================================
                        # UNIQUE TAXPAYER VIEW
                        # =====================================

                        else:

                            st.caption(
                                "Each row represents one unique TIN. "
                                "Financial values shown are from the "
                                "latest available assessment-year "
                                "record where assessment year is "
                                "available. Revenue and profit are "
                                "not summed across records."
                            )


                            st.dataframe(

                                unique_taxpayer_summary,

                                use_container_width=True,

                                hide_index=True,

                                column_config={

                                    "Latest Record Revenue":
                                        st.column_config.NumberColumn(
                                            "Latest Record Revenue",
                                            format="₦ %,.2f"
                                        ),

                                    "Latest Record Profit":
                                        st.column_config.NumberColumn(
                                            "Latest Record Profit",
                                            format="₦ %,.2f"
                                        ),

                                    "Maximum Confidence (%)":
                                        st.column_config.NumberColumn(
                                            "Maximum Confidence (%)",
                                            format="%.2f%%"
                                        )
                                }
                            )


                        # =====================================
                        # CONFLICTING RISK CASES
                        # =====================================

                        with st.expander(
                            "View TINs with Different Risk Classes"
                        ):

                            conflicting_tins = (

                                risk_class_counts[

                                    risk_class_counts > 1

                                ]

                                .index
                            )


                            conflicting_records = (

                                taxpayer_data[

                                    taxpayer_data[
                                        "TIN"
                                    ]
                                    .isin(
                                        conflicting_tins
                                    )
                                ]

                                .copy()
                            )


                            if conflicting_records.empty:

                                st.info(
                                    "No TIN has different predicted "
                                    "risk classes across its records "
                                    "in the current dataset."
                                )


                            else:

                                conflict_columns = []


                                for column in [

                                    "TIN",
                                    "custName",
                                    taxpayer_year_column,
                                    taxpayer_type_column,
                                    "Division_Description",
                                    "Revenue",
                                    "Total Profit",
                                    "Predicted_Risk_Class",
                                    "Prediction_Confidence"

                                ]:

                                    if (

                                        column

                                        and

                                        column
                                        in conflicting_records.columns

                                        and

                                        column
                                        not in conflict_columns
                                    ):

                                        conflict_columns.append(
                                            column
                                        )


                                conflict_display = (

                                    conflicting_records[
                                        conflict_columns
                                    ]
                                    .copy()
                                )


                                if (
                                    "Prediction_Confidence"
                                    in conflict_display.columns
                                ):

                                    conflict_display[
                                        "Prediction_Confidence"
                                    ] = (

                                        conflict_display[
                                            "Prediction_Confidence"
                                        ]
                                        * 100
                                    ).round(2)


                                conflict_display = (

                                    conflict_display.rename(

                                        columns={

                                            "custName":
                                                "Taxpayer Name",

                                            "Division_Description":
                                                "Sector",

                                            "Prediction_Confidence":
                                                "Confidence (%)"
                                        }
                                    )
                                )


                                st.dataframe(

                                    conflict_display,

                                    use_container_width=True,

                                    hide_index=True
                                )



                    # =========================================
                    # HIGH-RISK INTELLIGENCE
                    # =========================================

                    st.markdown("---")

                    st.header(
                        "High-Risk Intelligence"
                    )

                    st.write(
                        "This section focuses on records classified "
                        "as High Risk for compliance review and "
                        "further investigation."
                    )


                    # -----------------------------------------
                    # FILTER HIGH-RISK RECORDS
                    # -----------------------------------------

                    high_risk_df = (

                        prediction_results[

                            prediction_results[
                                "Predicted_Risk_Class"
                            ] == "High"

                        ]

                        .copy()
                    )


                    # =========================================
                    # HIGH-RISK SUMMARY METRICS
                    # =========================================

                    high_kpi1, high_kpi2, high_kpi3 = (
                        st.columns(3)
                    )


                    with high_kpi1:

                        st.metric(
                            "High-Risk Records",
                            f"{len(high_risk_df):,}"
                        )


                    with high_kpi2:

                        if "TIN" in high_risk_df.columns:

                            high_unique_tins = (

                                high_risk_df[
                                    "TIN"
                                ]
                                .nunique()
                            )

                            st.metric(
                                "Unique High-Risk TINs",
                                f"{high_unique_tins:,}"
                            )

                        else:

                            st.metric(
                                "Unique High-Risk TINs",
                                "N/A"
                            )


                    with high_kpi3:

                        average_high_confidence = (

                            high_risk_df[
                                "Prediction_Confidence"
                            ]
                            .mean()
                            * 100
                        )

                        st.metric(
                            "Average High-Risk Confidence",
                            f"{average_high_confidence:.2f}%"
                        )


                    # =========================================
                    # HIGH-RISK SECTOR DISTRIBUTION
                    # =========================================

                    st.subheader(
                        "High-Risk Records by Sector"
                    )


                    if (
                        "Division_Description"
                        in high_risk_df.columns
                    ):

                        sector_distribution = (

                            high_risk_df[
                                "Division_Description"
                            ]

                            .fillna(
                                "Unknown"
                            )

                            .value_counts()

                            .rename_axis(
                                "Sector"
                            )

                            .reset_index(
                                name="High-Risk Records"
                            )
                        )


                        sector_chart = (

                            alt.Chart(
                                sector_distribution
                            )

                            .mark_bar(
                                color="#D62828",
                                cornerRadiusEnd=5,
                                size=24
                            )

                            .encode(

                                y=alt.Y(
                                    "Sector:N",
                                    sort="-x",
                                    title=None,

                                    axis=alt.Axis(
                                        labelLimit=500,
                                        labelFontSize=12,
                                        labelPadding=10
                                    )
                                ),

                                x=alt.X(
                                    "High-Risk Records:Q",
                                    title="Number of High-Risk Records",

                                    axis=alt.Axis(
                                        tickMinStep=1,
                                        titlePadding=12
                                    )
                                ),

                                tooltip=[

                                    alt.Tooltip(
                                        "Sector:N",
                                        title="Sector"
                                    ),

                                    alt.Tooltip(
                                        "High-Risk Records:Q",
                                        title="Records",
                                        format=","
                                    )
                                ]
                            )

                            .properties(

                                height=max(

                                    320,

                                    len(
                                        sector_distribution
                                    ) * 42
                                )
                            )

                            .configure_view(
                                strokeWidth=0
                            )
                        )


                        st.altair_chart(
                            sector_chart,
                            use_container_width=True
                        )


                        st.dataframe(
                            sector_distribution,
                            use_container_width=True,
                            hide_index=True
                        )


                    else:

                        st.warning(
                            "Sector information is not available "
                            "in the uploaded dataset."
                        )


                    # =========================================
                    # HIGH-RISK TAXPAYER REVIEW TABLE
                    # =========================================

                    st.subheader(
                        "High-Risk Taxpayer Review Table"
                    )


                    # -----------------------------------------
                    # IDENTIFY OPTIONAL YEAR COLUMN
                    # -----------------------------------------

                    possible_year_columns = [

                        "Assessment Year",
                        "Assessment_Year",
                        "assessment_year",
                        "Year of Assessment",
                        "Year"
                    ]


                    assessment_year_column = next(

                        (
                            column

                            for column
                            in possible_year_columns

                            if column
                            in high_risk_df.columns
                        ),

                        None
                    )


                    # -----------------------------------------
                    # IDENTIFY OPTIONAL ASSESSMENT TYPE COLUMN
                    # -----------------------------------------

                    possible_type_columns = [

                        "Assessment Type",
                        "Assessment_Type",
                        "assessment_type",
                        "AssessmentType"
                    ]


                    assessment_type_column = next(

                        (
                            column

                            for column
                            in possible_type_columns

                            if column
                            in high_risk_df.columns
                        ),

                        None
                    )


                    # -----------------------------------------
                    # BUILD REVIEW TABLE COLUMN LIST
                    # -----------------------------------------

                    review_columns = []


                    preferred_columns = [

                        "TIN",
                        "custName",
                        "Division_Description",
                        "SubDivision_Description",
                        "Revenue",
                        "Total Profit"
                    ]


                    for column in preferred_columns:

                        if column in high_risk_df.columns:

                            review_columns.append(
                                column
                            )


                    if assessment_year_column:

                        review_columns.append(
                            assessment_year_column
                        )


                    if assessment_type_column:

                        review_columns.append(
                            assessment_type_column
                        )


                    review_columns.extend([

                        "Predicted_Risk_Class",

                        "Probability_High",

                        "Prediction_Confidence"
                    ])


                    review_columns = [

                        column

                        for column in review_columns

                        if column in high_risk_df.columns
                    ]


                    high_risk_review = (

                        high_risk_df[
                            review_columns
                        ]

                        .sort_values(
                            by="Prediction_Confidence",
                            ascending=False
                        )

                        .copy()
                    )


                    # -----------------------------------------
                    # FORMAT CONFIDENCE AS PERCENTAGE
                    # -----------------------------------------

                    if (
                        "Prediction_Confidence"
                        in high_risk_review.columns
                    ):

                        high_risk_review[
                            "Prediction_Confidence"
                        ] = (

                            high_risk_review[
                                "Prediction_Confidence"
                            ]
                            * 100
                        ).round(2)


                    if (
                        "Probability_High"
                        in high_risk_review.columns
                    ):

                        high_risk_review[
                            "Probability_High"
                        ] = (

                            high_risk_review[
                                "Probability_High"
                            ]
                            * 100
                        ).round(2)


                    # =========================================
                    # PREPARE USER-FRIENDLY DISPLAY TABLE
                    # =========================================

                    display_review = (
                        high_risk_review.copy()
                    )


                    # -----------------------------------------
                    # RENAME COLUMNS FOR DISPLAY
                    # -----------------------------------------

                    display_column_names = {

                        "custName":
                            "Taxpayer Name",

                        "Division_Description":
                            "Sector",

                        "SubDivision_Description":
                            "Subsector",

                        "Probability_High":
                            "High-Risk Probability (%)",

                        "Prediction_Confidence":
                            "Confidence (%)"
                    }


                    display_review = (
                        display_review.rename(
                            columns=display_column_names
                        )
                    )


                    # =========================================
                    # BUILD COMPACT OPERATIONAL TABLE
                    # =========================================

                    compact_columns = []


                    preferred_compact_columns = [

                        "TIN",

                        "Taxpayer Name",

                        "Sector",

                        "Revenue",

                        "Total Profit"
                    ]


                    for column in preferred_compact_columns:

                        if column in display_review.columns:

                            compact_columns.append(
                                column
                            )


                    # -----------------------------------------
                    # ADD ASSESSMENT YEAR IF AVAILABLE
                    # -----------------------------------------

                    if (
                        assessment_year_column

                        and

                        assessment_year_column
                        in display_review.columns
                    ):

                        compact_columns.append(
                            assessment_year_column
                        )


                    # -----------------------------------------
                    # ADD ASSESSMENT TYPE IF AVAILABLE
                    # -----------------------------------------

                    if (
                        assessment_type_column

                        and

                        assessment_type_column
                        in display_review.columns
                    ):

                        compact_columns.append(
                            assessment_type_column
                        )


                    # -----------------------------------------
                    # ADD RISK OUTPUT COLUMNS
                    # -----------------------------------------

                    for column in [

                        "High-Risk Probability (%)",

                        "Confidence (%)"

                    ]:

                        if column in display_review.columns:

                            compact_columns.append(
                                column
                            )


                    compact_review = (

                        display_review[
                            compact_columns
                        ]
                        .copy()
                    )


                    # =========================================
                    # DISPLAY COMPACT REVIEW TABLE
                    # =========================================

                    st.caption(
                        "Operational review table. "
                        "Records are ranked from highest to "
                        "lowest prediction confidence."
                    )


                    st.dataframe(

                        compact_review,

                        use_container_width=True,

                        hide_index=True,

                        column_config={

                            "Revenue":
                                st.column_config.NumberColumn(
                                    "Revenue",
                                    format="₦ %,.2f"
                                ),

                            "Total Profit":
                                st.column_config.NumberColumn(
                                    "Total Profit",
                                    format="₦ %,.2f"
                                ),

                            "High-Risk Probability (%)":
                                st.column_config.NumberColumn(
                                    "High-Risk Probability (%)",
                                    format="%.2f%%"
                                ),

                            "Confidence (%)":
                                st.column_config.NumberColumn(
                                    "Confidence (%)",
                                    format="%.2f%%"
                                )
                        }
                    )


                    # =========================================
                    # EXPANDED RECORD DETAILS
                    # =========================================

                    with st.expander(
                        "View Full High-Risk Record Details"
                    ):

                        st.write(
                            "This expanded view contains the wider "
                            "record-level details available for each "
                            "High-Risk prediction."
                        )


                        st.dataframe(

                            display_review,

                            use_container_width=True,

                            hide_index=True,

                            column_config={

                                "Revenue":
                                    st.column_config.NumberColumn(
                                        "Revenue",
                                        format="₦ %,.2f"
                                    ),

                                "Total Profit":
                                    st.column_config.NumberColumn(
                                        "Total Profit",
                                        format="₦ %,.2f"
                                    ),

                                "High-Risk Probability (%)":
                                    st.column_config.NumberColumn(
                                        "High-Risk Probability (%)",
                                        format="%.2f%%"
                                    ),

                                "Confidence (%)":
                                    st.column_config.NumberColumn(
                                        "Confidence (%)",
                                        format="%.2f%%"
                                    )
                            }
                        )



                    # =========================================
                    # MEDIUM-RISK INTELLIGENCE — MODULAR VIEW
                    # =========================================

                    render_medium_risk_intelligence(
                        prediction_results
                    )

                    render_low_risk_intelligence(
                        prediction_results
                    )

                    st.markdown("---")
                    st.header("Explainable AI")

                    taxpayer_options = prediction_results["custName"].tolist()

                    selected_taxpayer = st.selectbox(
                        "Select a taxpayer",
                        taxpayer_options,
                        key="explainability_taxpayer"
                    )

                    st.success(f"Selected taxpayer: {selected_taxpayer}")

                    selected_index = prediction_results[
                        prediction_results["custName"] == selected_taxpayer
                    ].index[0]


                    if "X_encoded" in st.session_state:


                        feature_vector = (
                            st.session_state["X_encoded"]
                            .iloc[[selected_index]]
                        )


                        st.dataframe(feature_vector.iloc[:, :10])

                        # ============================================================
                        # CALCULATE SHAP VALUES
                        # ============================================================

                        #shap_values = shap_explainer.shap_values(feature_vector)
                        shap_values = shap_explainer.shap_values(
                            feature_vector,
                            check_additivity=False
                        )

                        #explanation = shap_explainer(feature_vector)






                        # ============================================================
                        # PREDICTED CLASS
                        # ============================================================

                        predicted_class = loaded_model.predict(feature_vector)[0]

                        predicted_probabilities = loaded_model.predict_proba(feature_vector)


                        # ============================================================
                        # SHAP VALUES FOR PREDICTED CLASS
                        # ============================================================

                        selected_shap_values = shap_values[0, :, predicted_class]

                        # ----------------------------------------------------------
                        # Scale SHAP values for presentation
                        # Preserves feature ranking and direction
                        # ----------------------------------------------------------

                        max_abs_shap = abs(selected_shap_values).max()

                        if max_abs_shap > 0: 
                          scaled_shap_values = selected_shap_values / max_abs_shap
                        else:
                          scaled_shap_values = selected_shap_values.copy()

                        # ============================================================
                        # FEATURE NAMES
                        # ============================================================



                        # ============================================================
                        # FEATURE IMPORTANCE TABLE
                        # ============================================================

                        importance_df = pd.DataFrame({
                            "Feature": feature_vector.columns,
                            #"SHAP Value": selected_shap_values#
                            "SHAP Value": scaled_shap_values
                        })

                        importance_df["Absolute SHAP"] = importance_df["SHAP Value"].abs()

                        importance_df = importance_df.sort_values(
                            "Absolute SHAP",
                            ascending=False
                        )


                        #st.dataframe(importance_df.head(10))

                        # ==========================================================
                        # PREDICTION EXPLANATION
                        # ==========================================================

                        # ============================================================
                        # PREPARE SHAP EXPLANATION DATA
                        # ============================================================

                        top_features = importance_df.head(10).copy()

                        top_features.insert(
                            0,
                            "Rank",
                            range(1, len(top_features) + 1)
                        )

                        top_features["Feature Value"] = top_features["Feature"].map(
                            feature_vector.iloc[0].to_dict()
                        )

                        # Keep explicit SHAP terminology for the technical explanation
                        top_features = top_features.rename(
                            columns={
                                "SHAP Value": "SHAP Contribution",
                                "Absolute SHAP": "Contribution Strength"
                            }
                        )

                        top_features["SHAP Contribution"] = (
                            top_features["SHAP Contribution"].round(4)
                        )

                        top_features["Contribution Strength"] = (
                            top_features["Contribution Strength"].round(4)
                        )

                        top_features["Direction"] = top_features["SHAP Contribution"].apply(
                            lambda x: (
                                "Supports the risk assessment"
                                if x > 0
                                else "Reduces support for the risk assessment"
                                if x < 0
                                else "Minimal influence"
                            )
                        )

                        # ============================================================
                        # BUSINESS FEATURE EXPLANATION
                        # ============================================================

                        # These are the non-sector variables actually used by the model.
                        business_feature_names = [
                            "Revenue",
                            "Total Profit",
                            "Total Profit to Turnover",
                            "Cost of Sales to Turnover",
                            "Operating Expenses to Turnover",
                            "Operating Expenses to Net Profit",
                            "Net Profit to Total Assets",
                            "Gross Profit to Turnover",
                            "Share Holders Equity to Total Assets",
                            "Liquid Assets to Total Assets",
                            "Current Assets to Current Liabilities",
                            "zero_profit_flag",
                            "high_rev_no_profit_flag"
                        ]

                        business_explanation_df = importance_df[
                            importance_df["Feature"].isin(business_feature_names)
                        ].copy()

                        business_explanation_df["Feature Value"] = (
                            business_explanation_df["Feature"].map(
                                feature_vector.iloc[0].to_dict()
                            )
                        )

                        business_explanation_df = business_explanation_df.rename(
                            columns={
                                "SHAP Value": "SHAP Contribution",
                                "Absolute SHAP": "Contribution Strength"
                            }
                        )

                        business_explanation_df["SHAP Contribution"] = (
                            business_explanation_df["SHAP Contribution"].round(4)
                        )

                        business_explanation_df["Contribution Strength"] = (
                            business_explanation_df["Contribution Strength"].round(4)
                        )

                        business_explanation_df["Direction"] = (
                            business_explanation_df["SHAP Contribution"].apply(
                                lambda x: (
                                    "Supports the risk assessment"
                                    if x > 0
                                    else "Reduces support for the risk assessment"
                                    if x < 0
                                    else "Minimal influence"
                                )
                            )
                        )

                        business_explanation_df = business_explanation_df.sort_values(
                            "Contribution Strength",
                            ascending=False
                        ).reset_index(drop=True)

                        business_explanation_df.insert(
                            0,
                            "Rank",
                            range(1, len(business_explanation_df) + 1)
                        )

                        # ------------------------------------------------------------
                        # Business-friendly feature names
                        # ------------------------------------------------------------

                        business_labels = {
                            "Revenue": "Revenue",
                            "Total Profit": "Total Profit",
                            "Total Profit to Turnover": "Profit Margin",
                            "Cost of Sales to Turnover": "Cost of Sales Ratio",
                            "Operating Expenses to Turnover": "Operating Expense Ratio",
                            "Operating Expenses to Net Profit": "Operating Expenses vs Net Profit",
                            "Net Profit to Total Assets": "Return on Assets",
                            "Gross Profit to Turnover": "Gross Profit Margin",
                            "Share Holders Equity to Total Assets": "Equity to Assets Ratio",
                            "Liquid Assets to Total Assets": "Liquid Assets Ratio",
                            "Current Assets to Current Liabilities": "Current Ratio",
                            "zero_profit_flag": "Zero Profit Condition",
                            "high_rev_no_profit_flag": "High Revenue / No Profit Condition"
                        }

                        business_explanation_df["Business Indicator"] = (
                            business_explanation_df["Feature"].map(business_labels)
                        )

                        business_explanation_df["Business Indicator"] = (
                            business_explanation_df["Business Indicator"]
                            .fillna(business_explanation_df["Feature"])
                        )

                        # ------------------------------------------------------------
                        # Plain-language interpretations
                        # ------------------------------------------------------------

                        def explain_business_driver(row):
                            feature = row["Feature"]
                            feature_value = row["Feature Value"]
                            shap_value = row["SHAP Contribution"]

                            if shap_value > 0:
                                direction_text = (
                                    "increased the model's support for the risk assessment"
                                )
                            elif shap_value < 0:
                                direction_text = (
                                    "reduced the model's support for the risk assessment"
                                )
                            else:
                                direction_text = "had minimal influence on the risk assessment"

                            # Binary indicators need explicit condition wording.
                            if feature == "zero_profit_flag":
                                detected = str(feature_value).strip().lower().startswith("yes")
                                condition_text = (
                                    "The zero-profit condition was detected for this taxpayer"
                                    if detected
                                    else "The zero-profit condition was not detected for this taxpayer"
                                )
                                return condition_text + ". This feature " + direction_text + "."

                            if feature == "high_rev_no_profit_flag":
                                detected = str(feature_value).strip().lower().startswith("yes")
                                condition_text = (
                                    "The high-revenue/no-profit condition was detected for this taxpayer"
                                    if detected
                                    else "The high-revenue/no-profit condition was not detected for this taxpayer"
                                )
                                return condition_text + ". This feature " + direction_text + "."

                            explanations = {
                                "Revenue":
                                    "The taxpayer's revenue level " + direction_text + ".",
                                "Total Profit":
                                    "The taxpayer's total profit " + direction_text + ".",
                                "Total Profit to Turnover":
                                    "The taxpayer's profit margin " + direction_text + ".",
                                "Cost of Sales to Turnover":
                                    "The taxpayer's cost of sales relative to turnover "
                                    + direction_text + ".",
                                "Operating Expenses to Turnover":
                                    "The taxpayer's operating expenses relative to turnover "
                                    + direction_text + ".",
                                "Operating Expenses to Net Profit":
                                    "The relationship between operating expenses and net profit "
                                    + direction_text + ".",
                                "Net Profit to Total Assets":
                                    "The taxpayer's return on assets " + direction_text + ".",
                                "Gross Profit to Turnover":
                                    "The taxpayer's gross profit margin " + direction_text + ".",
                                "Share Holders Equity to Total Assets":
                                    "The taxpayer's equity position relative to total assets "
                                    + direction_text + ".",
                                "Liquid Assets to Total Assets":
                                    "The taxpayer's liquid asset position " + direction_text + ".",
                                "Current Assets to Current Liabilities":
                                    "The taxpayer's short-term liquidity position "
                                    + direction_text + ".",
                            }

                            return explanations.get(
                                feature,
                                "This business indicator " + direction_text + "."
                            )

                        business_explanation_df["Business Interpretation"] = (
                            business_explanation_df.apply(
                                explain_business_driver,
                                axis=1
                            )
                        )

                        # ============================================================
                        # DISPLAY: BUSINESS EXPLANATION
                        # ============================================================

                        st.subheader("Business Explanation")

                        st.caption(
                            "This section translates the model explanation into business terms. "
                            "It focuses on the taxpayer's financial and operating indicators. "
                            "The SHAP contribution shows how each indicator influenced the "
                            "predicted class."
                        )

                        if not business_explanation_df.empty:

                            # Show the five strongest business drivers
                            business_top = business_explanation_df.head(5).copy()

                            for _, row in business_top.iterrows():

                                st.markdown(
                                    f"**{row['Business Indicator']}**  \n"
                                    f"{row['Business Interpretation']}  \n"
                                    f"SHAP contribution: `{row['SHAP Contribution']}`"
                                )

                            st.write("Business Driver Details")

                            # Auditor-friendly copy for presentation only.
                            # Underlying model features and SHAP calculations
                            # remain unchanged.
                            business_display_df = business_explanation_df[
                                [
                                    "Rank",
                                    "Business Indicator",
                                    "Feature Value",
                                    "SHAP Contribution",
                                    "Direction"
                                ]
                            ].copy()

                            business_display_df = business_display_df.rename(
                                columns={
                                    "Feature Value": "Observed Value",
                                    "SHAP Contribution": "Model Influence"
                                }
                            )

                            def format_observed_value(row):
                                indicator = row["Business Indicator"]
                                value = row["Observed Value"]

                                # Binary indicators already contain
                                # auditor-friendly Yes/No descriptions.
                                if indicator in [
                                    "Zero Profit Condition",
                                    "High Revenue / No Profit Condition"
                                ]:
                                    return value

                                try:
                                    numeric_value = float(value)

                                    # Monetary / absolute financial values
                                    if indicator in [
                                        "Revenue",
                                        "Total Profit"
                                    ]:
                                        return f"{numeric_value:,.2f}"

                                    # Financial ratios
                                    return f"{numeric_value:.4f}"

                                except (TypeError, ValueError):
                                    return value

                            business_display_df["Observed Value"] = (
                                business_display_df.apply(
                                    format_observed_value,
                                    axis=1
                                )
                            )

                            business_display_df["Model Influence"] = (
                                business_display_df["Model Influence"]
                                .apply(
                                    lambda x: (
                                        f"{float(x):.4f}"
                                        if x is not None
                                        else ""
                                    )
                                )
                            )

                            st.dataframe(
                                business_display_df,
                                use_container_width=True,
                                hide_index=True
                            )

                        else:
                            st.info(
                                "No financial or operating indicators were available "
                                "for this explanation."
                            )

                        # ============================================================
                        # DISPLAY: TECHNICAL SHAP EXPLANATION
                        # ============================================================

                        st.subheader("SHAP Explanation")

                        st.caption(
                            "SHAP (SHapley Additive exPlanations) is used to explain the "
                            "individual prediction. Positive SHAP contributions support the "
                            "predicted class, while negative contributions oppose it. "
                            "Values are normalized relative to the strongest contribution "
                            "for presentation."
                        )

                        st.dataframe(
                            top_features[
                                [
                                    "Rank",
                                    "Feature",
                                    "Feature Value",
                                    "SHAP Contribution",
                                    "Direction"
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                        # ============================================================
                        # INDUSTRY CONTEXT
                        # ============================================================

                        st.subheader("Industry Context")

                        st.caption(
                            "Division and subdivision variables are categorical industry "
                            "information encoded for the machine-learning model. They provide "
                            "industry context and should not be interpreted as financial ratios "
                            "or separate taxpayer behaviours."
                        )

                        industry_features = top_features[
                            top_features["Feature"].str.startswith(
                                (
                                    "Division_Description_",
                                    "SubDivision_Description_"
                                ),
                                na=False
                            )
                        ].copy()

                        if not industry_features.empty:

                            industry_features["Industry Category"] = (
                                industry_features["Feature"]
                                .str.replace(
                                    "SubDivision_Description_",
                                    "",
                                    regex=False
                                )
                                .str.replace(
                                    "Division_Description_",
                                    "",
                                    regex=False
                                )
                            )

                            st.dataframe(
                                industry_features[
                                    [
                                        "Industry Category",
                                        "Feature Value",
                                        "SHAP Contribution",
                                        "Direction"
                                    ]
                                ],
                                use_container_width=True,
                                hide_index=True
                            )

                        else:
                            st.info(
                                "No industry variables were among the ten strongest "
                                "SHAP contributions for this prediction."
                            )

                        # ============================================================
                        # VISUALIZATION: TOP SHAP CONTRIBUTIONS
                        # ============================================================

                        st.subheader("Technical SHAP Feature Contributions")

                        st.caption(
                            "This chart shows the ten strongest SHAP contributions to the "
                            "prediction. Values to the right of zero support the predicted "
                            "class; values to the left oppose it."
                        )

                        chart_df = top_features[
                            ["Feature", "SHAP Contribution"]
                        ].copy()

                        # Clean encoded category prefixes for display only.
                        chart_df["Display Feature"] = (
                            chart_df["Feature"]
                            .str.replace(
                                "SubDivision_Description_",
                                "",
                                regex=False
                            )
                            .str.replace(
                                "Division_Description_",
                                "",
                                regex=False
                            )
                        )

                        # Shorten labels for chart readability while preserving the
                        # complete feature names in the SHAP table above.
                        chart_df["Display Feature"] = chart_df[
                            "Display Feature"
                        ].apply(
                            lambda x: x if len(str(x)) <= 45 else str(x)[:42] + "..."
                        )

                        chart_df = chart_df.sort_values(
                            "SHAP Contribution",
                            ascending=True
                        )

                        st.bar_chart(
                            chart_df.set_index(
                                "Display Feature"
                            )["SHAP Contribution"],
                            horizontal=True
                        )


                        # SHAP VALUE SUMMARY
                        # ============================================================




                        # ============================================================
                        # MODEL INFORMATION
                        # ============================================================




                        # ============================================================
                        # FEATURE ORDER CHECK
                        # ============================================================







                        # ============================================================
                        # EXPORT ASSESSMENT
                        # ============================================================

                        st.markdown("---")
                        st.subheader("Export Assessment")

                        st.caption(
                            "Download the selected taxpayer's risk assessment for "
                            "review, documentation, or further analysis."
                        )

                        # ------------------------------------------------------------
                        # Risk classification
                        # ------------------------------------------------------------

                        export_risk_class = str(predicted_class)

                        try:
                            export_risk_class = str(
                                loaded_encoder.inverse_transform(
                                    [int(predicted_class)]
                                )[0]
                            )
                        except Exception:
                            pass


                        # ------------------------------------------------------------
                        # Prediction confidence
                        # ------------------------------------------------------------

                        try:
                            export_confidence = float(
                                predicted_probabilities[0][int(predicted_class)]
                            ) * 100

                        except Exception:

                            try:
                                export_confidence = float(
                                    max(predicted_probabilities[0])
                                ) * 100

                            except Exception:
                                export_confidence = None


                        confidence_text = (
                            f"{export_confidence:.2f}%"
                            if export_confidence is not None
                            else "Not available"
                        )


                        # ------------------------------------------------------------
                        # Business driver table
                        # ------------------------------------------------------------

                        export_drivers_df = business_explanation_df.copy()

                        preferred_export_columns = [
                            "Business Indicator",
                            "Feature Value",
                            "SHAP Contribution",
                            "Direction",
                        ]

                        available_export_columns = [
                            column
                            for column in preferred_export_columns
                            if column in export_drivers_df.columns
                        ]

                        if available_export_columns:

                            export_drivers_df = export_drivers_df[
                                available_export_columns
                            ]


                        # ------------------------------------------------------------
                        # CSV
                        # ------------------------------------------------------------

                        csv_export_df = export_drivers_df.copy()

                        csv_export_df.insert(
                            0,
                            "Taxpayer",
                            str(selected_taxpayer)
                        )

                        csv_export_df.insert(
                            1,
                            "Risk Classification",
                            export_risk_class
                        )

                        csv_export_df.insert(
                            2,
                            "Model Confidence (%)",
                            (
                                round(export_confidence, 2)
                                if export_confidence is not None
                                else ""
                            )
                        )

                        csv_data = csv_export_df.to_csv(
                            index=False
                        ).encode("utf-8")


                        # ------------------------------------------------------------
                        # HTML REPORT
                        # ------------------------------------------------------------

                        driver_table_html = export_drivers_df.to_html(
                            index=False,
                            border=0,
                            justify="left"
                        )

                        html_report = f"""
                        <!DOCTYPE html>

                        <html>

                        <head>

                        <meta charset="UTF-8">

                        <title>Taxpayer Risk Assessment</title>

                        <style>

                        body {{
                            font-family: Arial, sans-serif;
                            margin: 40px;
                            color: #222;
                            line-height: 1.5;
                        }}

                        h1 {{
                            color: #9b1c1c;
                            border-bottom: 2px solid #9b1c1c;
                            padding-bottom: 10px;
                        }}

                        h2 {{
                            margin-top: 30px;
                        }}

                        .summary {{
                            background: #f4f4f4;
                            padding: 18px;
                            border-radius: 6px;
                            margin-bottom: 25px;
                        }}

                        table {{
                            border-collapse: collapse;
                            width: 100%;
                            margin-top: 15px;
                        }}

                        th, td {{
                            border: 1px solid #cccccc;
                            padding: 8px;
                            text-align: left;
                        }}

                        th {{
                            background: #eeeeee;
                        }}

                        .note {{
                            margin-top: 30px;
                            padding: 15px;
                            background: #f8f8f8;
                            border-left: 4px solid #777;
                        }}

                        </style>

                        </head>

                        <body>

                        <h1>Taxpayer Risk Assessment</h1>

                        <div class="summary">

                        <strong>Taxpayer:</strong>
                        {selected_taxpayer}

                        <br>

                        <strong>Risk Classification:</strong>
                        {export_risk_class}

                        <br>

                        <strong>Model Confidence:</strong>
                        {confidence_text}

                        </div>

                        <h2>Business Risk Drivers</h2>

                        <p>
                        The following indicators represent the principal business
                        factors used to explain the model's assessment for this taxpayer.
                        </p>

                        {driver_table_html}

                        <div class="note">

                        <strong>Important:</strong>

                        This model output is intended to support risk assessment and
                        prioritisation. It should be considered alongside other taxpayer
                        information, available evidence, and professional judgement.

                        </div>

                        </body>

                        </html>
                        """


                        # ------------------------------------------------------------
                        # Safe file name
                        # ------------------------------------------------------------

                        safe_taxpayer_name = "".join(
                            character if character.isalnum() else "_"
                            for character in str(selected_taxpayer)
                        ).strip("_")


                        # ------------------------------------------------------------
                        # Download buttons
                        # ------------------------------------------------------------

                        export_col1, export_col2 = st.columns(2)

                        with export_col1:

                            st.download_button(
                                label="Download Assessment Report",
                                data=html_report.encode("utf-8"),
                                file_name=(
                                    f"{safe_taxpayer_name}_risk_assessment.html"
                                ),
                                mime="text/html",
                                use_container_width=True,
                            )


                        with export_col2:

                            st.download_button(
                                label="Download Assessment Data (CSV)",
                                data=csv_data,
                                file_name=(
                                    f"{safe_taxpayer_name}_risk_assessment.csv"
                                ),
                                mime="text/csv",
                                use_container_width=True,
                            )

                    else:

                        st.error("X_encoded not found in Session State")


                else:

                    st.info(
                        "A different file has been uploaded. "
                        "Click Run Risk Prediction to analyse "
                        "the current dataset."
                    )


    # --------------------------------------------------------
    # FILE READING ERROR
    # --------------------------------------------------------

    except Exception as error:

        st.error(
            "The uploaded file could not be processed."
        )

        st.write(
            "Error details:"
        )

        st.code(
            str(error)
        )
