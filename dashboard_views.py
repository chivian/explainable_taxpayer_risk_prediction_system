
# ============================================================
# DASHBOARD VIEWS
# Taxpayer Risk Prediction System
# ============================================================

import pandas as pd
import streamlit as st
import altair as alt


# ============================================================
# HELPER FUNCTION: FIND FIRST AVAILABLE COLUMN
# ============================================================

def find_first_available_column(dataframe, candidates):

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


# ============================================================
# MEDIUM-RISK INTELLIGENCE
# ============================================================

def render_medium_risk_intelligence(prediction_results):

    st.markdown("---")

    st.header(
        "Medium-Risk Intelligence"
    )

    st.write(
        "This section focuses on taxpayers classified as "
        "Medium Risk. It supports monitoring, prioritisation, "
        "and identification of cases that may require closer "
        "compliance attention."
    )


    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMN
    # --------------------------------------------------------

    if (
        "Predicted_Risk_Class"
        not in prediction_results.columns
    ):

        st.warning(
            "Medium-Risk Intelligence cannot be displayed "
            "because Predicted_Risk_Class is unavailable."
        )

        return


    # --------------------------------------------------------
    # FILTER MEDIUM-RISK RECORDS
    # --------------------------------------------------------

    medium_risk_df = (

        prediction_results[

            prediction_results[
                "Predicted_Risk_Class"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            == "medium"

        ]

        .copy()
    )


    # ========================================================
    # KPI SECTION
    # ========================================================

    kpi1, kpi2, kpi3 = st.columns(3)


    with kpi1:

        st.metric(
            "Medium-Risk Records",
            f"{len(medium_risk_df):,}"
        )


    with kpi2:

        if "TIN" in medium_risk_df.columns:

            unique_medium_tins = (

                medium_risk_df[
                    "TIN"
                ]

                .dropna()

                .astype(str)

                .nunique()
            )

            st.metric(
                "Unique Medium-Risk TINs",
                f"{unique_medium_tins:,}"
            )

        else:

            st.metric(
                "Unique Medium-Risk TINs",
                "N/A"
            )


    with kpi3:

        confidence_column = (
            find_first_available_column(
                medium_risk_df,
                [
                    "Prediction_Confidence",
                    "Confidence",
                    "prediction_confidence"
                ]
            )
        )


        if (
            confidence_column is not None
            and
            not medium_risk_df.empty
        ):

            confidence_series = pd.to_numeric(
                medium_risk_df[
                    confidence_column
                ],
                errors="coerce"
            )


            average_confidence = (
                confidence_series.mean()
            )


            if pd.notna(average_confidence):

                if average_confidence <= 1:

                    average_confidence = (
                        average_confidence * 100
                    )


                st.metric(
                    "Average Medium-Risk Confidence",
                    f"{average_confidence:.2f}%"
                )

            else:

                st.metric(
                    "Average Medium-Risk Confidence",
                    "N/A"
                )

        else:

            st.metric(
                "Average Medium-Risk Confidence",
                "N/A"
            )


    # ========================================================
    # HANDLE EMPTY MEDIUM-RISK RESULTS
    # ========================================================

    if medium_risk_df.empty:

        st.info(
            "No Medium-Risk records were identified "
            "in the current prediction results."
        )

        return


    # ========================================================
    # SECTOR DISTRIBUTION
    # ========================================================

    st.subheader(
        "Medium-Risk Records by Sector"
    )


    sector_column = (
        find_first_available_column(
            medium_risk_df,
            [
                "Division_Description",
                "Sector",
                "sector"
            ]
        )
    )


    if sector_column is not None:

        medium_sector_distribution = (

            medium_risk_df[
                sector_column
            ]

            .fillna(
                "Unknown"
            )

            .astype(str)

            .value_counts()

            .rename_axis(
                "Sector"
            )

            .reset_index(
                name="Medium-Risk Records"
            )
        )


        medium_sector_chart = (

            alt.Chart(
                medium_sector_distribution
            )

            .mark_bar(
                color="#D62828",
                cornerRadiusEnd=4
            )

            .encode(

                y=alt.Y(
                    "Sector:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(
                        labelLimit=500
                    )
                ),

                x=alt.X(
                    "Medium-Risk Records:Q",
                    title="Number of Medium-Risk Records",
                    axis=alt.Axis(
                        tickMinStep=1
                    )
                ),

                tooltip=[

                    alt.Tooltip(
                        "Sector:N",
                        title="Sector"
                    ),

                    alt.Tooltip(
                        "Medium-Risk Records:Q",
                        title="Records",
                        format=","
                    )
                ]
            )

            .properties(
                height=max(
                    300,
                    len(
                        medium_sector_distribution
                    ) * 38
                )
            )
        )


        st.altair_chart(
            medium_sector_chart,
            use_container_width=True
        )


        st.dataframe(
            medium_sector_distribution,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "Sector information is not available "
            "for the Medium-Risk records."
        )


    # ========================================================
    # IDENTIFY OPTIONAL COLUMNS
    # ========================================================

    year_column = (
        find_first_available_column(
            medium_risk_df,
            [
                "Assessment Year",
                "Assessment_Year",
                "assessment_year",
                "Year of Assessment",
                "Year"
            ]
        )
    )


    assessment_type_column = (
        find_first_available_column(
            medium_risk_df,
            [
                "Assessment Type",
                "Assessment_Type",
                "assessment_type",
                "AssessmentType"
            ]
        )
    )


    high_probability_column = (
        find_first_available_column(
            medium_risk_df,
            [
                "Probability_High",
                "High_Risk_Probability",
                "High Risk Probability"
            ]
        )
    )


    medium_probability_column = (
        find_first_available_column(
            medium_risk_df,
            [
                "Probability_Medium",
                "Medium_Risk_Probability",
                "Medium Risk Probability"
            ]
        )
    )


    confidence_column = (
        find_first_available_column(
            medium_risk_df,
            [
                "Prediction_Confidence",
                "Confidence",
                "prediction_confidence"
            ]
        )
    )


    # ========================================================
    # BUILD REVIEW TABLE
    # ========================================================

    st.subheader(
        "Medium-Risk Taxpayer Review Table"
    )


    review_columns = []


    candidate_columns = [

        "TIN",
        "custName",
        "Division_Description",
        "SubDivision_Description",
        "Revenue",
        "Total Profit",
        year_column,
        assessment_type_column,
        high_probability_column,
        medium_probability_column,
        confidence_column
    ]


    for column in candidate_columns:

        if (
            column is not None
            and
            column in medium_risk_df.columns
            and
            column not in review_columns
        ):

            review_columns.append(
                column
            )


    medium_review = (

        medium_risk_df[
            review_columns
        ]

        .copy()
    )


    # --------------------------------------------------------
    # CONVERT PROBABILITY COLUMNS TO DISPLAY PERCENTAGES
    # --------------------------------------------------------

    probability_columns = [

        high_probability_column,
        medium_probability_column,
        confidence_column
    ]


    for column in probability_columns:

        if (
            column is not None
            and
            column in medium_review.columns
        ):

            numeric_series = pd.to_numeric(
                medium_review[column],
                errors="coerce"
            )


            if numeric_series.max() <= 1:

                numeric_series = (
                    numeric_series * 100
                )


            medium_review[column] = (
                numeric_series.round(2)
            )


    # --------------------------------------------------------
    # PRIORITISE CASES CLOSEST TO HIGH-RISK
    # --------------------------------------------------------

    if (
        high_probability_column is not None
        and
        high_probability_column
        in medium_review.columns
    ):

        medium_review = (

            medium_review.sort_values(
                by=high_probability_column,
                ascending=False
            )
        )


    elif (
        confidence_column is not None
        and
        confidence_column
        in medium_review.columns
    ):

        medium_review = (

            medium_review.sort_values(
                by=confidence_column,
                ascending=False
            )
        )


    # ========================================================
    # FRIENDLY COLUMN NAMES
    # ========================================================

    rename_map = {

        "custName":
            "Taxpayer Name",

        "Division_Description":
            "Sector",

        "SubDivision_Description":
            "Subsector"
    }


    if high_probability_column is not None:

        rename_map[
            high_probability_column
        ] = "High-Risk Probability (%)"


    if medium_probability_column is not None:

        rename_map[
            medium_probability_column
        ] = "Medium-Risk Probability (%)"


    if confidence_column is not None:

        rename_map[
            confidence_column
        ] = "Confidence (%)"


    medium_display = (

        medium_review.rename(
            columns=rename_map
        )
    )


    # ========================================================
    # DISPLAY TABLE
    # ========================================================

    st.caption(
        "Cases are prioritised by High-Risk probability where "
        "that score is available. This helps identify Medium-"
        "Risk taxpayers that may be closest to the High-Risk "
        "decision boundary."
    )


    st.dataframe(
        medium_display,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # EXPANDED DETAILS
    # ========================================================

    with st.expander(
        "View Full Medium-Risk Record Details"
    ):

        st.write(
            "This view contains the available record-level "
            "details for taxpayers classified as Medium Risk."
        )


        st.dataframe(
            medium_risk_df,
            use_container_width=True,
            hide_index=True
        )

#st.success("Finished Medium Risk section")

# ============================================================
# LOW-RISK INTELLIGENCE
# ============================================================

def render_low_risk_intelligence(prediction_results):

    st.markdown("---")

    st.header(
        "Low-Risk Intelligence"
    )

    st.write(
        "This section focuses on taxpayers classified as "
        "Low Risk. These taxpayers generally demonstrate "
        "lower predicted compliance risk and support routine "
        "monitoring, taxpayer service planning, and analysis "
        "of sustained compliance patterns."
    )


    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMN
    # --------------------------------------------------------

    if (
        "Predicted_Risk_Class"
        not in prediction_results.columns
    ):

        st.warning(
            "Low-Risk Intelligence cannot be displayed "
            "because Predicted_Risk_Class is unavailable."
        )

        return


    # --------------------------------------------------------
    # FILTER LOW-RISK RECORDS
    # --------------------------------------------------------

    low_risk_df = (

        prediction_results[

            prediction_results[
                "Predicted_Risk_Class"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            == "low"

        ]

        .copy()
    )


    # ========================================================
    # KPI SECTION
    # ========================================================

    kpi1, kpi2, kpi3 = st.columns(3)


    with kpi1:

        st.metric(
            "Low-Risk Records",
            f"{len(low_risk_df):,}"
        )


    with kpi2:

        if "TIN" in low_risk_df.columns:

            unique_low_tins = (

                low_risk_df[
                    "TIN"
                ]

                .dropna()

                .astype(str)

                .nunique()
            )

            st.metric(
                "Unique Low-Risk TINs",
                f"{unique_low_tins:,}"
            )

        else:

            st.metric(
                "Unique Low-Risk TINs",
                "N/A"
            )


    with kpi3:

        confidence_column = (
            find_first_available_column(
                low_risk_df,
                [
                    "Prediction_Confidence",
                    "Confidence",
                    "prediction_confidence"
                ]
            )
        )


        if (
            confidence_column is not None
            and
            not low_risk_df.empty
        ):

            confidence_series = pd.to_numeric(
                low_risk_df[
                    confidence_column
                ],
                errors="coerce"
            )


            average_confidence = (
                confidence_series.mean()
            )


            if pd.notna(average_confidence):

                if average_confidence <= 1:

                    average_confidence = (
                        average_confidence * 100
                    )


                st.metric(
                    "Average Low-Risk Confidence",
                    f"{average_confidence:.2f}%"
                )

            else:

                st.metric(
                    "Average Low-Risk Confidence",
                    "N/A"
                )

        else:

            st.metric(
                "Average Low-Risk Confidence",
                "N/A"
            )


    # ========================================================
    # HANDLE EMPTY LOW-RISK RESULTS
    # ========================================================

    if low_risk_df.empty:

        st.info(
            "No Low-Risk records were identified "
            "in the current prediction results."
        )

        return


    # ========================================================
    # SECTOR DISTRIBUTION
    # ========================================================

    st.subheader(
        "Low-Risk Records by Sector"
    )


    sector_column = (
        find_first_available_column(
            low_risk_df,
            [
                "Division_Description",
                "Sector",
                "sector"
            ]
        )
    )


    if sector_column is not None:

        low_sector_distribution = (

            low_risk_df[
                sector_column
            ]

            .fillna(
                "Unknown"
            )

            .astype(str)

            .value_counts()

            .rename_axis(
                "Sector"
            )

            .reset_index(
                name="Low-Risk Records"
            )
        )


        low_sector_chart = (

            alt.Chart(
                low_sector_distribution
            )

            .mark_bar(
                color="#D62828",
                cornerRadiusEnd=4
            )

            .encode(

                y=alt.Y(
                    "Sector:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(
                        labelLimit=500
                    )
                ),

                x=alt.X(
                    "Low-Risk Records:Q",
                    title="Number of Low-Risk Records",
                    axis=alt.Axis(
                        tickMinStep=1
                    )
                ),

                tooltip=[

                    alt.Tooltip(
                        "Sector:N",
                        title="Sector"
                    ),

                    alt.Tooltip(
                        "Low-Risk Records:Q",
                        title="Records",
                        format=","
                    )
                ]
            )

            .properties(
                height=max(
                    300,
                    len(
                        low_sector_distribution
                    ) * 38
                )
            )
        )


        st.altair_chart(
            low_sector_chart,
            use_container_width=True
        )


        st.dataframe(
            low_sector_distribution,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "Sector information is not available "
            "for the Low-Risk records."
        )


    # ========================================================
    # IDENTIFY OPTIONAL COLUMNS
    # ========================================================

    year_column = (
        find_first_available_column(
            low_risk_df,
            [
                "Assessment Year",
                "Assessment_Year",
                "assessment_year",
                "Year of Assessment",
                "Year"
            ]
        )
    )


    assessment_type_column = (
        find_first_available_column(
            low_risk_df,
            [
                "Assessment Type",
                "Assessment_Type",
                "assessment_type",
                "AssessmentType"
            ]
        )
    )


    low_probability_column = (
        find_first_available_column(
            low_risk_df,
            [
                "Probability_Low",
                "Low_Risk_Probability",
                "Low Risk Probability"
            ]
        )
    )


    confidence_column = (
        find_first_available_column(
            low_risk_df,
            [
                "Prediction_Confidence",
                "Confidence",
                "prediction_confidence"
            ]
        )
    )


    # ========================================================
    # BUILD REVIEW TABLE
    # ========================================================

    st.subheader(
        "Low-Risk Taxpayer Review Table"
    )


    review_columns = []


    candidate_columns = [

        "TIN",
        "custName",
        "Division_Description",
        "SubDivision_Description",
        "Revenue",
        "Total Profit",
        year_column,
        assessment_type_column,
        low_probability_column,
        confidence_column
    ]


    for column in candidate_columns:

        if (
            column is not None
            and
            column in low_risk_df.columns
            and
            column not in review_columns
        ):

            review_columns.append(
                column
            )


    low_review = (

        low_risk_df[
            review_columns
        ]

        .copy()
    )


    # --------------------------------------------------------
    # CONVERT PROBABILITY COLUMNS TO DISPLAY PERCENTAGES
    # --------------------------------------------------------

    probability_columns = [

        low_probability_column,
        confidence_column
    ]


    for column in probability_columns:

        if (
            column is not None
            and
            column in low_review.columns
        ):

            numeric_series = pd.to_numeric(
                low_review[column],
                errors="coerce"
            )


            if (
                numeric_series.notna().any()
                and
                numeric_series.max() <= 1
            ):

                numeric_series = (
                    numeric_series * 100
                )


            low_review[column] = (
                numeric_series.round(2)
            )


    # --------------------------------------------------------
    # SORT BY CONFIDENCE
    # --------------------------------------------------------

    if (
        confidence_column is not None
        and
        confidence_column
        in low_review.columns
    ):

        low_review = (

            low_review.sort_values(
                by=confidence_column,
                ascending=False
            )
        )


    elif (
        low_probability_column is not None
        and
        low_probability_column
        in low_review.columns
    ):

        low_review = (

            low_review.sort_values(
                by=low_probability_column,
                ascending=False
            )
        )


    # ========================================================
    # FRIENDLY COLUMN NAMES
    # ========================================================

    rename_map = {

        "custName":
            "Taxpayer Name",

        "Division_Description":
            "Sector",

        "SubDivision_Description":
            "Subsector"
    }


    if low_probability_column is not None:

        rename_map[
            low_probability_column
        ] = "Low-Risk Probability (%)"


    if confidence_column is not None:

        rename_map[
            confidence_column
        ] = "Confidence (%)"


    low_display = (

        low_review.rename(
            columns=rename_map
        )
    )


    # ========================================================
    # DISPLAY TABLE
    # ========================================================

    st.caption(
        "These records represent taxpayers classified as "
        "Low Risk and are shown for routine monitoring, "
        "taxpayer service planning, and general compliance "
        "management. Records are ordered by prediction "
        "confidence where that score is available."
    )


    st.dataframe(
        low_display,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # EXPANDED DETAILS
    # ========================================================

    with st.expander(
        "View Full Low-Risk Record Details"
    ):

        st.write(
            "This view contains the available record-level "
            "details for taxpayers classified as Low Risk."
        )


        st.dataframe(
            low_risk_df,
            use_container_width=True,
            hide_index=True
        )

