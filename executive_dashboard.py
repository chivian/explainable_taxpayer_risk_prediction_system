
import streamlit as st
import pandas as pd


def render_executive_dashboard(prediction_results):

    st.markdown("---")

    st.header("Executive Dashboard")

    st.write(
        """
        This dashboard provides an executive summary of
        taxpayer risk prediction results.
        """
    )

    if prediction_results is None or prediction_results.empty:

        st.warning("No prediction results available.")

        return

    total_records = len(prediction_results)

    high_count = (
        prediction_results["Predicted_Risk_Class"]
        .eq("High")
        .sum()
    )

    medium_count = (
        prediction_results["Predicted_Risk_Class"]
        .eq("Medium")
        .sum()
    )

    low_count = (
        prediction_results["Predicted_Risk_Class"]
        .eq("Low")
        .sum()
    )

    confidence = prediction_results[
        "Prediction_Confidence"
    ].mean()

    if confidence <= 1:
        confidence *= 100

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Total Taxpayers", f"{total_records:,}")

    with c2:
        st.metric("High Risk", f"{high_count:,}")

    with c3:
        st.metric("Medium Risk", f"{medium_count:,}")

    with c4:
        st.metric("Low Risk", f"{low_count:,}")

    with c5:
        st.metric(
            "Average Confidence",
            f"{confidence:.2f}%"
        )
