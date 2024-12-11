import logging
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import dotenv
import os
import json

from frontend.integration import prediction

dotenv.load_dotenv()

API_URL = os.getenv("API_URL")
CREDIT_SCORE_MAPPING = {"Standard": 0, "Poor": 1, "Good": 2}
CREDIT_SCORE_MAPPING_REV = {v: k for k, v in CREDIT_SCORE_MAPPING.items()}
CREDIT_SCORE_COLORS = {0: "blue", 1: "red", 2: "green"}


def run():
    logging.basicConfig(
        filename="app.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    st.title("📊 Credit Score Prediction Results")
    st.markdown(
        """
        Upload your dataset and visualize the prediction results with charts that 
        show each sample and the corresponding credit score prediction.
        """
    )

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    sample_count = st.number_input(
        "Number of samples to predict:",
        min_value=1,
        step=1,
        help="Select how many samples to predict from the dataset.",
    )

    if st.button("Predict and Visualize"):
        if uploaded_file:
            try:
                logging.info("File uploaded successfully. Processing...")

                data = pd.read_csv(uploaded_file)

                if data.empty:
                    st.error(
                        "The uploaded file is empty. Please provide a valid dataset."
                    )
                    logging.warning("Uploaded file is empty.")
                else:
                    with st.spinner("Processing predictions..."):
                        files = {"file": uploaded_file.getvalue()}
                        payload = {"number_of_samples": sample_count}
                        # response = requests.post(API_URL, files=files, data=payload)
                        response = prediction(data, sample_count)
                        if response[1] == 200:
                            logging.info("Prediction request successful.")
                            result = response[0]

                            sampled_df = pd.DataFrame(result["samples"])

                            predictions = pd.DataFrame(
                                {
                                    "Sample Index": sampled_df.index + 1,
                                    "Credit Score": result["predictions"],
                                    "Credit Score (Numeric)": [
                                        CREDIT_SCORE_MAPPING_REV[pred]
                                        for pred in result["predictions"]
                                    ],
                                }
                            )
                            probabilities = pd.DataFrame(
                                result["probabilities"],
                                columns=["Poor", "Standard", "Good"],
                            )

                            combined_data = pd.concat(
                                [
                                    sampled_df.reset_index(drop=True),
                                    predictions,
                                    probabilities,
                                ],
                                axis=1,
                            )

                            st.success(
                                "Predictions and probabilities successfully retrieved!"
                            )
                            st.write("### Sampled Data with Predictions")
                            st.dataframe(combined_data)

                            fig_scatter = px.scatter(
                                predictions,
                                x="Sample Index",
                                y="Credit Score (Numeric)",
                                color="Credit Score (Numeric)",
                                color_discrete_map=CREDIT_SCORE_COLORS,
                                labels={"Credit Score (Numeric)": "Credit Score"},
                                title="Scatter Plot of Predictions",
                            )
                            fig_scatter.update_traces(marker=dict(size=12))
                            st.plotly_chart(fig_scatter, use_container_width=True)

                            fig_bar = px.bar(
                                probabilities,
                                x=probabilities.index + 1,
                                y=["Poor", "Standard", "Good"],
                                barmode="group",
                                labels={"x": "Sample Index", "value": "Probability"},
                                title="Probabilities for Each Sample",
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                        else:
                            error_message = response[0].get(
                                "error", "Unknown error occurred."
                            )
                            st.error(f"Error: {error_message}.")
                            logging.error(f"Prediction request failed: {error_message}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logging.exception("An unexpected error occurred.")
        else:
            st.warning("Please upload a dataset before clicking Predict and Visualize.")
            logging.warning(
                "No file uploaded when Predict and Visualize button clicked."
            )
