import streamlit as st

st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("💳 Credit Score Prediction System")
st.markdown(
    """
    Welcome to the Credit Score Prediction System! This application allows you to:
    - Upload your dataset and explore it using **Exploratory Data Analysis (EDA)** tools.
    - Predict credit score categories for customers based on their financial and credit information.
    - Gain insights into missing values, outliers, and patterns in your dataset.
    - Handle predictions efficiently through a backend Flask API.

    ### 🔧 Features:
    - **EDA**: Explore patterns, outliers, and relationships in your data.
    - **Predictions**: Upload a dataset, select the number of samples, and receive predictions.
    - **Error Handling**: Comprehensive feedback on errors to guide corrections.

    ---
    Navigate through the pages using the sidebar to begin!
    """
)
