# Credit Score Prediction with Machine Learning

[CLICK ON ME TO TEST THE MODEL](https://credit-score-moatasem.streamlit.app/)

This project aims to predict credit scores based on individual financial data using machine learning models. The backend is a Flask-based API that handles prediction logic, while the frontend is a Streamlit application that visualizes the results and provides an interactive user experience.

## Project Overview

- **Modeling**: A machine learning model is used to predict credit scores (Standard, Poor, Good) based on a dataset containing features related to customer financial and credit information.
- **Frontend**: A Streamlit application that allows users to upload datasets, interact with the prediction process, and visualize the results using charts (scatter, bar, and pie).
- **Backend**: A Flask API that processes the uploaded data, runs predictions using the machine learning model, and returns predictions with probabilities.
- **Deployment**: The project can be deployed locally for development purposes, and is also ready for cloud deployment using services like AWS or Heroku.

## Project Structure
``` CSS
Credit Score Prediction/
├── .venv
├── backend/
│   ├── models/ # Trained machine learning model for predictions
│   │       └── voting_classifier.joblib
│   ├── processing/ # Data preprocessing functions
│   │       └── pipeline.py
│   │       └── preprocessing.py
│   │       └── transformers.py
│   └── app.py # Flask API for predictions
├── frontend/
│   ├── pages/ # Trained machine learning model for predictions
│   │       └── EDA.py
│   │       └── Predictions.py
│   └── main.py # Run Streamlit app
├── data/
│     └── train.csv
│     └── test.csv
├── experimental/
│     └── credit_card_status.ipynb
│     └── credit-card-status.ipynb
├── .env
├── app.log
├── README.md
├── requirements.txt
```

## Modeling

1. **Data Preprocessing**:
    - The dataset contains features related to customer financial data. The preprocessing step involves cleaning the data, handling missing values, encoding categorical variables, and scaling numeric features.
    - Preprocessed data is then used to train a machine learning model that predicts credit score categories (Standard, Poor, Good).

2. **Machine Learning Model**:
    - The model used in this project is an ensemble classifier, specifically a voting classifier that combines predictions from multiple models to improve performance.
    - The model is trained and saved as `voting_classifier.joblib` inside the `/backend/models` folder.

3. **Training the Model**:
    - The model is trained on a labeled dataset that includes features such as income, credit history, loan amount, etc.
    - After training, the model is serialized and saved as a `.joblib` file for use in the Flask API.

## Frontend

The frontend of this project is built using **Streamlit**. It serves as an interface for users to upload datasets, trigger predictions, and visualize the results.

### Features:
- **File Upload**: Users can upload a CSV dataset containing the necessary features for prediction.
- **Select Number of Samples**: Users can specify how many samples they want to predict from the dataset.
- **Visualization**:
  - **Scatter Plot**: Displays a scatter plot of the predictions with different colors based on the predicted credit score.
  - **Bar Chart**: Shows the probabilities of each credit score category for each sample.
  - **Pie Chart**: Displays the distribution of predicted credit scores across all samples.

### Running the Frontend:
1. Install the required dependencies for the frontend:
   ```bash
   pip install -r requirements.txt
   ```
2. Navigate to the /frontend folder and run the Streamlit app
   ```bash
   cd frontend
   streamlit run main.py
   ```
## Backend

The project's backend is a Flask API that accepts CSV data, processes it, and returns predictions and probabilities.

### API Endpoint: /predict
- **Method**: POST
- **Parameters**:
  - `file`: The dataset file (CSV format).
  - `number_of_samples`: The number of samples to predict (optional, defaults to 1).
- **Response**: 
  The API returns a JSON object containing:
  - `samples`: The sampled data from the uploaded file.
  - `predictions`: The predicted credit scores for each sample.
  - `probabilities`: The probabilities for each credit score category (Standard, Poor, Good).

### Running the Backend:
1. Install the required dependencies for the backend:

    ```bash
    pip install -r requirements.txt
    ```

2. Navigate to the `/backend` folder and run the Flask app:

    ```bash
    cd backend
    python app.py
    ```

The API will be available at `http://127.0.0.1:5000/predict`.

### API Error Handling:
- If the uploaded file is missing or empty, the API will return an error message.
- If an issue occurs during prediction, the API will return an error with a detailed message.

## Deployment

To deploy this project to the cloud (e.g., on AWS or Heroku), follow these steps:

### Backend Deployment:
- For cloud deployment, set up a virtual environment, install dependencies, and configure the API server.
- If using Heroku, push the Flask API to a new Heroku app and deploy it using git and the Heroku CLI.

### Frontend Deployment:
- The Streamlit app can also be deployed to platforms like Heroku or Streamlit Cloud.
- Push the frontend code to your chosen platform and configure it to interact with the deployed Flask API.

## Installation

### Clone the repository:

```bash
git clone https://github.com/moatasem75291/Credit-Score-Prediction.git
cd credit-score-prediction
```

### Set up a virtual environment for both frontend and backend
``` bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

## Contributing
Contributions are welcome! If you want to contribute to the project, please fork the repository, create a branch, and submit a pull request with your changes.
