import os
from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

from processing.preprocessing import preprocess_data

model_path = os.path.join("backend", "models", "voting_classifier.joblib")
model = load(model_path)
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict the credit score category based on input features.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request."}), 400

        file = request.files["file"]
        number_of_samples = int(request.form.get("number_of_samples", 1))
        if file.filename == "":
            return jsonify({"error": "No file selected for uploading."}), 400

        df = pd.read_csv(file)

        if number_of_samples > len(df):
            return (
                jsonify(
                    {
                        "error": f"Number of samples exceeds the dataset size ({len(df)})."
                    }
                ),
                400,
            )

        sampled_df = df.sample(number_of_samples)

        X_preprocessed, _ = preprocess_data(sampled_df)

        if number_of_samples == 1:
            X_preprocessed = X_preprocessed.reshape(1, -1)

        predictions = model.predict(X_preprocessed)
        prediction_prob = model.predict_proba(X_preprocessed).tolist()

        response = {
            "samples": sampled_df.to_dict(orient="records"),
            "predictions": predictions.tolist(),
            "probabilities": prediction_prob,
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
