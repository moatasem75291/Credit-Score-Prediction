from .processing.preprocessing import preprocess_data
import os
from joblib import load


model_path = os.path.join("backend", "models", "voting_classifier.joblib")
model = load(model_path)


def prediction(df, number_of_samples):
    try:
        if number_of_samples > len(df):
            return {
                "error": f"Number of samples exceeds the dataset size ({len(df)})."
            }, 400

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
        return response, 200
    except Exception as e:
        return {"error": str(e)}, 500
