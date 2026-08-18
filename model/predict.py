import os
import joblib
import pandas as pd

# Anchor path to this file's location, not the current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "best_model_xgboost.pkl"))


def predict_output(input_df: pd.DataFrame):
    """
    Takes an already-encoded DataFrame (matching model_columns)
    and returns the predicted charge.
    """
    output = model.predict(input_df)[0]
    return output