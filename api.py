import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

from schema.user_input import UserInput
from model.predict import predict_output

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

model_columns = joblib.load(os.path.join(BASE_DIR, "model", "model_columns.pkl"))


def encode_input(data: UserInput, columns: list) -> pd.DataFrame:
    # Start with every training column set to 0
    row = {col: 0 for col in columns}

    row["age"] = data.age
    row["bmi"] = data.bmi
    row["children"] = data.children
    row["sex"] = 1 if data.sex == "female" else 0
    row["smoker"] = 1 if data.smoker == "yes" else 0

    # One-hot region — 'northeast' was the dropped baseline during training,
    # so it stays all-zeros; other regions flip their own column to 1
    region_col = f"region_{data.region}"
    if region_col in row:
        row[region_col] = 1

    return pd.DataFrame([row], columns=columns)


@app.get('/')
def home():
    return {'message': 'insurance premium api'}


@app.get('/health')
def health_check():
    return {
        'status': 'ok',
        'model_loaded': predict_output is not None
    }


@app.post("/predict")
def predict_premium(data: UserInput):
    input_df = encode_input(data, model_columns)
    prediction = predict_output(input_df)

    return JSONResponse(
        status_code=200,
        content={"predicted_charges": float(prediction)}
    )