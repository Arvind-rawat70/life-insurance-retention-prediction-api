from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import joblib
import pandas as pd


# Load model AND the column order it was trained on
model = joblib.load("best_model_xgboost.pkl")
model_columns = joblib.load("model_columns.pkl")   # <-- you need this file too


app = FastAPI()


class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=65, description="Age of the user")]
    sex: Annotated[Literal["male", "female"], Field(..., description="Sex of the user")]
    bmi: Annotated[float, Field(..., gt=0, description="BMI of the user")]
    children: Annotated[int, Field(..., ge=0, description="Number of children")]
    smoker: Annotated[Literal["yes", "no"], Field(..., description="Smoking status")]
    region: Annotated[
        Literal["northeast", "northwest", "southeast", "southwest"],
        Field(..., description="Region of the user")
    ]


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

    # Ensures exact column order the model expects
    return pd.DataFrame([row], columns=columns)


@app.post("/predict")
def predict_premium(data: UserInput):
    input_df = encode_input(data, model_columns)
    prediction = model.predict(input_df)[0]

    return JSONResponse(
        status_code=200,
        content={"predicted_charges": float(prediction)}
    )