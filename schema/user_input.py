from pydantic import BaseModel, Field
from typing import Literal, Annotated

class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=18, lt=65, description="Age of the user")]
    sex: Annotated[Literal["male", "female"], Field(..., description="Sex of the user")]
    bmi: Annotated[float, Field(..., gt=0, description="BMI of the user")]
    children: Annotated[int, Field(..., ge=0, description="Number of children")]
    smoker: Annotated[Literal["yes", "no"], Field(..., description="Smoking status")]
    region: Annotated[
        Literal["northeast", "northwest", "southeast", "southwest"],
        Field(..., description="Region of the user")
    ]