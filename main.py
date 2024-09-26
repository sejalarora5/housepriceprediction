from fastapi import FastAPI, HTTPException
import uvicorn
from production_model.regression_model.predict import make_prediction
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from schemas import prediction
from fastapi.encoders import jsonable_encoder
import json
from typing import Any

app = FastAPI(
    title='House Price Prediction App using FastAPI',
    description="A simple Demo",
    version='1.0'
)


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def index():
    return {"message": "Welcome to House Price Prediction App using FastApi"}


@app.post("/predict", response_model=prediction.PredictionResults, status_code=200)
async def predict(input_data: prediction.MultipleHouseDataInputs) -> Any:

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        # logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    # logger.info(f"Prediction results: {results.get('predictions')}")

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)