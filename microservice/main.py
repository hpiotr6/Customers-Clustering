from fastapi import FastAPI, File, Form, UploadFile
from numpy.core.fromnumeric import prod
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from io import StringIO
from typing import List, Union

from RFM_model.features.build_features import FeatureBuilder
from RFM_model.models.train_model import Trainer
from RFM_model.models.predict_model import Predictor
import os
print(os.getcwd())

FeatureBuilder.from_files('../data/raw').build_save("../data/processed")
Trainer(3, 42, "../data/processed/processed.csv").train_save("../models")
pr = Predictor("../models")

app = FastAPI()


class DataTables(BaseModel):
    sessions: str
    products: str
    users: str


@app.post('/predict')
async def predict_species(tables: DataTables):
    data = tables.dict()
    x = FeatureBuilder.from_json(
        sessions=data["sessions"], users=data["users"], products=data["products"]).build()
    print(x)

    f = pr.predict(x)

    f.to_json(orient="records")

    return f.to_json(orient="records")
