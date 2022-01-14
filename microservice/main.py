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

from IUM21Z_Zad_05_03.features.build_features import FeaturesSimpleModel
from IUM21Z_Zad_05_03.models.predict_model import SimpleModelPredictor
from IUM21Z_Zad_05_03.models.train_model import SimpleModelTrainer


print(os.getcwd())

FeatureBuilder.from_files('../data/raw').build_save("../data/processed")
Trainer(3, 42, "../data/processed/processed.csv").train_save("../models")
predictor = Predictor("../models")

FeaturesSimpleModel.from_files(
    "../data/raw").generate_processed_files("../data/processed")
SimpleModelTrainer(
    "../data/processed/train/processed.csv").train_save_simple_model("../models")
simple_predictor = SimpleModelPredictor("../models")

app = FastAPI()


class DataTables(BaseModel):
    sessions: str
    products: str
    users: str


@app.post('/predict')
async def predict(tables: DataTables):
    data = tables.dict()
    x = FeatureBuilder.from_json(
        sessions=data["sessions"], users=data["users"], products=data["products"]).build()

    f = predictor.predict(x)

    return f.to_json(orient="records")


@app.post('/predict_simple')
async def predict_simple(tables: DataTables):
    data = tables.dict()
    sess = pd.read_json(data["sessions"])

    x = FeaturesSimpleModel.from_json(
        sessions=data["sessions"], users=data["users"], products=data["products"]).merge_dataframes_and_add_attributes(sess)
    result_frame = simple_predictor.predict_simple_model(x)
    best_group = simple_predictor.best_group(result_frame)
    f = result_frame[result_frame.Cluster == best_group].user_id
    return f.to_json(orient="records")
