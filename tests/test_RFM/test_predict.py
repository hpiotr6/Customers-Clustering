from RFM_model.models.predict_model import Predictor
import sklearn
import pytest
import pandas as pd


@pytest.fixture
def predictor():
    return Predictor("models")


@pytest.fixture
def df():
    return pd.read_csv("data/processed/processed.csv")


def test_init(predictor):
    assert type(predictor.model) == sklearn.cluster._kmeans.KMeans


def test_predict(df, predictor):
    users = predictor.predict(df)
    best_df = df[df.user_id.isin(users)]
    worse_df = df[~df.user_id.isin(users)]
    assert best_df.MonetaryValue.mean() > worse_df.MonetaryValue.mean()
