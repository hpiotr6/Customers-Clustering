from numpy import mod
from pandas.core.reshape.merge import merge
from IUM21Z_Zad_05_03.features.build_features import FeaturesSimpleModel
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class TestFeatures:
    def test_from_files(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        assert type(model) == FeaturesSimpleModel

    def test_from_json(self):
        model = FeaturesSimpleModel.from_json(
            "data/raw/sessions.jsonl", "data/raw/users.jsonl", "data/raw/products.jsonl")
        assert type(model) == FeaturesSimpleModel

    def test_merge_dataframes_and_add_attributes_type(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        assert type(frame) == pd.DataFrame

    def test_merge_dataframes_and_add_attributes_size(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        assert frame.size > 400

    def test_merge_dataframes_and_add_attributes_check_if_attributes_exist(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        for column_name in ['is_male', 'amount', 'recency', 'frequency', 'viewed_num', 'bought_num', 'bought/sum']:
            if column_name in frame.columns:
                assert True
            else:
                assert False

    def test_delete_attributes_except_amount(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        minimal_frame = model.delete_attributes_except_amount(frame)
        for column_name in ['is_male', 'recency', 'frequency', 'viewed_num', 'bought_num', 'bought/sum']:
            assert column_name not in frame.columns
        assert 'amount' not in frame.columns

    def test_delete_customers_with_below_10_items_bought(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        frame = model.delete_customers_with_below_10_items_bought(frame)
        assert frame['viewed_num'].min() >= 10

    def test_generate_processed_files_minimal(self):
        model = FeaturesSimpleModel.from_files("data/raw")

        model.generate_processed_files_minimal(
            "E:\code\gitlab elka repo\ium-21z\data\processed")
        frame = model.merge_dataframes_and_add_attributes()
        merged = model.delete_attributes_except_amount(frame)
        final_frame = model.delete_customers_with_below_10_items_bought(
            merged)
        test_path = os.path.join(
            "E:\code\gitlab elka repo\ium-21z\data\processed", "processed.csv")
        assert pd.read_json(path_or_buf=test_path, lines=True).size > 0

    def test_generate_processed_files_maximal(self, out_path):
        model = FeaturesSimpleModel.from_files("data/raw")
        model.generate_processed_files_maximal("path")
