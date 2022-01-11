from datetime import timedelta
import pandas as pd
import os


class FeatureBuilder:
    def __init__(self, raw_path) -> None:
        SESSIONS_PATH = os.path.join(raw_path, "sessions.jsonl")
        USERS_PATH = os.path.join(raw_path, "users.jsonl")
        PRODUCTS_PATH = os.path.join(raw_path, "products.jsonl")

        self.raw_path = raw_path

        self.sess_df = self.load_data(SESSIONS_PATH)
        self.usr_df = self.load_data(USERS_PATH)
        self.prod_df = self.load_data(PRODUCTS_PATH)

    def load_data(self, path: str) -> pd.DataFrame:
        return pd.read_json(path_or_buf=path, lines=True)

    def merge_dataframes(self, sess_df, usr_df, prod_df):
        sess_df.dropna(subset=["user_id"], inplace=True)
        sess_df.user_id = sess_df.user_id.astype(int)
        sess_df.product_id = sess_df.product_id.astype(pd.Int64Dtype())
        sess_df.dropna(subset=["product_id"], inplace=True)
        merged = sess_df.merge(prod_df).merge(usr_df)
        return merged

    def get_rfm(self, df):
        snapshot_date = max(df.timestamp) + timedelta(days=1)
        df_rfm = df[df.event_type == "BUY_PRODUCT"].groupby(['user_id'], as_index=False).agg({'timestamp': lambda x: (snapshot_date - x.max()).days,
                                                                                              'session_id': 'count',
                                                                                              'price': 'sum'}).rename(columns={'timestamp': 'Recency',
                                                                                                                               'session_id': 'Frequency',
                                                                                                                               'price': 'MonetaryValue'})
        return df_rfm

    def build(self):
        merged = self.merge_dataframes(self.sess_df, self.usr_df, self.prod_df)
        rfm = self.get_rfm(merged)
        return rfm

    def build_save(self, output_path):
        rfm = self.build()
        path = os.path.join(output_path, "processed.csv")
        rfm.to_csv(path, index=False)
