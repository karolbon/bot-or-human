from read_data import load_dataframe
import pandas as pd


def feature_engineering(df):
    # Drop Label column as we don't want to be biased by bringing the label into the feature engineering
    features = df.drop(labels='Label', axis='columns')

    print(features.head())


df = load_dataframe('training_df')
feature_engineering(df)
