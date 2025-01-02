import numpy as np
import pandas as pd


def load_processed_data():
    """Load the processed data"""
    interactions_train = pd.read_csv("processed_data/interactions_train.csv")
    interactions_valid = pd.read_csv("processed_data/interactions_valid.csv")
    product_features = pd.read_csv("processed_data/product_features.csv")
    interactions_valid = interactions_valid.drop_duplicates()

    num_cols = interactions_train.columns[9:20]
    new_col_names = {col: f'property_{i+1}' for i, col in enumerate(num_cols)}
    interactions_train.rename(columns=new_col_names, inplace=True)
    interactions_valid.rename(columns=new_col_names, inplace=True)
    product_features.rename(columns=new_col_names, inplace=True)
    interactions_train.dropna(inplace=True)
    interactions_valid.dropna(inplace=True)
    interactions_valid = interactions_valid[interactions_valid['visitorid'].isin(interactions_train['visitorid'])]
    interactions_train['visitorid'] = interactions_train['visitorid'].astype(str)
    interactions_valid['visitorid'] = interactions_valid['visitorid'].astype(str)
    for col in ['itemid', 'categoryid', 'parent_level_1', 'parent_level_2', 'parent_level_3', 'parent_level_4', 'parent_level_5']:
        interactions_train[col] = interactions_train[col].astype(int).astype(str)
        interactions_valid[col] = interactions_valid[col].astype(int).astype(str)
        product_features[col] = product_features[col].astype(int).astype(str)
    return interactions_train, interactions_valid, product_features

    
