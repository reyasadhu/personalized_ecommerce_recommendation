import numpy as np
import pandas as pd


def load_processed_data():
    """Load the processed data"""
    interactions_train = pd.read_csv("processed_data/interactions_train.csv")
    interactions_valid = pd.read_csv("processed_data/interactions_valid.csv")
    product_features = pd.read_csv("processed_data/product_features.csv")
    # popularity_item = pd.read_csv("processed_data/popularity_item.csv")
    interactions_valid = interactions_valid.drop_duplicates()
    
     # Remove duplicate user-item events, keeping the last one
    # event_priority = {'view': 0, 'addtocart': 1, 'transaction': 2}
    # interactions_train['event_priority'] = interactions_train['event'].map(event_priority)
    # interactions_valid['event_priority'] = interactions_valid['event'].map(event_priority)
    # interactions_train = interactions_train.sort_values(by=['visitorid', 'itemid', 'event_priority'])
    # interactions_valid = interactions_valid.sort_values(by=['visitorid', 'itemid', 'event_priority'])
    # interactions_train = interactions_train.drop_duplicates(subset=['visitorid', 'itemid'], keep='last')
    # interactions_valid = interactions_valid.drop_duplicates(subset=['visitorid', 'itemid'], keep='last')
    # interactions_train = interactions_train.drop(columns=['event_priority'])
    # interactions_valid = interactions_valid.drop(columns=['event_priority'])

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
    # popularity_item['itemid'] = popularity_item['itemid'].astype(str)
    return interactions_train, interactions_valid, product_features

def prepare_interaction_data(interaction, popularity_user, popularity_item):
    """Preprocess the interactions dataset"""
    
    interaction = interaction.merge(popularity_user[['visitorid', 'number_of_addtocart', 'number_of_purchases',
       'number_of_views','number_of_unique_items']], on="visitorid", how="left")
    
    for col in ['number_of_addtocart', 'number_of_purchases', 'number_of_views', 'number_of_unique_items']:
        interaction[col].fillna(popularity_user[col].mean(), inplace=True)
    
    interaction = interaction.rename(columns={"number_of_views": "user_number_of_views", "number_of_addtocart": "user_number_of_addtocart", "number_of_purchases": "user_number_of_purchases"})
    
    interaction = interaction.merge(popularity_item[['itemid', 'number_of_views', 'number_of_addtocart', 'number_of_purchases','number_of_unique_visitors']], on="itemid", how="left")

    for col in ['number_of_addtocart', 'number_of_purchases', 'number_of_views', 'number_of_unique_visitors']:
        interaction[col].fillna(popularity_item[col].mean(), inplace=True)

    interaction = interaction.rename(columns={"number_of_views": "item_number_of_views", "number_of_addtocart": "item_number_of_addtocart", "number_of_purchases": "item_number_of_purchases"})

    return interaction



def prepare_product_data(products, popularity_item):
    """Preprocess the products dataset"""
    
    products = products.merge(popularity_item[['itemid', 'number_of_views', 'number_of_addtocart', 'number_of_purchases','number_of_unique_visitors']], on="itemid", how="left")
    
    for col in ['number_of_addtocart', 'number_of_purchases', 'number_of_views', 'number_of_unique_visitors']:
        products[col].fillna(popularity_item[col].mean(), inplace=True)

    products = products.rename(columns={"number_of_views": "item_number_of_views", "number_of_addtocart": "item_number_of_addtocart", "number_of_purchases": "item_number_of_purchases"})

    return products
    