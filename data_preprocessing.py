import pandas as pd
import numpy as np
import os
from datetime import datetime

def preprocess_data():
    """
    Loads preprocessed data if it exists, otherwise preprocesses the raw data and saves it.
    """
    if os.path.exists("processed_data"):
        print("Loading preprocessed data")
        return
    folder_path = os.path.join(os.getcwd(), "raw_data")
    assert os.path.exists(folder_path), f"Folder raw_data does not exist in the current directory."

    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")
    print("Preprocessing data")
    events = pd.read_csv("raw_data/events.csv")
    category_tree = pd.read_csv("raw_data/category_tree.csv")
    item_properties_1= pd.read_csv("raw_data/item_properties_part1.csv")
    item_properties_2= pd.read_csv("raw_data/item_properties_part2.csv")
    item_properties = pd.concat([item_properties_1, item_properties_2], ignore_index=True)
    del item_properties_1, item_properties_2
    
    events = events.drop(['transactionid'],axis=1)
    events=events.drop_duplicates()
    item_properties = item_properties.drop_duplicates()
    events['timestamp']= events['timestamp'].apply(lambda x:datetime.fromtimestamp(x/1000))
    item_properties['timestamp']= item_properties['timestamp'].apply(lambda x:datetime.fromtimestamp(x/1000))

    #Check if all the items has all 1104 properties in the training data
    property_item_count = item_properties[item_properties['itemid'].isin(events['itemid'])]
    property_item_count = pd.DataFrame(property_item_count.groupby(['property'])['itemid'].nunique().reset_index(name='unique_item_count'))
    property_item_count = property_item_count.sort_values(by='unique_item_count', ascending=False).reset_index(drop=True)

    #Only keep the properties that have value for atleast 150000 items
    top_properties = property_item_count[property_item_count['unique_item_count']>150000]['property']
    item_properties = item_properties[item_properties['property'].isin(top_properties)]

    #Sample 10000 items
    available_items = item_properties[(item_properties['property']=='available') & (item_properties['value']== '1')]['itemid'].unique()
    np.random.seed(1)
    sample_items = np.random.choice(list(available_items), size=10000, replace=False)

    events = events[events['itemid'].isin(sample_items)]
    item_properties = item_properties[item_properties['itemid'].isin(sample_items)]

    events['visitorid'] = events['visitorid'].astype(str)
    events['itemid'] = events['itemid'].astype(str)
    item_properties['itemid'] = item_properties['itemid'].astype(str)

    category_tree=category_tree.dropna()
    category_tree['categoryid'] = category_tree['categoryid'].astype(str)
    category_tree['parentid'] = category_tree['parentid'].astype(int).astype(str)
    
    category_tree_df = pd.DataFrame(item_properties[item_properties['property']=='categoryid']['value'])
    category_tree_df = category_tree_df.drop_duplicates()
    category_tree_df.rename(columns={'value':'categoryid'},inplace=True)
    # category_tree_df['categoryid'] = category_tree_df['categoryid'].astype(int)
    category_tree_df = merge_with_category_paths(category_tree_df, category_tree)

    #Splitting by time
    cutoff_time = pd.to_datetime('2015-08-01')
    events_train = events[events['timestamp']<=cutoff_time]
    events_valid = events[events['timestamp']>cutoff_time]

    users_train  = set(events_train['visitorid'].unique())
    users_valid = set(events_valid['visitorid'].unique())
    users_both = users_train.intersection(users_valid)
    events_valid = events_valid[events_valid['visitorid'].isin(users_both)]

    item_properties = item_properties[item_properties['itemid'].isin(events_train['itemid'])]

    item_properties_train = item_properties[item_properties['timestamp']<=cutoff_time]


    #We are preprocessing the data separately to avoid any data leakage
    popularity_item = process_popularity(events_train, level = "itemid")
    popularity_user = process_popularity(events_train, level = "visitorid")     
    popularity_item.to_csv("processed_data/popularity_item.csv", index=False)
    popularity_user.to_csv("processed_data/popularity_user.csv", index=False)

    #Make each property to a column
    item_properties_train = item_properties_train.pivot(index=['itemid','timestamp'], columns='property', values='value')
    item_properties = item_properties.pivot(index=['itemid','timestamp'], columns='property', values='value')

    #If some columns have all the same values, we cant use them
    for col in item_properties_train.columns:
        if item_properties_train[col].nunique()==1:
            item_properties.drop([col],axis=1,inplace=True)
            item_properties_train.drop([col],axis=1,inplace=True)

    # Missing Value imputation

    # Sort within each `itemid` by timestamp
    item_properties_train = item_properties_train.sort_index(level=['itemid', 'timestamp'])

    # Forward fill null values within each `itemid`, if some value is missing in forward filling, it does backward fill
    item_properties_train = item_properties_train.groupby('itemid').ffill().groupby('itemid').bfill()

    item_properties = item_properties.sort_index(level=['itemid', 'timestamp'])
    item_properties= item_properties.groupby('itemid').ffill().groupby('itemid').bfill()

    for col in item_properties_train.columns:
        if item_properties_train[col].isna().sum()!=0:
            item_properties_train[col].fillna(item_properties_train[col].mode()[0], inplace=True)
        if item_properties[col].isna().sum()!=0:
            item_properties[col].fillna(item_properties[col].mode()[0], inplace=True)

    item_properties_train['available'] = item_properties_train['available'].astype(int)
    item_properties['available'] = item_properties['available'].astype(int)

    for col in item_properties_train.columns:
        if col=='available' or col=='categoryid':
            continue
        if item_properties_train[col].apply(lambda x:x.startswith('n') and (len(x.split()))==1).all():
            item_properties_train[col] = item_properties_train[col].apply(lambda x:x[1:]) # Remove 'n' 
            item_properties[col] = item_properties[col].apply(lambda x:x[1:]) 
            if not item_properties_train[col].apply(lambda x:x.endswith('.000')).all():
                item_properties_train[col] = item_properties_train[col].astype(float)
                item_properties[col] = item_properties[col].astype(float)
            else:
                item_properties_train[col]=item_properties_train[col].apply(lambda x:x[:-4])
                item_properties[col] = item_properties[col].astype(int)

    item_properties_train = item_properties_train.reset_index(level=['itemid', 'timestamp'])
    item_properties = item_properties.reset_index(level=['itemid', 'timestamp'])

    item_properties = pd.merge(item_properties, category_tree_df, on='categoryid', how='left')
    item_properties_train = pd.merge(item_properties_train, category_tree_df, on='categoryid', how='left')

    for i in range(1, 6):
        if i == 1:
            item_properties.loc[item_properties[f'parent_level_{i}'].isna(), f'parent_level_{i}'] = item_properties['categoryid']
            item_properties_train.loc[item_properties_train[f'parent_level_{i}'].isna(), f'parent_level_{i}'] = item_properties_train['categoryid']
        else:
            item_properties.loc[item_properties[f'parent_level_{i}'].isna(), f'parent_level_{i}'] = item_properties[f'parent_level_{i-1}']
            item_properties_train.loc[item_properties_train[f'parent_level_{i}'].isna(), f'parent_level_{i}'] = item_properties_train[f'parent_level_{i-1}']
    

    events_train = events_train.sort_values('timestamp')
    events_valid = events_valid.sort_values('timestamp')
    
    events_train = pd.merge_asof(events_train, 
                                item_properties_train.reset_index(drop=True).sort_values('timestamp'), 
                                by='itemid', 
                                on='timestamp', 
                                direction='nearest')

    events_valid = pd.merge_asof(events_valid, 
                                item_properties.reset_index(drop=True).sort_values('timestamp'), 
                                by='itemid', 
                                on='timestamp', 
                                direction='nearest')
    
    del item_properties_train

    item_properties = item_properties.sort_values('timestamp')
    item_properties = item_properties.groupby(['itemid']).last().reset_index()

    
    events_train.to_csv("processed_data/interactions_train.csv", index=False)
    events_valid.to_csv("processed_data/interactions_valid.csv", index=False)
    item_properties.to_csv("processed_data/product_features.csv",index=False)
    return 



def process_popularity(df, level):

    if level=="itemid":
        col= "visitorid"
    elif level=="visitorid":
        col="itemid"
    # Count occurrences of each event type
    event_counts = df.groupby([level, "event"])[col].count().unstack(fill_value=0)
    event_counts.columns = ["number_of_addtocart", "number_of_purchases","number_of_views"]

   
    event_counts = event_counts.reset_index()

    event_counts["addtocart_to_view_ratio"] = (
    event_counts["number_of_addtocart"] / event_counts["number_of_views"]
    ).where(event_counts["number_of_views"] > 0, 0)

    event_counts["purchase_to_view_ratio"] = (
        event_counts["number_of_purchases"] / event_counts["number_of_views"]
    ).where(event_counts["number_of_views"] > 0, 0)

    event_counts["purchase_to_addtocart_ratio"] = (
        event_counts["number_of_purchases"] / event_counts["number_of_addtocart"]
    ).where(event_counts["number_of_addtocart"] > 0, 0)

    # Compute unique visitors and visits
    visitor_counts = df.groupby(level)[col].nunique().reset_index()
    visitor_counts.columns = [level, f"number_of_unique_{col[:-2]}s"]

    # visit_counts = df.groupby([level, "timestamp"]).size().groupby(level).size().reset_index()
    # visit_counts.columns = [level, "number_of_visits"]

    # result = event_counts.merge(visitor_counts, on=level).merge(visit_counts, on=level)
    result = event_counts.merge(visitor_counts, on=level)

    return result



def get_category_path(category_tree_df, category_id):
  
    path = []
    current_id = category_id
    
    # Prevent infinite loops in case of circular references
    max_depth = 100  
    depth = 0
    
    while current_id is not None and depth < max_depth:
        path.append(current_id)
        # Find the parent of current category
        parent_row = category_tree_df[category_tree_df['categoryid'] == current_id]
        
        if parent_row.empty:
            break
            
        current_id = parent_row['parentid'].iloc[0]
        # If we've reached root (parent_id is None or 0)
        if pd.isna(current_id) or current_id == 0:
            break
            
        depth += 1
    
    return path

def merge_with_category_paths(main_df, category_tree_df, category_column='categoryid'):

    result_df = main_df.copy()
    
    paths = {}
    unique_categories = main_df[category_column].unique()
    
    for cat_id in unique_categories:
        if pd.notna(cat_id):  # Skip if category_id is NaN
            paths[cat_id] = get_category_path(category_tree_df, cat_id)
    
    max_path_length = max(len(path) for path in paths.values())
    
    for level in range(1,max_path_length):
        result_df[f'parent_level_{level}'] = np.nan
    
    # Fill in path levels
    for idx, row in result_df.iterrows():
        cat_id = row[category_column]
        if pd.notna(cat_id) and cat_id in paths:
            path = paths[cat_id]
            for level, path_cat_id in enumerate(path):
                if level==0:
                    continue
                result_df.at[idx, f'parent_level_{level}'] = int(path_cat_id)

    
    return result_df



