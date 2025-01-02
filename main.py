# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import warnings
import datetime
warnings.filterwarnings("ignore")
import tensorflow as tf
import faiss
from tensorflow.keras import models, layers
import tensorflow_recommenders as tfrs
from data_preprocessing import preprocess_data
from data_preparation import prepare_interaction_data, prepare_product_data, load_processed_data
from utilities import get_product_features, visualisation
from user_tower import UserModel
from item_tower import ItemModel
from two_tower_model import TwoTowerModel as model
from metrics import recall_at_k, precision_at_k

# %%
preprocess_data()

# %%
interactions, interactions_valid, product_features = load_processed_data()


# %%
interactions.shape, product_features.shape, interactions_valid.shape

# %%
interactions_tf = tf.data.Dataset.from_tensor_slices((interactions.to_dict("list")))
interactions_valid_tf = tf.data.Dataset.from_tensor_slices((interactions_valid.to_dict("list")))
products_tf = tf.data.Dataset.from_tensor_slices((product_features.to_dict("list")))

# %%
customer_model = UserModel(interactions_tf, embedding_dim=64)
product_model = ItemModel(products_tf, embedding_dim=64)

# %%
products = get_product_features(products_tf)

# %%
model = model(customer_model, product_model, products)

# %%
train = interactions_tf.batch(8192).cache()
validation = interactions_valid_tf.batch(2048).cache()

# %%
interactions_valid = interactions_valid[['visitorid', 'itemid', 'timestamp_raw', 'user_number_of_views', 'user_number_of_addtocart', 'user_number_of_purchases', 'number_of_unique_items']]
interactions_valid_tf = tf.data.Dataset.from_tensor_slices((interactions_valid.to_dict("list")))

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01))
model.fit(train, epochs=30, validation_data=validation)


# %%
item_ids = products_tf.map(lambda x: x['itemid'])
k = 50
# Create the BruteForce index
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model, k=k)

# Index the dataset with itemid as the identifier
index.index_from_dataset(
    tf.data.Dataset.zip((
        item_ids.batch(100),  # Batch the item IDs
        products_tf.batch(100).map(model.candidate_model)  # Map to embeddings
    ))
)

# %%
# Iterate through the batches
recalls = []
interactions_valid_tf = interactions_valid_tf.batch(128).cache()
for batch in interactions_valid_tf:
    _, recommendations = index(batch)
    visitors = batch['visitorid'].numpy()
    items = batch['itemid'].numpy()
    for visitor,item, recommended in zip(visitors,items, recommendations):
        visitor_id = visitor.decode('utf-8')
        item_id = item.decode('utf-8')
        recommended_items = [item.decode('utf-8') for item in recommended.numpy()]  # Decode recommended item
        recall = recall_at_k(set([item_id]),recommended_items, k=50)
        recalls.append(recall)
print(f"Recall@{k}: ",np.mean(recalls))


# %% [markdown]
# ## Retrival with KNN using FAISS

item_ids = tf.concat(list(products_tf.map(lambda x: x['itemid'])), axis=0).numpy()
item_embeddings = tf.concat(
    list(products_tf.batch(100).map(model.candidate_model)), axis=0
).numpy()

# Ensure embeddings are in float32 for FAISS
item_embeddings = item_embeddings.astype('float32')

# Define the FAISS index for approximate nearest neighbors
embedding_dim = item_embeddings.shape[1]
num_clusters = 100  # Number of clusters for IVF (can be tuned)
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(embedding_dim), embedding_dim, num_clusters)

# Train the FAISS index with the item embeddings
index.train(item_embeddings)

# Add item embeddings to the FAISS index
index.add(item_embeddings)

# Save a mapping of FAISS indices to item IDs for retrieval
id_map = {i: item_ids[i].decode('utf-8') for i in range(len(item_ids))}

# %%
query_embeddings = tf.concat(
    list(interactions_valid_tf.batch(128).map(model.query_model)), axis=0
).numpy().astype('float32')

# Approximate KNN search with FAISS
k = 50  # Number of nearest neighbors to retrieve
distances, recommendations = index.search(query_embeddings, k)

# Extract relevant data from the batches
recalls = []
for i, batch in enumerate(interactions_valid_tf):
    visitors = batch['visitorid'].numpy()
    items = batch['itemid'].numpy()
    
    for visitor, item, recommended_indices in zip(visitors, items, recommendations[i]):
        visitor_id = visitor.decode('utf-8')
        item_id = item.decode('utf-8')
        recommended_items = [id_map[idx] for idx in recommended_indices]  # Decode item IDs
        
        # Calculate recall@k
        recall = recall_at_k(set([item_id]), recommended_items, k)
        recalls.append(recall)

# Calculate the average recall
average_recall = np.mean(recalls)
print(f"Recall@{k}: {average_recall:.4f}")




