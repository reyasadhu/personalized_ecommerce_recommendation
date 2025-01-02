import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import tensorflow_recommenders as tfrs



class TwoTowerModel(tfrs.models.Model):
  """ Retrieval model encompassing users as query and items as candidates"""

  def __init__(self, user_model, item_model, product_features):
    super().__init__()
    self.user_model = user_model
    self.item_model = item_model
    self.product_features = product_features

    # Query tower
    self.query_model = models.Sequential()
    self.query_model.add(self.user_model)
    self.query_model.add(layers.Dense(128, activation="relu", kernel_initializer='normal'))
    self.query_model.add(layers.Dense(64, activation="relu", kernel_initializer='normal'))
    self.query_model.add(layers.Dense(64, kernel_initializer='normal'))
    self.query_model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
               
    # Candidate tower
    self.candidate_model = models.Sequential()
    self.candidate_model.add(self.item_model)
    self.candidate_model.add(layers.Dense(128, activation="relu", kernel_initializer='normal'))
    self.candidate_model.add(layers.Dense(64, activation="relu", kernel_initializer='normal'))
    self.candidate_model.add(layers.Dense(64, kernel_initializer='normal'))
    self.candidate_model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    
    # Retrieval task for loss function
    metrics = tfrs.metrics.FactorizedTopK(
        candidates=product_features.batch(256).map(self.candidate_model))
    # )
    self.task = tfrs.tasks.Retrieval(metrics=metrics, temperature=0.1)
  
  def compute_loss(self, features, training=False):
    # Passing the embeddings into the loss function
    user_embeddings = self.query_model({
                                    "visitorid": features["visitorid"],
                                    # "event": features["event"],
                                    "timestamp_raw": features["timestamp_raw"],
                                    "user_number_of_views": features["user_number_of_views"],
                                    "user_number_of_addtocart": features["user_number_of_addtocart"],
                                    "user_number_of_purchases": features["user_number_of_purchases"],
                                    "number_of_unique_items": features["number_of_unique_items"]
                                    })
    
    
    
    product_embeddings = self.candidate_model(
        {"itemid": features["itemid"], 
         "property_1": features["property_1"], 
         "property_2": features["property_2"], 
         "property_3": features["property_3"], 
         "property_4": features["property_4"], 
         "property_5": features["property_5"], 
         "property_6": features["property_6"], 
         "property_7": features["property_7"], 
         "property_8": features["property_8"], 
         "property_9": features["property_9"], 
         "property_10": features["property_10"], 
         "property_11": features["property_11"], 
        #  "available": features["available"],
         "categoryid": features["categoryid"],
         "parent_level_1": features["parent_level_1"],
         "parent_level_2": features["parent_level_2"],
         "parent_level_3": features["parent_level_3"],
         "parent_level_4": features["parent_level_4"],
         "parent_level_5": features["parent_level_5"]
        #  "item_number_of_views": features["item_number_of_views"],
        #  "item_number_of_addtocart": features["item_number_of_addtocart"],
        #  "item_number_of_purchases": features["item_number_of_purchases"],
        #  "number_of_unique_visitors": features["number_of_unique_visitors"]
         })


    # Calculate the loss via task for query and candidate embeddings
    return self.task(user_embeddings, product_embeddings, compute_metrics=not training)