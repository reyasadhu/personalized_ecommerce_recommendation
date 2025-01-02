import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np

class UserModel(tf.keras.Model):
  """ Features Model representing the user and its features """
  def __init__(self, dataset, embedding_dim=64):
    super().__init__()

    self.dataset = dataset

    #Preprocessing

    self.user_vocab = layers.StringLookup(mask_token=None)
    self.user_vocab.adapt(self.dataset.map(lambda x: x["visitorid"]))

    # User Continuous Timestamp
    self.user_time_norm = layers.Normalization(axis=None)
    self.timestamps = np.concatenate(list(self.dataset.map(lambda x: x["timestamp_raw"]).batch(1000)))
    self.user_time_norm.adapt(self.timestamps)

    #User popularity data
    self.user_views_norm = layers.Normalization(axis=None)
    self.number_of_views = np.concatenate(list(self.dataset.map(lambda x: x["user_number_of_views"]).batch(1000)))
    self.user_views_norm.adapt(self.number_of_views)

    self.user_atc_norm = layers.Normalization(axis=None)
    self.number_of_addtocart = np.concatenate(list(self.dataset.map(lambda x: x["user_number_of_addtocart"]).batch(1000)))
    self.user_atc_norm.adapt(self.number_of_addtocart)

    self.user_purchases_norm = layers.Normalization(axis=None)
    self.number_of_purchases = np.concatenate(list(self.dataset.map(lambda x: x["user_number_of_purchases"]).batch(1000)))
    self.user_purchases_norm.adapt(self.number_of_purchases)

    self.user_n_items_norm = layers.Normalization(axis=None)
    self.number_of_items = np.concatenate(list(self.dataset.map(lambda x: x["number_of_unique_items"]).batch(1000)))
    self.user_n_items_norm.adapt(self.number_of_items)


    # Dimensions for embedding into high dimensional vectors
    self.embedding_dim = embedding_dim

    #Embedding + norm layers
    self.user_embedding = models.Sequential()
    self.user_embedding.add(self.user_vocab)
    self.user_embedding.add(layers.Embedding(self.user_vocab.vocabulary_size(), self.embedding_dim, embeddings_initializer='normal'))

    # Time Continuous
    self.time_continuous = models.Sequential()
    self.time_continuous.add(self.user_time_norm)

    #Popularity features
    self.user_views = models.Sequential()
    self.user_views.add(self.user_views_norm)

    self.user_atc = models.Sequential()
    self.user_atc.add(self.user_atc_norm)

    self.user_purchases = models.Sequential()
    self.user_purchases.add(self.user_purchases_norm)

    self.user_n_items = models.Sequential()
    self.user_n_items.add(self.user_n_items_norm)



  # Forward pass
  def call(self, inputs):
    """ 
    Forward Pass with user features 
    """
    return tf.concat([
                      self.user_embedding(inputs["visitorid"]),
                      tf.reshape(self.time_continuous(inputs["timestamp_raw"]), (-1, 1)),
                      tf.reshape(self.user_views(inputs["user_number_of_views"]), (-1, 1)),
                      tf.reshape(self.user_atc(inputs["user_number_of_addtocart"]),( -1, 1)),
                      tf.reshape(self.user_purchases(inputs["user_number_of_purchases"]),( -1, 1)),
                      tf.reshape(self.user_n_items(inputs["number_of_unique_items"]), (-1, 1))
    ], axis=1)
