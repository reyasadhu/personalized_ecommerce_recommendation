import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np



# Defining the Product Features

class ItemModel(tf.keras.Model):
  """ 
  Item Model representing products and its features 
  """
  def __init__(self, dataset, embedding_dim=64):
    super().__init__()

    self.dataset = dataset
    #Preprocessing

    # Products
    self.products_vocab = layers.StringLookup(mask_token=None)
    self.products_vocab.adapt(self.dataset.map(lambda x: x["itemid"]))

    # # Availibility
    # self.product_availability = layers.CategoryEncoding(num_tokens=2, output_mode="one_hot")

    # category
    self.category_vocab = layers.StringLookup(mask_token=None)
    self.category_vocab.adapt(self.dataset.map(lambda x: tf.strings.as_string(x['categoryid'])))

    self.parent1_vocab = layers.StringLookup(mask_token=None)
    self.parent1_vocab.adapt(self.dataset.map(lambda x: tf.strings.as_string(x['parent_level_1'])))

    self.parent2_vocab = layers.StringLookup(mask_token=None)
    self.parent2_vocab.adapt(self.dataset.map(lambda x: tf.strings.as_string(x['parent_level_2'])))

    self.parent3_vocab = layers.StringLookup(mask_token=None)
    self.parent3_vocab.adapt(self.dataset.map(lambda x: tf.strings.as_string(x['parent_level_3'])))

    self.parent4_vocab = layers.StringLookup(mask_token=None)
    self.parent4_vocab.adapt(self.dataset.map(lambda x: tf.strings.as_string(x['parent_level_4'])))

    self.parent5_vocab = layers.StringLookup(mask_token=None)
    self.parent5_vocab.adapt(self.dataset.map(lambda x: tf.strings.as_string(x['parent_level_5'])))

    # # Sales
    # self.product_sales_norm = layers.Normalization(axis=None)
    # self.sales = np.concatenate(list(self.dataset.map(lambda x: x["item_number_of_purchases"]).batch(1000)))
    # self.product_sales_norm.adapt(self.sales)

    # # Views
    # self.product_views_norm = layers.Normalization(axis=None)
    # self.views = np.concatenate(list(self.dataset.map(lambda x: x["item_number_of_views"]).batch(1000)))
    # self.product_views_norm.adapt(self.views)

    # # Addto-carts
    # self.product_carts_norm = layers.Normalization(axis=None)
    # self.carts = np.concatenate(list(self.dataset.map(lambda x: x["item_number_of_addtocart"]).batch(1000)))
    # self.product_carts_norm.adapt(self.carts)

    # # users
    # self.product_n_users_norm = layers.Normalization(axis=None)
    # self.n_users = np.concatenate(list(self.dataset.map(lambda x: x["number_of_unique_visitors"]).batch(1000)))
    # self.product_n_users_norm.adapt(self.n_users)

    # Product properties
    self.max_features = 10000

    self.product_property1_vocab = layers.TextVectorization(max_tokens=self.max_features, output_mode='int',output_sequence_length=None)
    self.type1 = np.concatenate(list(self.dataset.map(lambda x: x["property_1"]).batch(1000)))
    self.product_property1_vocab.adapt(self.type1)
    
    self.product_property2_vocab = layers.TextVectorization(max_tokens=self.max_features, output_mode='int',output_sequence_length=None)
    self.type2 = np.concatenate(list(self.dataset.map(lambda x: x["property_2"]).batch(1000)))
    self.product_property2_vocab.adapt(self.type2)

    self.product_property3_vocab = layers.TextVectorization(max_tokens=self.max_features, output_mode='int',output_sequence_length=None)
    self.type3 = np.concatenate(list(self.dataset.map(lambda x: x["property_3"]).batch(1000)))
    self.product_property3_vocab.adapt(self.type3)

    self.product_property4_norm = layers.Normalization(axis=None)
    self.type4 = np.concatenate(list(self.dataset.map(lambda x: x["property_4"]).batch(1000)))
    self.product_property4_norm.adapt(self.type4)

    self.product_property5_vocab = layers.TextVectorization(max_tokens=self.max_features, output_mode='int',output_sequence_length=None)
    self.type5 = np.concatenate(list(self.dataset.map(lambda x: x["property_5"]).batch(1000)))
    self.product_property5_vocab.adapt(self.type5)

    self.product_property6_vocab = layers.TextVectorization(max_tokens=self.max_features, output_mode='int',output_sequence_length=None)
    self.type6 = np.concatenate(list(self.dataset.map(lambda x: x["property_6"]).batch(1000)))
    self.product_property6_vocab.adapt(self.type6)

    self.product_property7_norm = layers.Normalization(axis=None)
    self.type7 = np.concatenate(list(self.dataset.map(lambda x: x["property_7"]).batch(1000)))
    self.product_property7_norm.adapt(self.type7)

    self.product_property8_norm = layers.Normalization(axis=None)
    self.type8 = np.concatenate(list(self.dataset.map(lambda x: x["property_8"]).batch(1000)))
    self.product_property8_norm.adapt(self.type8)

    self.product_property9_vocab = layers.TextVectorization(max_tokens=self.max_features, output_mode='int',output_sequence_length=None)
    self.type9 = np.concatenate(list(self.dataset.map(lambda x: x["property_9"]).batch(1000)))
    self.product_property9_vocab.adapt(self.type9)

    self.product_property10_vocab = layers.TextVectorization(max_tokens=self.max_features, output_mode='int',output_sequence_length=None)
    self.type10 = np.concatenate(list(self.dataset.map(lambda x: x["property_10"]).batch(1000)))
    self.product_property10_vocab.adapt(self.type10)

    self.product_property11_vocab = layers.TextVectorization(max_tokens=self.max_features, output_mode='int',output_sequence_length=None)
    self.type11 = np.concatenate(list(self.dataset.map(lambda x: x["property_11"]).batch(1000)))
    self.product_property11_vocab.adapt(self.type11)
    

    # Dimensions for embedding into high dimensional vectors
    self.embedding_dim = embedding_dim

    # Embedding + Norm Layers

    self.product_embedding = models.Sequential()
    self.product_embedding.add(self.products_vocab)
    self.product_embedding.add(layers.Embedding(self.products_vocab.vocabulary_size(), self.embedding_dim, embeddings_initializer='normal'))

    # # Availibility
    # self.availability = models.Sequential()
    # self.availability.add(self.product_availability)

    # Product Category
    self.category = models.Sequential()
    self.category.add(self.category_vocab)
    self.category.add(layers.Embedding(self.category_vocab.vocabulary_size(), self.embedding_dim, embeddings_initializer='normal'))

    self.type1_embedding = models.Sequential()
    self.type1_embedding.add(self.parent1_vocab)
    self.type1_embedding.add(layers.Embedding(self.parent1_vocab.vocabulary_size(), self.embedding_dim, embeddings_initializer='normal'))

    self.type2_embedding = models.Sequential()
    self.type2_embedding.add(self.parent2_vocab)
    self.type2_embedding.add(layers.Embedding(self.parent2_vocab.vocabulary_size(), self.embedding_dim, embeddings_initializer='normal'))

    self.type3_embedding = models.Sequential()
    self.type3_embedding.add(self.parent3_vocab)
    self.type3_embedding.add(layers.Embedding(self.parent3_vocab.vocabulary_size(), self.embedding_dim, embeddings_initializer='normal'))

    self.type4_embedding = models.Sequential()
    self.type4_embedding.add(self.parent4_vocab)
    self.type4_embedding.add(layers.Embedding(self.parent4_vocab.vocabulary_size(), self.embedding_dim, embeddings_initializer='normal'))

    self.type5_embedding = models.Sequential()
    self.type5_embedding.add(self.parent5_vocab)
    self.type5_embedding.add(layers.Embedding(self.parent5_vocab.vocabulary_size(), self.embedding_dim, embeddings_initializer='normal'))

    # # Sales
    # self.sales = models.Sequential()
    # self.sales.add(self.product_sales_norm)

    # # Views
    # self.views = models.Sequential()
    # self.views.add(self.product_views_norm)

    # # Carts
    # self.carts = models.Sequential()
    # self.carts.add(self.product_carts_norm)

    # # Users
    # self.users = models.Sequential()
    # self.users.add(self.product_n_users_norm)

    
    # Product properties
    self.property1_embedding = models.Sequential()
    self.property1_embedding.add(layers.Input(shape=(None,)))
    self.property1_embedding.add(layers.Embedding(self.product_property1_vocab.vocabulary_size(), self.embedding_dim, mask_zero=True, embeddings_initializer='normal'))
    self.property1_embedding.add(layers.GlobalAveragePooling1D())

    self.property2_embedding = models.Sequential()
    self.property2_embedding.add(layers.Input(shape=(None,)))
    self.property2_embedding.add(layers.Embedding(self.product_property2_vocab.vocabulary_size(), self.embedding_dim, mask_zero=True, embeddings_initializer='normal'))
    self.property2_embedding.add(layers.GlobalAveragePooling1D())

    self.property3_embedding = models.Sequential()
    self.property3_embedding.add(layers.Input(shape=(None,)))
    self.property3_embedding.add(layers.Embedding(self.product_property3_vocab.vocabulary_size(), self.embedding_dim, mask_zero=True, embeddings_initializer='normal'))
    self.property3_embedding.add(layers.GlobalAveragePooling1D())

    self.property4 = models.Sequential()
    self.property4.add(self.product_property4_norm)

    self.property5_embedding = models.Sequential()
    self.property5_embedding.add(layers.Input(shape=(None,)))
    self.property5_embedding.add(layers.Embedding(self.product_property5_vocab.vocabulary_size(), self.embedding_dim, mask_zero=True, embeddings_initializer='normal'))
    self.property5_embedding.add(layers.GlobalAveragePooling1D())

    self.property6_embedding = models.Sequential()
    self.property6_embedding.add(layers.Input(shape=(None,)))
    self.property6_embedding.add(layers.Embedding(self.product_property6_vocab.vocabulary_size(), self.embedding_dim, mask_zero=True, embeddings_initializer='normal'))
    self.property6_embedding.add(layers.GlobalAveragePooling1D())

    self.property7 = models.Sequential()
    self.property7.add(self.product_property7_norm)

    self.property8 = models.Sequential()
    self.property8.add(self.product_property8_norm)

    self.property9_embedding = models.Sequential()
    self.property9_embedding.add(layers.Input(shape=(None,)))
    self.property9_embedding.add(layers.Embedding(self.product_property9_vocab.vocabulary_size(), self.embedding_dim, mask_zero=True, embeddings_initializer='normal'))
    self.property9_embedding.add(layers.GlobalAveragePooling1D())

    self.property10_embedding = models.Sequential()
    self.property10_embedding.add(layers.Input(shape=(None,)))
    self.property10_embedding.add(layers.Embedding(self.product_property10_vocab.vocabulary_size(), self.embedding_dim, mask_zero=True, embeddings_initializer='normal'))
    self.property10_embedding.add(layers.GlobalAveragePooling1D())

    self.property11_embedding = models.Sequential()
    self.property11_embedding.add(layers.Input(shape=(None,)))
    self.property11_embedding.add(layers.Embedding(self.product_property11_vocab.vocabulary_size(), self.embedding_dim, mask_zero=True, embeddings_initializer='normal'))
    self.property11_embedding.add(layers.GlobalAveragePooling1D())


  def call(self, inputs):
      return tf.concat([
          self.product_embedding(inputs["itemid"]),
          # self.availability(inputs["available"]),
          self.category(inputs["categoryid"]),
          self.type1_embedding(inputs["parent_level_1"]),
          self.type2_embedding(inputs["parent_level_2"]),
          self.type3_embedding(inputs["parent_level_3"]),
          self.type4_embedding(inputs["parent_level_4"]),
          self.type5_embedding(inputs["parent_level_5"]),
          # tf.reshape(self.sales(inputs["item_number_of_purchases"]), (-1, 1)),
          # tf.reshape(self.views(inputs["item_number_of_views"]), (-1, 1)),
          # tf.reshape(self.carts(inputs["item_number_of_addtocart"]), (-1, 1)),
          # tf.reshape(self.users(inputs["number_of_unique_visitors"]), (-1, 1)),
          self.property1_embedding(self.product_property1_vocab(inputs["property_1"])),
          self.property2_embedding(self.product_property2_vocab(inputs["property_2"])),
          self.property3_embedding(self.product_property3_vocab(inputs["property_3"])),
          tf.reshape(self.property4(inputs["property_4"]), (-1,1)),
          self.property5_embedding(self.product_property5_vocab(inputs["property_5"])),
          self.property6_embedding(self.product_property5_vocab(inputs["property_6"])),
          tf.reshape(self.property7(inputs["property_7"]), (-1,1)),
          tf.reshape(self.property8(inputs["property_8"]), (-1,1)),
          self.property9_embedding(self.product_property5_vocab(inputs["property_9"])),
          self.property10_embedding(self.product_property10_vocab(inputs["property_10"])),
          self.property11_embedding(self.product_property11_vocab(inputs["property_11"]))
          ], axis=1)
     