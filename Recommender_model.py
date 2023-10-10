#Recommender system models are a class of machine learning models 
#that are used to recommend items to users based on their historical 
#preferences or behavior

# pip install tensorflow_recommenders




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress warnings

import random
import numpy as np
import tensorflow as tf

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)

from typing import Dict, Text

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


# Load the movielens/100k-ratings train dataset
ratings = tfds.load("movielens/latest-small-ratings", split="train")

# Load the movielens/100k-movies train dataset
movies = tfds.load("movielens/latest-small-movies", split="train")

# Select movie title and user id from the ratings dataset
ratings = ratings.map(lambda x: {"movie_title": x["movie_title"], "user_id": x["user_id"]})

# Select movie title from the movies dataset
movies = movies.map(lambda x: x["movie_title"])


# ************************************ Build vocabulary to convert user ids from the ratings

# Create a list of all movie titles
user_ids = []
for example in ratings:
    user_ids.append(example['user_id'].numpy().decode('utf-8'))

# Create a StringLookup layer to build the vocabulary
title_lookup = tf.keras.layers.StringLookup()
title_lookup.adapt(user_ids)

# Print the vocabulary
print(title_lookup.get_vocabulary())


# Print the size of the user id vocabulary

vocabulary_size_rating = len(title_lookup.get_vocabulary())
print("Vocabulary size:", vocabulary_size_rating)

# Print the id of the 30th and 31st users

titles = list(ratings.map(lambda x: x['user_id']).take(31).as_numpy_iterator()) 
embedding_ids_30th = title_lookup(tf.constant([titles[29]])).numpy().flatten()
embedding_ids_31st = title_lookup(tf.constant([titles[30]])).numpy().flatten()

# Print the embedding IDs
print("30th movie embedding IDs:", embedding_ids_30th)
print("31st movie embedding IDs:", embedding_ids_31st)


# Build vocabulary to convert user ids into integer indices for embedding layers
user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup()
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))




# ****************************** Build vocabulary to convert movie titles into integer indices for embedding layers


movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup()
movie_titles_vocabulary.adapt(movies)














# Define a Sequential user model using the user id vocabulary you built to create embedding with output dimension of 28
user_model = tf.keras.Sequential([
    user_ids_vocabulary,
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 28)
])

# Define a Sequential movie model using the movie title vocabulary you built to create embedding with output dimension of 28
movie_model = tf.keras.Sequential([
    movie_titles_vocabulary,
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 28)
])

# Define a TFRS Retrieval task object with FactorizedTopK as the metric
task = tfrs.tasks.Retrieval(
    metrics=tfrs.metrics.FactorizedTopK(
        movies.batch(256).map(movie_model)
    )
)



# ************************************* Make the task object batch movie features into batches of 256, and apply the movie model you built to each batch

class MovieTower(tfrs.models.Model):
    def __init__(self, movie_model):
        super().__init__()
        self.movie_model = movie_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                movies.batch(256).map(self.movie_model)
            )
        )

    def compute_loss(self, features, training=False):
        movie_embeddings = self.movie_model(features['movie_title'])
        return self.task(None, movie_embeddings)

movie_tower = MovieTower(movie_model)
movie_tower.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.75))

import tensorflow as tf

class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["movie_title"])

    return self.task(user_embeddings, movie_embeddings)


# Create a MovieLensModel object using the user model, movie model, and task you created above
model = MovieLensModel(user_model, movie_model, task)

# Compile the model with the Adagrad optimizer with a .65 learning rate
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.75))

# Train the model with the ratings dataset, batch size of 3500, and 3 epochs
model.fit(ratings.batch(3500), epochs=3)




# *********************************** Create a brute-force nearest neighbor search index for a recommendation model

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(movies.batch(100).map(lambda title: movie_model(title)))





# ************************************ Use the index created above to make top 5 recommendations for user 17




