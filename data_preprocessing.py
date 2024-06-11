import pandas as pd

# Load movie data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocess data
movies['genres'] = movies['genres'].str.split('|')

# Create a user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_item_matrix = user_item_matrix.fillna(0)

# Save preprocessed data
movies.to_csv('preprocessed_movies.csv', index=False)
user_item_matrix.to_csv('user_item_matrix.csv', index=False)
