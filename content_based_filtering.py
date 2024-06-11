import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed data
movies = pd.read_csv('preprocessed_movies.csv')

# Create feature vectors for movies
tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
tfidf_matrix = tfidf.fit_transform(movies['genres'].apply(lambda x: ' '.join(x)))

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['movieId'], columns=movies['movieId'])

def get_movie_recommendations(movie_id, n_recommendations=5):
    movie_similarities = cosine_sim_df[movie_id].sort_values(ascending=False)
    return movie_similarities.head(n_recommendations).index.tolist()

# Example usage
movie_id = 1
recommended_movies = get_movie_recommendations(movie_id)
print("Movies similar to", movie_id, ":", recommended_movies)
