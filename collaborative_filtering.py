import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed data
user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col=0)

# Compute similarity matrix
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_user_recommendations(user_id, n_recommendations=5):
    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    recommendations = pd.Series(dtype='float64')

    for similar_user in similar_users.index:
        if similar_user == user_id:
            continue
        similar_user_ratings = user_item_matrix.loc[similar_user]
        for movie_id, rating in similar_user_ratings.items():
            if pd.isna(user_ratings[movie_id]):
                if movie_id not in recommendations:
                    recommendations[movie_id] = 0
                recommendations[movie_id] += rating * similar_users[similar_user]
    
    recommendations = recommendations.sort_values(ascending=False)
    return recommendations.head(n_recommendations).index.tolist()

# Example usage
user_id = 1
recommended_movies = get_user_recommendations(user_id)
print("Recommended movies for user", user_id, ":", recommended_movies)
