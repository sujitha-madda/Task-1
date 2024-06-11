from flask import Flask, request, jsonify
import pandas as pd
from collaborative_filtering import get_user_recommendations

app = Flask(__name__)

# Load preprocessed data
ratings = pd.read_csv('ratings.csv')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = 1  # In a real application, fetch the logged-in user's ID
    movie_id = int(data['movieId'])
    rating = float(data['rating'])
    
    # Add new rating to the dataset
    global ratings
    ratings = ratings.append({'userId': user_id, 'movieId': movie_id, 'rating': rating}, ignore_index=True)
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Save updated data
    user_item_matrix.to_csv('user_item_matrix.csv', index=False)
    
    # Get recommendations
    recommendations = get_user_recommendations(user_id)
    
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
