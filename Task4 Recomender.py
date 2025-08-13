import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

# Load datasets
movies = pd.read_csv("/content/movies.csv")
ratings = pd.read_csv("/content/ratings.csv")

# Create user-item rating matrix
ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Convert to numpy array for SVD
ratings_np = ratings_matrix.to_numpy()

# Apply SVD (Singular Value Decomposition)
U, sigma, Vt = svds(ratings_np, k=50)  # Keeping 50 latent factors
sigma = np.diag(sigma)

# Reconstruct the ratings matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Convert back to DataFrame
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=ratings_matrix.index, columns=ratings_matrix.columns)

def recommend_movies(user_id, num_recommendations=5):
    # Get predicted ratings for the user
    user_ratings = predicted_ratings_df.loc[user_id]

    # Sort movies by predicted rating in descending order
    recommended_movie_ids = user_ratings.sort_values(ascending=False).index[:num_recommendations]

    # Get movie titles
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]

    return recommended_movies[['movieId', 'title']]

# Example: Recommend top 5 movies for user ID 1
print(recommend_movies(user_id=1, num_recommendations=5))
