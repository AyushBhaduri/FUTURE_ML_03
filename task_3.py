import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. Load the MovieLens Dataset
try:
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    print("✅ Datasets loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'movies.csv' or 'ratings.csv' not found. Please upload them to your environment.")

# 2. Data Preprocessing
# Clean movie titles and extract genres
movies['genres'] = movies['genres'].str.replace('|', ' ')

# 3. Build Content-Based Engine (TF-IDF)
# This converts genres into a mathematical matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate Cosine Similarity (The 'Distance' between movies)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 4. Recommendation Logic
def get_recommendations(movie_title, num_recommendations=5):
    # Check if movie exists in our data
    if movie_title not in movies['title'].values:
        return f"Movie '{movie_title}' not found in database. Check spelling/year."

    # Get the index of the movie that matches the title
    idx = movies.index[movies['title'] == movie_title][0]

    # Get pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar movies (skip the first one as it is the movie itself)
    sim_scores = sim_scores[1:num_recommendations+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top most similar movies
    return movies[['title', 'genres']].iloc[movie_indices]

# 5. Test the System
print("\n--- Recommendations for 'Toy Story (1995)' ---")
print(get_recommendations('Toy Story (1995)'))

print("\n--- Recommendations for 'Jumanji (1995)' ---")
print(get_recommendations('Jumanji (1995)'))