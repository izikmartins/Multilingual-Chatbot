from math import sqrt
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# ---------- DATA LOADING ----------
ratings = pd.read_csv('data/ratings.csv', sep=',', encoding='latin-1',
                      usecols=['userId','movieId','rating','timestamp'])
movies = pd.read_csv('data/movies.csv', sep=',', encoding='latin-1',
                     usecols=['movieId','title','genres'])

df_movies = movies
df_ratings = ratings

# Merge and aggregate (for analysis, not critical for recommendations)
merge_ratings_movies = pd.merge(df_movies, df_ratings, on='movieId', how='inner')
merge_ratings_movies = merge_ratings_movies.drop('timestamp', axis=1)

ratings_grouped_by_users = merge_ratings_movies.groupby('userId')['rating'].agg(['count', 'mean'])
ratings_grouped_by_users.columns = ['size', 'mean']

ratings_grouped_by_movies = merge_ratings_movies.groupby('movieId')['rating'].agg(['mean', 'count'])
ratings_grouped_by_movies.columns = ['mean', 'count']

low_rated_movies_filter = ratings_grouped_by_movies['mean'] < 1.5
low_rated_movies = ratings_grouped_by_movies[low_rated_movies_filter]

# ---------- CONTENT-BASED RECOMMENDATION ----------
tfidf_movies_genres = TfidfVectorizer(token_pattern=r'[a-zA-Z0-9\-]+')
df_movies['genres'] = df_movies['genres'].replace(to_replace="(no genres listed)", value="")
tfidf_movies_genres_matrix = tfidf_movies_genres.fit_transform(df_movies['genres'])
cosine_sim_movies = linear_kernel(tfidf_movies_genres_matrix, tfidf_movies_genres_matrix)

def get_recommendations_based_on_genres(movie_title, cosine_sim_movies=cosine_sim_movies):
    idx_movie = df_movies.loc[df_movies['title'].isin([movie_title])]
    idx_movie = idx_movie.index
    if len(idx_movie) == 0:
        return pd.Series(dtype=object)
    sim_scores_movies = list(enumerate(cosine_sim_movies[idx_movie][0]))
    sim_scores_movies = sorted(sim_scores_movies, key=lambda x: x[1], reverse=True)
    sim_scores_movies = sim_scores_movies[1:9]  # 8 similar (exclude self)
    movie_indices = [i[0] for i in sim_scores_movies]
    return df_movies['title'].iloc[movie_indices]

def get_recommendation_content_model(userId):
    recommended_movie_list = []
    movie_list = []
    df_rating_filtered = df_ratings[df_ratings["userId"] == userId]
    for key, row in df_rating_filtered.iterrows():
        movie_list.append((df_movies["title"][row["movieId"] == df_movies["movieId"]]).values)
    for index, movie in enumerate(movie_list):
        for movie_recommended in get_recommendations_based_on_genres(movie[0]):
            recommended_movie_list.append(movie_recommended)
    # Remove already watched
    for movie_title in recommended_movie_list[:]:
        if movie_title in movie_list:
            recommended_movie_list.remove(movie_title)
    return set(recommended_movie_list)

# ---------- SVD COLLABORATIVE FILTERING ----------
reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

SVD_MODEL_PATH = 'svd_model.pkl'

def train_svd():
    """Train SVD and save to disk"""
    print("Training SVD model on full dataset...")
    algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    trainset_full = data.build_full_trainset()
    algo.fit(trainset_full)
    # Save to disk
    with open(SVD_MODEL_PATH, 'wb') as f:
        pickle.dump(algo, f)
    print("SVD model trained and saved to 'svd_model.pkl'")
    return algo

def load_or_train_svd():
    """Load SVD from disk if exists, otherwise train it"""
    try:
        with open(SVD_MODEL_PATH, 'rb') as f:
            algo = pickle.load(f)
        print("SVD model loaded from 'svd_model.pkl'")
        return algo
    except FileNotFoundError:
        return train_svd()

# Load the model ONCE when the module is imported (so GUI gets it instantly)
svd_model = load_or_train_svd()

def hybrid_content_svd_model(userId):
    """
    Hybrid recommendation: content-based candidates + SVD rating predictions.
    Returns top 6 movies with predicted ratings.
    """
    recommended_movies_by_content_model = get_recommendation_content_model(userId)
    recommended_movies_by_content_model = df_movies[
        df_movies.apply(lambda movie: movie["title"] in recommended_movies_by_content_model, axis=1)
    ]
    for key, columns in recommended_movies_by_content_model.iterrows():
        predict = svd_model.predict(userId, columns["movieId"])
        recommended_movies_by_content_model.loc[key, "svd_rating"] = predict.est
    return recommended_movies_by_content_model.sort_values("svd_rating", ascending=False).iloc[0:6]

# ---------- MAIN (only runs when executing hybrid.py directly) ----------
if __name__ == "__main__":
    print("="*50)
    print("TRAINING SVD AND RUNNING CROSS-VALIDATION")
    print("="*50)
    algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    
    # Train and save the full model for GUI use
    train_svd()
    print("\n✅ SVD model saved. You can now run the GUI.")