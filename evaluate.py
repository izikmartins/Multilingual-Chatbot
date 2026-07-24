import json
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
import nltk
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIG ----------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("MULTILINGUAL CHATBOT RECOMMENDATION SYSTEM - EVALUATION")
print("=" * 60)

# ---------- 1. EVALUATE CHATBOT INTENT CLASSIFIER ----------
print("\n" + "=" * 60)
print("1. CHATBOT INTENT CLASSIFIER EVALUATION")
print("=" * 60)

# Load your pre-trained vocabulary and classes
lemmatizer = WordNetLemmatizer()
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained Keras model
model = load_model('chatbot_model.h5')

# Load raw intents
with open('data/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Prepare dataset (patterns and tags)
patterns = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        if pattern:  # skip empty patterns
            patterns.append(pattern)
            tags.append(intent['tag'])

# Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    patterns, tags, test_size=0.2, random_state=RANDOM_SEED
)

# Preprocessing function 
def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Convert test patterns to feature vectors
test_features = np.array([bag_of_words(pattern) for pattern in X_test])

# Predict
y_pred_probs = model.predict(test_features, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Map predictions back to tag names
tag_to_idx = {tag: i for i, tag in enumerate(classes)}
idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
y_test_int = np.array([tag_to_idx[tag] for tag in y_test])

# Metrics
acc = accuracy_score(y_test_int, y_pred)
prec = precision_score(y_test_int, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test_int, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test_int, y_pred, average='weighted', zero_division=0)

print(f"Test set size: {len(X_test)} patterns")
print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print(f"F1-score:      {f1:.4f}")

# ---------- 2. SVD EVALUATION ----------
print("\n" + "=" * 60)
print("2. SVD COLLABORATIVE FILTERING EVALUATION")
print("=" * 60)

ratings = pd.read_csv('data/ratings.csv')
reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

print("5-Fold Cross-Validation RMSE/MAE:")
cv_results = cross_validate(
    SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42),
    data, measures=['RMSE', 'MAE'], cv=5, verbose=True
)

print(f"\nMean RMSE: {np.mean(cv_results['test_rmse']):.4f}")
print(f"Mean MAE:  {np.mean(cv_results['test_mae']):.4f}")

# ---------- 3. CONTENT-BASED HIT/FAULT (OPTIMIZED) ----------
print("\n" + "=" * 60)
print("3. CONTENT-BASED MODEL EVALUATION (HIT/FAULT)")
print("=" * 60)

movies = pd.read_csv('data/movies.csv')
df_movies = movies

# Build TF-IDF matrix
tfidf_movies_genres = TfidfVectorizer(token_pattern=r'[a-zA-Z0-9\-]+')
df_movies['genres'] = df_movies['genres'].replace(to_replace="(no genres listed)", value="")
tfidf_movies_genres_matrix = tfidf_movies_genres.fit_transform(df_movies['genres'])
cosine_sim_movies = linear_kernel(tfidf_movies_genres_matrix, tfidf_movies_genres_matrix)

# RECOMMENDATION FUNCTION 
def get_recommendations_based_on_genres(movie_title, cosine_sim_movies=cosine_sim_movies):
    idx_movie = df_movies.loc[df_movies['title'].isin([movie_title])]
    idx_movie = idx_movie.index
    if len(idx_movie) == 0:
        return pd.Series(dtype=object)
    sim_scores_movies = list(enumerate(cosine_sim_movies[idx_movie][0]))
    sim_scores_movies = sorted(sim_scores_movies, key=lambda x: x[1], reverse=True)
    # EXACT MATCH: 9 similar movies, excluding the movie itself
    sim_scores_movies = sim_scores_movies[1:10]  # <-- THIS IS THE KEY LINE
    movie_indices = [i[0] for i in sim_scores_movies]
    return df_movies['title'].iloc[movie_indices]


def get_movie_label(movie_id, x_data, y_data):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_data, y_data)
    y_pred = classifier.predict(x_data[movie_id])
    return y_pred

true_count = 0
false_count = 0
x_data = tfidf_movies_genres_matrix
y_data = df_movies['genres']

print("Processing... (this will take ~10-15 minutes, matching your notebook)")
for key, columns in df_movies.iterrows():
    movies_recommended = get_recommendations_based_on_genres(columns["title"])
    if len(movies_recommended) == 0:
        continue
    
    for movie_idx in movies_recommended.index:
        pred = get_movie_label(movie_idx, x_data, y_data)
        
        for p in pred:
            if p == columns["genres"]:
                true_count += 1
            else:
                false_count += 1

    if (key + 1) % 500 == 0:
        print(f"Processed {key + 1} movies...")

total = true_count + false_count
hit = true_count / total if total > 0 else 0
fault = false_count / total if total > 0 else 0

print(f"\nTrue Count:  {true_count}")
print(f"False Count: {false_count}")
print(f"Total:       {total}")
print(f"HIT:         {hit:.8f}")   # Should output ~0.92934288
print(f"FAULT:       {fault:.8f}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print("\nSummary of Results:")
print(f"  Chatbot Accuracy:  {acc:.4f}")
print(f"  Chatbot F1-score:  {f1:.4f}")
print(f"  SVD RMSE:          {np.mean(cv_results['test_rmse']):.4f}")
print(f"  SVD MAE:           {np.mean(cv_results['test_mae']):.4f}")
print(f"  Content Hit Rate:  {hit:.8f}")
print(f"  Content Fault Rate:{fault:.8f}")