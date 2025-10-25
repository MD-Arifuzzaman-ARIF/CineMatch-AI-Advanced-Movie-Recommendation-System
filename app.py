import gradio as gr
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic, SVDpp
import pickle
import tensorflow as tf
from tensorflow import keras
import requests
from urllib.parse import quote
import os
from huggingface_hub import hf_hub_download, snapshot_download

# TMDB API Configuration
TMDB_API_KEY = "e351a1c171428203e75388658d54ba5e"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Hugging Face Model Repository
HF_REPO_ID = "marshmallow343/savedmodels"

# Global variables for collaborative filtering models
svd_model = None
user_based_model = None
ncf_model = None
svdpp_model = None
item_based_model = None
movie_titles = {}
df_ratings = None
user_to_index = {}
item_to_index = {}
index_to_user = {}
index_to_item = {}
n_users = 0
n_items = 0

# Global variables for content-based model
cb_tfidf = None
cb_tfidf_matrix = None
cb_cosine_sim = None
cb_movies_df = None

# Try to import rapidfuzz for better fuzzy matching; fallback to difflib
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    from difflib import get_close_matches
    HAVE_RAPIDFUZZ = False

# ---------------------------
# Helper Functions
# ---------------------------

def get_movie_poster(movie_title, year=None):
    """Fetch movie poster from TMDB API"""
    try:
        clean_title = movie_title.split('(')[0].strip()
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {'api_key': TMDB_API_KEY, 'query': clean_title, 'year': year}
        response = requests.get(search_url, params=params, timeout=5)
        data = response.json()
        
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"{TMDB_IMAGE_BASE}{poster_path}"
        
        return "https://via.placeholder.com/500x750/1a1a2e/eebbc3?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750/1a1a2e/eebbc3?text=No+Poster"

def download_model_from_hf(filename, subfolder=None, use_root=False):
    """Download model file from Hugging Face Hub"""
    try:
        print(f"📥 Downloading {filename} from Hugging Face...")
        
        # For content_based_model.pkl, it's in the root directory
        if use_root:
            full_path = filename
        elif subfolder:
            full_path = f"saved_models/{subfolder}/{filename}"
        else:
            full_path = f"saved_models/{filename}"
        
        file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=full_path,
            repo_type="model"
        )
        print(f"✅ Downloaded {filename} to {file_path}")
        return file_path
    except Exception as e:
        print(f"⚠️ Could not download {filename}: {str(e)[:100]}")
        return None

# ---------------------------
# Load Collaborative Filtering Models
# ---------------------------

def load_models_and_data():
    """Load all trained models and data"""
    global svd_model, svdpp_model, user_based_model, item_based_model, ncf_model, movie_titles, df_ratings
    global user_to_index, item_to_index, index_to_user, index_to_item, n_users, n_items
    
    print("🎬 Loading MovieLens data and models...")
    
    # Load ratings and movies
    url_data = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    url_item = "http://files.grouplens.org/datasets/movielens/ml-100k/u.item"
    
    column_names = ["user_id", "item_id", "rating", "timestamp"]
    df_ratings = pd.read_csv(url_data, sep="\t", names=column_names)
    
    movie_columns = [
        "item_id", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    df_movies = pd.read_csv(url_item, sep="|", names=movie_columns, encoding="latin-1")
    movie_titles = dict(zip(df_movies["item_id"], df_movies["title"]))
    
    # Prepare data for Surprise library
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_ratings[["user_id", "item_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    
    # Download and load SVD model
    print("Loading SVD model...")
    svd_path = download_model_from_hf("svd_model.pkl")
    if svd_path:
        with open(svd_path, 'rb') as f:
            svd_model = pickle.load(f)
        print("✅ SVD model loaded from Hugging Face")
    else:
        print("⚠️ Training SVD model from scratch...")
        svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
        svd_model.fit(trainset)
    
    # Download and load SVD++ model
    print("Loading SVD++ model...")
    svdpp_path = download_model_from_hf("svdpp_model.pkl")
    if svdpp_path:
        with open(svdpp_path, 'rb') as f:
            svdpp_model = pickle.load(f)
        print("✅ SVD++ model loaded from Hugging Face")
    else:
        print("⚠️ Training SVD++ model from scratch...")
        svdpp_model = SVDpp(n_factors=100, n_epochs=20, random_state=42)
        svdpp_model.fit(trainset)
    
    # Download and load Item-based CF model
    print("Loading Item-based CF model...")
    item_cf_path = download_model_from_hf("item_based_cf.pkl")
    if item_cf_path:
        with open(item_cf_path, 'rb') as f:
            item_based_model = pickle.load(f)
        print("✅ Item-based CF model loaded from Hugging Face")
    else:
        print("⚠️ Training Item-based CF model from scratch...")
        item_based_model = KNNBasic(sim_options={'user_based': False, 'name': 'cosine'})
        item_based_model.fit(trainset)
    
    # Download and load User-based CF model
    print("Loading User-based CF model...")
    user_cf_path = download_model_from_hf("user_based_cf.pkl")
    if user_cf_path:
        with open(user_cf_path, 'rb') as f:
            user_based_model = pickle.load(f)
        print("✅ User-based CF model loaded from Hugging Face")
    else:
        print("⚠️ Training User-based CF model from scratch...")
        user_based_model = KNNBasic(sim_options={'user_based': True, 'name': 'cosine'})
        user_based_model.fit(trainset)
    
    # Prepare data for Neural Network
    print("Preparing data for Neural Network...")
    
    # Try to load mappings from HuggingFace
    mappings_path = download_model_from_hf("mappings.pkl")
    if mappings_path:
        try:
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
            user_to_index = mappings.get('user_to_index', {})
            item_to_index = mappings.get('item_to_index', {})
            index_to_user = mappings.get('index_to_user', {})
            index_to_item = mappings.get('index_to_item', {})
            n_users = mappings.get('n_users', 0)
            n_items = mappings.get('n_items', 0)
            print("✅ Mappings loaded from Hugging Face")
        except:
            print("⚠️ Could not load mappings, creating new ones...")
            mappings_path = None
    
    if not mappings_path:
        # Create mappings from scratch
        user_ids = df_ratings['user_id'].unique()
        item_ids = df_ratings['item_id'].unique()
        
        n_users = len(user_ids)
        n_items = len(item_ids)
        
        user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
        index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
        index_to_item = {idx: item_id for item_id, idx in item_to_index.items()}
    
    # Download and load NCF model
    print("Loading Neural Collaborative Filtering model...")
    try:
        print("📥 Downloading NCF model from Hugging Face...")
        model_path = snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns="saved_models/ncf_model/**",
            repo_type="model"
        )
        
        ncf_model_path = os.path.join(model_path, "saved_models", "ncf_model")
        
        if os.path.exists(ncf_model_path):
            ncf_model = keras.models.load_model(ncf_model_path)
            print("✅ NCF model loaded from Hugging Face")
        else:
            raise Exception("NCF model directory not found in downloaded files")
        
    except Exception as e:
        print(f"⚠️ Could not load NCF model from HuggingFace: {str(e)[:100]}")
        print("Building new NCF model from scratch...")
        
        from keras import layers, Model, regularizers
        
        user_input = layers.Input(shape=(1,))
        item_input = layers.Input(shape=(1,))
        
        user_embedding = layers.Embedding(n_users, 50, embeddings_regularizer=regularizers.l2(1e-6))(user_input)
        item_embedding = layers.Embedding(n_items, 50, embeddings_regularizer=regularizers.l2(1e-6))(item_input)
        
        user_vec = layers.Flatten()(user_embedding)
        item_vec = layers.Flatten()(item_embedding)
        
        concat = layers.Concatenate()([user_vec, item_vec])
        dense1 = layers.Dense(128, activation='relu')(concat)
        dropout1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(64, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.3)(dense2)
        output = layers.Dense(1)(dropout2)
        
        ncf_model = Model(inputs=[user_input, item_input], outputs=output)
        ncf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Quick training
        df_ratings['user_idx'] = df_ratings['user_id'].map(user_to_index)
        df_ratings['item_idx'] = df_ratings['item_id'].map(item_to_index)
        X = [df_ratings['user_idx'].values, df_ratings['item_idx'].values]
        y = df_ratings['rating'].values
        
        print("Training NCF model (this may take a few minutes)...")
        ncf_model.fit(X, y, epochs=5, batch_size=256, verbose=0, validation_split=0.1)
    
    print("✅ All collaborative filtering models loaded successfully!")

# ---------------------------
# Load Content-Based Model
# ---------------------------

def load_content_based_model():
    """Load content-based recommendation model remotely from Hugging Face"""
    global cb_tfidf, cb_tfidf_matrix, cb_cosine_sim, cb_movies_df
    
    try:
        print("🌐 Loading content-based model remotely from Hugging Face...")
        
        # Direct URL using resolve endpoint (like accessing it via browser)
        model_url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/content_based_model.pkl"
        
        print(f"📡 Fetching from: {model_url}")
        response = requests.get(model_url, timeout=30)
        
        if response.status_code == 200:
            # Load pickle directly from response content
            data = pickle.loads(response.content)
            
            cb_tfidf = data.get("tfidf_vectorizer")
            cb_tfidf_matrix = data.get("tfidf_matrix")
            cb_cosine_sim = data.get("cosine_sim")
            cb_movies_df = data.get("movies_df")
            
            # Ensure item_id is int
            if cb_movies_df is not None:
                cb_movies_df["item_id"] = cb_movies_df["item_id"].astype(int)
            
            print("✅ Content-based model loaded successfully from remote!")
            return True
        else:
            print(f"⚠️ Failed to fetch model. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading content-based model: {str(e)}")
        return False

# ---------------------------
# Recommendation Functions
# ---------------------------

def hybrid_recommend(user_id, N=10, weights=None):
    """Hybrid recommendation combining all models"""
    if weights is None:
        weights = {'svdpp': 0.30, 'item_cf': 0.30, 'user_cf': 0.20, 'ncf': 0.20}
    
    all_movie_ids = list(movie_titles.keys())
    rated_movies = df_ratings[df_ratings['user_id'] == user_id]['item_id'].values
    unrated_movies = [mid for mid in all_movie_ids if mid not in rated_movies]
    
    predictions = {}
    
    for movie_id in unrated_movies:
        score = 0
        total_weight = 0
        
        # SVD++
        if 'svdpp' in weights:
            svdpp_pred = svdpp_model.predict(user_id, movie_id).est
            score += weights['svdpp'] * svdpp_pred
            total_weight += weights['svdpp']
        
        # Item-CF
        if 'item_cf' in weights:
            item_pred = item_based_model.predict(user_id, movie_id).est
            score += weights['item_cf'] * item_pred
            total_weight += weights['item_cf']
        
        # User-CF
        if 'user_cf' in weights:
            user_pred = user_based_model.predict(user_id, movie_id).est
            score += weights['user_cf'] * user_pred
            total_weight += weights['user_cf']
        
        # NCF
        if 'ncf' in weights and user_id in user_to_index and movie_id in item_to_index:
            try:
                user_idx = user_to_index[user_id]
                item_idx = item_to_index[movie_id]
                X_pred = np.array([[user_idx, item_idx]], dtype=np.int64)
                ncf_pred = ncf_model.predict(X_pred, verbose=0)[0][0]
                ncf_pred = np.clip(ncf_pred, 1, 5)
                score += weights['ncf'] * ncf_pred
                total_weight += weights['ncf']
            except Exception:
                pass
        
        if total_weight > 0:
            predictions[movie_id] = score / total_weight
    
    top_n = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:N]
    return top_n

def recommend_movies_professional(user_id, num_recommendations, model_choice):
    """Generate recommendations using selected model"""
    try:
        user_id = int(user_id)
        num_recommendations = int(num_recommendations)
        
        if user_id < 1 or user_id > 943:
            return create_error_html("Please enter a valid User ID between 1 and 943")
        
        if num_recommendations < 1 or num_recommendations > 20:
            return create_error_html("Please select between 1 and 20 recommendations")
        
        all_movie_ids = list(movie_titles.keys())
        rated_movies = df_ratings[df_ratings['user_id'] == user_id]['item_id'].values
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        unrated_movies = [mid for mid in all_movie_ids if mid not in rated_movies]
        
        if not unrated_movies:
            return create_error_html(f"User {user_id} has rated all movies!")
        
        # Get predictions based on model choice
        predictions = []
        
        if model_choice == "SVD (Matrix Factorization)":
            for movie_id in unrated_movies:
                pred = svd_model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))
        
        elif model_choice == "SVD++ (Enhanced MF)":
            for movie_id in unrated_movies:
                pred = svdpp_model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))
        
        elif model_choice == "User-based Collaborative Filtering":
            for movie_id in unrated_movies:
                pred = user_based_model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))
        
        elif model_choice == "Item-based Collaborative Filtering":
            for movie_id in unrated_movies:
                pred = item_based_model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))
        
        elif model_choice == "Neural Collaborative Filtering":
            if user_id not in user_to_index:
                return create_error_html("User ID not found in neural network mapping")
            
            user_idx = user_to_index[user_id]
            unrated_item_indices = [item_to_index[mid] for mid in unrated_movies if mid in item_to_index]
            
            user_indices = np.full(len(unrated_item_indices), user_idx, dtype=np.int64)
            item_indices = np.array(unrated_item_indices, dtype=np.int64)
            X_pred = np.column_stack([user_indices, item_indices])
            
            pred_ratings = ncf_model.predict(X_pred, verbose=0).flatten()
            pred_ratings = np.clip(pred_ratings, 1, 5)
            
            for idx, rating in zip(unrated_item_indices, pred_ratings):
                movie_id = index_to_item[idx]
                predictions.append((movie_id, rating))
        
        elif model_choice == "Hybrid Model (Recommended)":
            top_n = hybrid_recommend(user_id, num_recommendations)
            html_output = create_professional_html(user_id, top_n, user_ratings, model_choice)
            return html_output
        
        # Sort and limit results for non-hybrid models
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:num_recommendations]
        
        html_output = create_professional_html(user_id, top_n, user_ratings, model_choice)
        return html_output
        
    except Exception as e:
        return create_error_html(f"Error: {str(e)}")

# ---------------------------
# Content-Based Search Functions
# ---------------------------

def fuzzy_best_match(query, titles_list):
    """Find best matching title using fuzzy matching"""
    q = str(query).strip()
    if not q:
        return None, 0.0
    
    if HAVE_RAPIDFUZZ:
        res = rf_process.extractOne(q, titles_list, scorer=rf_fuzz.WRatio)
        if res:
            return res[0], float(res[1]) / 100.0
        return None, 0.0
    else:
        # difflib fallback
        matches = get_close_matches(q, titles_list, n=1, cutoff=0.0)
        if matches:
            return matches[0], 1.0
        return None, 0.0

def get_top_similar_from_best_match(query, top_n=6):
    """
    Auto-pick best fuzzy match, then return top_n similar movies (excluding the movie itself).
    Returns: (best_matched_title, list of tuples: (movie_id, title, sim_score))
    """
    if cb_movies_df is None or cb_cosine_sim is None:
        return None, []
    
    titles = cb_movies_df["title"].tolist()
    best_title, best_score = fuzzy_best_match(query, titles)
    
    if best_title is None:
        return None, []
    
    # Find item_id and index
    row = cb_movies_df[cb_movies_df["title"] == best_title].iloc[0]
    item_id = int(row["item_id"])
    idx = cb_movies_df[cb_movies_df["item_id"] == item_id].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cb_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip first (itself) and get top_n
    recs = []
    for movie_idx, sim in sim_scores[1:top_n+1]:
        r = cb_movies_df.iloc[movie_idx]
        recs.append((int(r["item_id"]), r["title"], float(sim)))
        if len(recs) >= top_n:
            break
    
    return best_title, recs

# ---------------------------
# HTML Generation Functions
# ---------------------------

def create_error_html(message):
    """Create professional error message"""
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 15px; text-align: center;">
        <div style="background: rgba(255,255,255,0.1); padding: 30px; border-radius: 10px; backdrop-filter: blur(10px);">
            <h2 style="color: #ffffff; margin: 0; font-size: 28px;">⚠️ Oops!</h2>
            <p style="color: #ffffff; margin-top: 15px; font-size: 18px;">{message}</p>
        </div>
    </div>
    """

def create_professional_html(user_id, recommendations, user_ratings, model_name):
    """Create beautiful professional HTML for recommendations"""
    
    avg_user_rating = user_ratings['rating'].mean()
    total_rated = len(user_ratings)
    
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        .container {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px;
            border-radius: 20px;
            font-family: 'Poppins', sans-serif;
        }}
        
        .hero {{
            background: linear-gradient(135deg, #0f3460 0%, #533483 50%, #e94560 100%);
            padding: 35px;
            border-radius: 15px;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(233, 69, 96, 0.3);
        }}
        
        .hero::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        }}
        
        .hero-content {{
            position: relative;
            z-index: 1;
        }}
        
        .hero-title {{
            font-size: 42px;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 10px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .hero-subtitle {{
            color: rgba(255,255,255,0.9);
            font-size: 16px;
            margin: 0 0 20px 0;
        }}
        
        .model-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 8px 20px;
            border-radius: 25px;
            font-size: 14px;
            color: #ffffff;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }}
        
        .stat-card {{
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s ease;
        }}
        
        .stat-card:hover {{
            background: rgba(255,255,255,0.1);
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }}
        
        .stat-value {{
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, #e94560, #0f3460);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: rgba(255,255,255,0.7);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 5px;
        }}
        
        .section-title {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #e94560, #0f3460);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 30px 0 20px 0;
        }}
        
        .movie-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        
        .movie-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 15px;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            cursor: pointer;
            position: relative;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .movie-card:hover {{
            transform: scale(1.05) translateY(-10px);
            box-shadow: 0 15px 40px rgba(233,69,96,0.4);
            border-color: #e94560;
        }}
        
        .movie-poster {{
            width: 100%;
            height: 300px;
            object-fit: cover;
        }}
        
        .movie-rank {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: linear-gradient(135deg, #e94560, #0f3460);
            color: white;
            width: 45px;
            height: 45px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 18px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 2;
        }}
        
        .movie-info {{
            padding: 15px;
            background: linear-gradient(to top, #1a1a2e, transparent);
        }}
        
        .movie-title {{
            color: #ffffff;
            font-size: 14px;
            font-weight: 600;
            margin: 0 0 10px 0;
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }}
        
        .rating-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, rgba(233,69,96,0.2), rgba(15,52,96,0.2));
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid rgba(233,69,96,0.3);
        }}
        
        .rating-stars {{
            color: #FFD700;
            font-size: 14px;
        }}
        
        .rating-text {{
            color: #ffffff;
            font-weight: 600;
            font-size: 14px;
        }}
    </style>
    
    <div class="container">
        <div class="hero">
            <div class="hero-content">
                <h1 class="hero-title">🎬 Your Personalized Recommendations</h1>
                <p class="hero-subtitle">Powered by {model_name} • User #{user_id}</p>
                <span class="model-badge">🤖 {model_name}</span>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-value">{total_rated}</span>
                        <span class="stat-label">Movies Rated</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">{avg_user_rating:.1f}</span>
                        <span class="stat-label">Avg Rating</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">{len(recommendations)}</span>
                        <span class="stat-label">New Picks</span>
                    </div>
                </div>
            </div>
        </div>
        
        <h2 class="section-title">🔥 Top Recommendations for You</h2>
        <div class="movie-grid">
    """
    
    for i, (movie_id, rating) in enumerate(recommendations, 1):
        title = movie_titles.get(movie_id, f"Movie {movie_id}")
        year = None
        if '(' in title and ')' in title:
            year_str = title.split('(')[-1].split(')')[0]
            if year_str.isdigit():
                year = year_str
        
        poster_url = get_movie_poster(title, year)
        stars = "⭐" * int(round(rating))
        
        html += f"""
            <div class="movie-card">
                <div class="movie-rank">{i}</div>
                <img src="{poster_url}" alt="{title}" class="movie-poster" 
                     onerror="this.src='https://via.placeholder.com/500x750/1a1a2e/e94560?text={quote(title[:20])}'">
                <div class="movie-info">
                    <h3 class="movie-title">{title}</h3>
                    <div class="rating-badge">
                        <span class="rating-stars">{stars}</span>
                        <span class="rating-text">{rating:.1f}</span>
                    </div>
                </div>
            </div>
        """
    
    html += """
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1); text-align: center;">
            <p style="color: rgba(255,255,255,0.6); font-size: 14px; margin: 0;">
                MovieLens 100K Dataset • TMDB API • DataSynthis ML Project
            </p>
        </div>
    </div>
    """
    
    return html

def render_content_based_html(recs, heading=None):
    """
    Create HTML for content-based recommendations in 3x2 grid
    recs: list of (movie_id, title, score)
    """
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        .cb-container {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px;
            border-radius: 20px;
            font-family: 'Poppins', sans-serif;
        }
        
        .cb-heading {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #e94560, #0f3460);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 25px 0;
            text-align: center;
        }
        
        .cb-movie-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }
        
        .cb-movie-card {
            background: rgba(255,255,255,0.03);
            border-radius: 15px;
            overflow: hidden;
            position: relative;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            cursor: pointer;
        }
        
        .cb-movie-card:hover {
            transform: scale(1.05) translateY(-10px);
            box-shadow: 0 15px 40px rgba(233,69,96,0.4);
            border-color: #e94560;
        }
        
        .cb-movie-poster {
            width: 100%;
            height: 300px;
            object-fit: cover;
            display: block;
        }
        
        .cb-movie-info {
            padding: 15px;
            background: linear-gradient(to top, #1a1a2e, transparent);
        }
        
        .cb-movie-title {
            font-size: 14px;
            margin: 0 0 10px 0;
            font-weight: 600;
            color: #ffffff;
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        
        .cb-similarity-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, rgba(233,69,96,0.2), rgba(15,52,96,0.2));
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid rgba(233,69,96,0.3);
            color: #ffffff;
            font-weight: 600;
            font-size: 13px;
        }
    </style>
    """
    
    heading_html = f'<h2 class="cb-heading">{heading}</h2>' if heading else ""
    
    cards = []
    for i, (mid, title, score) in enumerate(recs, start=1):
        year = None
        if "(" in title and ")" in title:
            y = title.split("(")[-1].split(")")[0]
            if y.isdigit():
                year = y
        
        poster = get_movie_poster(title, year)
        
        card = f"""
        <div class="cb-movie-card">
            <img src="{poster}" class="cb-movie-poster" alt="{title}" 
                 onerror="this.src='https://via.placeholder.com/500x750/1a1a2e/e94560?text={quote(title[:20])}';">
            <div class="cb-movie-info">
                <div class="cb-movie-title">{title}</div>
                <div class="cb-similarity-badge">
                    🎯 Similarity: {score:.2%}
                </div>
            </div>
        </div>
        """
        cards.append(card)
    
    grid = '<div class="cb-movie-grid">' + "\n".join(cards) + "</div>"
    
    return css + '<div class="cb-container">' + heading_html + grid + "</div>"

# ---------------------------
# Initialize Models on Startup
# ---------------------------

print("=" * 60)
print("🎬 CINEMATCH AI - INITIALIZATION")
print("=" * 60)

# Load collaborative filtering models
load_models_and_data()

# Load content-based model
print("\n🔍 Loading content-based recommendation model...")
content_model_loaded = load_content_based_model()

if not content_model_loaded:
    print("⚠️ Content-based recommendations will not be available")
    cb_movies_df = None
    cb_cosine_sim = None

print("\n" + "=" * 60)
print("✅ ALL SYSTEMS READY")
print("=" * 60 + "\n")

# ---------------------------
# Gradio Interface
# ---------------------------

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    font-family: 'Poppins', sans-serif !important;
}

.header {
    background: linear-gradient(135deg, #0f3460 0%, #533483 50%, #e94560 100%);
    padding: 40px;
    border-radius: 20px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 15px 50px rgba(233, 69, 96, 0.3);
}

.header h1 {
    font-size: 56px;
    font-weight: 700;
    color: white;
    margin: 0;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
}

button.primary {
    background: linear-gradient(135deg, #e94560 0%, #0f3460 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 18px !important;
    padding: 15px 40px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 10px 30px rgba(233,69,96,0.5) !important;
}
"""

with gr.Blocks(css=custom_css, title="CineMatch AI - Movie Recommender") as demo:
    
    gr.HTML("""
        <div class="header">
            <h1>🎬 CineMatch AI</h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 20px; margin: 10px 0 0 0;">
                Next-Generation Movie Recommendation Engine
            </p>
        </div>
    """)
    
    with gr.Tabs():
        # Tab 1: Personalized Recommendations
        with gr.TabItem("🎯 Personalized Recommendations"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                        <div style="background: linear-gradient(135deg, rgba(233,69,96,0.2), rgba(15,52,96,0.2)); padding: 25px; border-radius: 15px; border: 2px solid rgba(233,69,96,0.3); margin-bottom: 20px;">
                            <h3 style="color: #e94560; margin-top: 0; font-size: 24px;">⚙️ Settings</h3>
                            <p style="color: rgba(255,255,255,0.7); font-size: 14px; margin: 0;">Configure your recommendation preferences</p>
                        </div>
                    """)
                    
                    model_choice = gr.Dropdown(
                        choices=[
                            "SVD (Matrix Factorization)",
                            "SVD++ (Enhanced MF)",
                            "User-based Collaborative Filtering",
                            "Item-based Collaborative Filtering", 
                            "Neural Collaborative Filtering",
                            "Hybrid Model (Recommended)"
                        ],
                        value="SVD (Matrix Factorization)",
                        label="🤖 Recommendation Model",
                        info="Choose your preferred algorithm"
                    )
                    
                    user_id_input = gr.Number(
                        label="👤 User ID", 
                        value=196, 
                        minimum=1, 
                        maximum=943,
                        info="Enter a user ID (1-943)"
                    )
                    
                    num_recs_input = gr.Slider(
                        label="🎬 Number of Recommendations", 
                        minimum=1, 
                        maximum=20, 
                        value=12, 
                        step=1,
                        info="How many movies?"
                    )
                    
                    submit_btn = gr.Button("🎯 Get Recommendations", variant="primary", size="lg")
                    
                    gr.Markdown("""
                        ### 📊 Model Info:

                        **SVD**: Best overall accuracy (RMSE: 0.934)

                        **SVD++**: Enhanced with implicit feedback (RMSE: 0.932)

                        **User-CF**: Finds similar users' preferences

                        **Item-CF**: More stable on sparse data

                        **Neural CF**: Deep learning approach

                        **Hybrid**: Combines all models (Recommended!)
                    """)
                
                with gr.Column(scale=3):
                    output = gr.HTML(label="Your Recommendations")
            
            submit_btn.click(
                fn=recommend_movies_professional,
                inputs=[user_id_input, num_recs_input, model_choice],
                outputs=output
            )
        
        # Tab 2: Content-Based Search
        with gr.TabItem("🔎 Search Similar Movies"):
            gr.HTML("""
                <div style="background: linear-gradient(135deg, rgba(233,69,96,0.2), rgba(15,52,96,0.2)); padding: 25px; border-radius: 15px; border: 2px solid rgba(233,69,96,0.3); margin-bottom: 20px; text-align: center;">
                    <h3 style="color: #e94560; margin-top: 0; font-size: 24px;">🔍 Find Similar Movies</h3>
                    <p style="color: rgba(255,255,255,0.7); font-size: 14px; margin: 0;">
                        Enter any movie title and discover 6 similar movies using content-based filtering
                    </p>
                </div>
            """)
            
            with gr.Row():
                search_input = gr.Textbox(
                    label="🎬 Movie Title", 
                    placeholder="e.g. Toy Story, The Matrix, Titanic...", 
                    lines=1,
                    scale=4
                )
                search_btn = gr.Button("🔍 Find Similar", variant="primary", scale=1, size="lg")
            
            search_output = gr.HTML()
            
            def on_search_click(query):
                if not query or str(query).strip() == "":
                    return create_error_html("Please enter a movie name to search.")
                
                if cb_movies_df is None or cb_cosine_sim is None:
                    return create_error_html("Content-based model is not available. Please try again later.")
                
                best_title, recs = get_top_similar_from_best_match(query, top_n=6)
                
                if best_title is None or not recs:
                    return create_error_html(f"No matches found for '{query}'. Try a different movie title.")
                
                html = render_content_based_html(recs, heading=f"🎯 Because you searched: {best_title}")
                return html
            
            search_btn.click(fn=on_search_click, inputs=search_input, outputs=search_output)
            search_input.submit(fn=on_search_click, inputs=search_input, outputs=search_output)
            
            gr.Markdown("""
                <div style="margin-top: 20px; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 14px;">
                        💡 <strong>How it works:</strong> Content-based filtering analyzes movie features like genres, 
                        keywords, and metadata to find similar movies. Results are ranked by similarity score.
                    </p>
                </div>
            """)

    # Footer
    gr.Markdown("""
        <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, rgba(233,69,96,0.1), rgba(15,52,96,0.1)); border-radius: 15px; border: 1px solid rgba(233,69,96,0.2); margin-top: 30px;">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 16px;">
                🤗 Models loaded from Hugging Face • Created with ❤️ for DataSynthis ML Job Task
            </p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch()