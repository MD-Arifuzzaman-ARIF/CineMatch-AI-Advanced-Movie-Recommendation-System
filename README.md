# 🎬 CineMatch AI - Advanced Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive-yellow.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art movie recommendation system that combines **6 different algorithms** to provide personalized movie suggestions. Built with modern ML techniques and deployed as an interactive web application.

![CineMatch AI Demo](https://via.placeholder.com/800x400/1a1a2e/e94560?text=CineMatch+AI+Demo)

## 🌟 Features

### 🤖 Multiple Recommendation Algorithms
- **SVD (Singular Value Decomposition)** - Matrix factorization for collaborative filtering
- **SVD++** - Enhanced SVD with implicit feedback
- **User-Based Collaborative Filtering** - Find similar users
- **Item-Based Collaborative Filtering** - Find similar movies
- **Neural Collaborative Filtering (NCF)** - Deep learning approach
- **Hybrid Model** - Combines all algorithms with weighted averaging

### 🔍 Content-Based Filtering
- **Fuzzy Movie Search** - Search by movie name with typo tolerance
- **TF-IDF Vectorization** - Analyzes movie features (genres, keywords, metadata)
- **Cosine Similarity** - Finds movies with similar content
- **Character N-Gram Matching** - Advanced title matching

### 🎨 Interactive Web Interface
- **Gradio-Powered UI** - Beautiful, responsive design
- **TMDB API Integration** - Real movie posters and metadata
- **Real-Time Recommendations** - Instant results
- **Model Comparison** - Test different algorithms side-by-side

## 📊 Performance Metrics

| Model | RMSE | Precision@10 | Recall@10 | NDCG@10 |
|-------|------|--------------|-----------|---------|
| SVD | 0.935 | 59.38% | 73.39% | 84.32% |
| SVD++ | **0.931** | 59.48% | 73.53% | **84.48%** |
| User-CF | 1.019 | 59.48% | **73.47%** | 84.60% |
| Item-CF | 1.026 | 56.90% | 71.69% | 79.45% |
| Neural CF | 0.944 | 57.95% | 72.51% | 82.49% |

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cinematch-ai.git
cd cinematch-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up TMDB API** (Optional - for movie posters)
```bash
# Get your free API key from https://www.themoviedb.org/
# Add to app.py:
TMDB_API_KEY = "your_api_key_here"
```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
```
http://localhost:7860
```

## 📁 Project Structure
```
cinematch-ai/
│
├── app.py                          # Main Gradio application
├── movie_recommendation_system.ipynb  # Training notebook
├── requirements.txt                # Python dependencies
│
├── saved_models/                   # Trained models
│   ├── svd_model.pkl
│   ├── svdpp_model.pkl
│   ├── user_based_cf.pkl
│   ├── item_based_cf.pkl
│   ├── ncf_model/                  # Neural network model
│   ├── content_based_model.pkl
│   └── mappings.pkl
│
└── README.md
```

## 🎯 Usage Examples

### 1. Personalized Recommendations
```python
# Get recommendations for User 196 using Hybrid model
recommendations = recommend_movies(user_id=196, N=10, model='hybrid')
```

### 2. Content-Based Search
```python
# Find movies similar to "The Matrix"
similar_movies = get_similar_movies("The Matrix", top_n=6)
```

### 3. Model Comparison
```python
# Compare different models for the same user
for model in ['svd', 'svdpp', 'user_cf', 'item_cf', 'ncf', 'hybrid']:
    recs = recommend_movies(user_id=196, N=5, model=model)
```

## 🧠 How It Works

### Collaborative Filtering
1. **Matrix Factorization** - Decomposes user-item matrix into latent factors
2. **Neighborhood Methods** - Finds similar users/items based on ratings
3. **Neural Networks** - Learns complex non-linear patterns

### Content-Based Filtering
1. **Feature Extraction** - Extracts genres, keywords, metadata
2. **TF-IDF Vectorization** - Converts features to numerical vectors
3. **Similarity Computation** - Calculates cosine similarity between movies

### Hybrid Approach
```
Final Score = 0.30 × SVD++ + 0.30 × Item-CF + 0.20 × User-CF + 0.20 × NCF
```

## 📚 Dataset

- **MovieLens 100K** - 100,000 ratings from 943 users on 1,682 movies
- **TMDB API** - Movie posters and additional metadata
- **Rating Scale** - 1 to 5 stars

## 🔧 Technologies Used

### Machine Learning
- **scikit-surprise** - Collaborative filtering algorithms
- **TensorFlow/Keras** - Neural collaborative filtering
- **scikit-learn** - TF-IDF, cosine similarity

### Web Framework
- **Gradio** - Interactive UI
- **Requests** - TMDB API integration

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations

## 📈 Model Training

To retrain models from scratch:
```bash
# Open the Jupyter notebook
jupyter notebook movie_recommendation_system.ipynb

# Or run the training script
python train_models.py
```

Training time on CPU:
- Traditional ML models: ~2-3 minutes
- Neural CF: ~5-10 minutes
- Content-based: ~1 minute

## 🎨 Web Interface Features

### Tab 1: Personalized Recommendations
- Select recommendation model
- Enter user ID (1-943)
- Choose number of recommendations
- View results with movie posters

### Tab 2: Search Similar Movies
- Search by movie name
- Fuzzy matching (typo-tolerant)
- Get 6 similar movies
- Display similarity scores

## 🐛 Troubleshooting

### Common Issues

**1. Module Not Found Error**
```bash
pip install --upgrade -r requirements.txt
```

**2. TMDB API Rate Limit**
- Free tier: 40 requests per 10 seconds
- Add delays between requests if needed

**3. Model Loading Error**
- Ensure all `.pkl` files are in `saved_models/` directory
- Re-run training notebook if models are corrupted

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MovieLens** - For the 100K dataset
- **TMDB** - For movie metadata and posters
- **scikit-surprise** - For collaborative filtering algorithms
- **Gradio** - For the amazing web interface framework

## 📧 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/cinematch-ai](https://github.com/yourusername/cinematch-ai)

Live Demo: [https://huggingface.co/spaces/yourusername/cinematch-ai](https://huggingface.co/spaces/yourusername/cinematch-ai)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
```

---

## Additional Files to Include

### `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Jupyter
.ipynb_checkpoints

# Models (if too large)
saved_models/*.pkl
saved_models/ncf_model/

# API Keys
.env
config.py

# OS
.DS_Store
Thumbs.db
```

### `requirements.txt`
```
gradio>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-surprise>=1.1.3
tensorflow>=2.13.0
scikit-learn>=1.3.0
requests>=2.31.0
rapidfuzz>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### `LICENSE` (MIT License)
```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full MIT License text...]
