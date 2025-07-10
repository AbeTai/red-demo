# Music Recommender Demo

A comprehensive music recommendation system featuring multiple algorithms and interfaces for exploring collaborative filtering and diversified recommendations.

## Features

- **Collaborative Filtering**: Implicit ALS (Alternating Least Squares) algorithm
- **MMR Reranking**: Maximal Marginal Relevance for balanced relevance-diversity recommendations
- **Demographic Filtering**: Search by gender and age categories
- **Multiple Interfaces**: Basic, enhanced, and MMR-enabled web applications
- **Real-time Parameter Tuning**: Dynamic adjustment of recommendation parameters

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd red-demo

# Install dependencies using uv
uv sync
```

### Generate Sample Data

```bash
# Create synthetic dataset with 1000 users, 20 artists, and demographics
uv run python data_generator.py
```

### Run Applications

Choose from multiple Streamlit interfaces:

```bash
# Basic recommender
uv run streamlit run app.py --server.port 8501

# With demographic filtering
uv run streamlit run app_enhanced.py --server.port 8502

# With MMR reranking
uv run streamlit run app_mmr.py --server.port 8503

# Full-featured (demographics + MMR)
uv run streamlit run app_enhanced_mmr.py --server.port 8504
```

## Applications Overview

### Basic App (`app.py`)
- User ID input or artist-based user search
- Standard collaborative filtering recommendations
- User listening history display

### Enhanced App (`app_enhanced.py`)
- All basic features plus:
- Gender filtering (Male/Female/Other)
- Age category filtering (5-year ranges: 15-19, 20-24, etc.)
- Combined artist + demographic search

### MMR Apps (`app_mmr.py`, `app_enhanced_mmr.py`)
- Side-by-side comparison of standard vs MMR recommendations
- Dynamic lambda parameter adjustment (0.0=diversity, 1.0=relevance)
- Configurable candidate pool size
- Real-time reranking with visual explanations

## Command Line Interface

### Basic Recommender

```bash
uv run python recommender.py --help

# Examples
uv run python recommender.py --user-id 10 --n-recommendations 5
uv run python recommender.py --csv-path data.csv --model-dir models/
uv run python recommender.py --alpha 0.6 --train
```

### MMR Recommender

```bash
uv run python recommender_mmr.py --help

# Examples
# Balanced recommendation
uv run python recommender_mmr.py --user-id 10 --lambda-param 0.5

# Diversity-focused
uv run python recommender_mmr.py --user-id 10 --lambda-param 0.1

# Relevance-focused  
uv run python recommender_mmr.py --user-id 10 --lambda-param 0.9

# Disable MMR (standard collaborative filtering)
uv run python recommender_mmr.py --user-id 10 --no-mmr
```

## Algorithm Details

### Implicit ALS
- Matrix factorization for collaborative filtering
- Confidence-based weighting: `1 + alpha * play_count`
- Configurable alpha parameter (default: 0.4)

### MMR (Maximal Marginal Relevance)
- Balances relevance and diversity in recommendations
- Formula: `» × relevance_score - (1-») × max_similarity`
- Uses cosine similarity between artist embeddings
- Lambda parameter controls trade-off (0=diverse, 1=relevant)

### Data Schema

The system uses synthetic music listening data with the following structure:

```csv
user_id,artist,play_count,gender,age
1,Taylor Swift,31,Female,25-29
1,Drake,42,Female,25-29
2,The Beatles,15,Male,45-49
```

- **user_id**: 1-1000 unique users
- **artist**: 20 popular artists
- **play_count**: 1-500 plays per user-artist pair
- **gender**: Male, Female, Other (distribution: 48%, 48%, 4%)
- **age**: 5-year categories from 15-19 to 65-70

## Model Persistence

Models are automatically saved and loaded based on parameters:
- Standard models: `recommender_model_alpha_{alpha}.pkl`
- MMR models: `recommender_mmr_model_alpha_{alpha}.pkl`
- Custom model directories supported via `--model-dir`

## Performance Features

- **Polars**: High-performance data processing
- **Streamlit Caching**: Model loading optimization
- **Sparse Matrices**: Efficient memory usage for large datasets
- **Configurable Paths**: Flexible file organization

## Use Cases

### Research & Education
- Compare collaborative filtering approaches
- Study relevance vs diversity trade-offs
- Analyze demographic bias in recommendations

### Demo & Prototyping
- Interactive parameter exploration
- Side-by-side algorithm comparison
- Real-time recommendation generation

### Development & Testing
- Command-line model training and evaluation
- Configurable data paths and model storage
- Extensible architecture for new algorithms

## Technical Requirements

- Python e3.13
- Dependencies managed via `uv`
- Web browser for Streamlit interfaces
- ~500MB memory for default dataset

## Contributing

When extending the system:

1. Follow the established patterns for configurable paths
2. Use Polars for data processing efficiency
3. Maintain backward compatibility for model loading
4. Add command-line arguments for new parameters
5. Update both CLI and web interfaces consistently

## License

[Add your license information here]