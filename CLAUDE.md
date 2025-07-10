# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a music recommendation system demo with multiple variants featuring different algorithms and UI capabilities. The system uses Implicit ALS (Alternating Least Squares) for collaborative filtering and includes MMR (Maximal Marginal Relevance) for diversified recommendations.

## Key Commands

### Environment Setup
```bash
# Install dependencies
uv add <package-name>

# Run applications
uv run streamlit run app.py --server.port 8501                    # Main integrated app (recommended)
uv run streamlit run app_enhanced.py --server.port 8502          # Legacy: demographics filtering  
uv run streamlit run app_mmr.py --server.port 8503               # Legacy: MMR reranking
uv run streamlit run app_enhanced_mmr.py --server.port 8504      # Legacy: full-featured app
```

### Data Generation
```bash
# Generate new synthetic dataset
uv run python data_generator.py
```

### Model Training and Testing
```bash
# Train/test basic recommender
uv run python models/recommender.py [--csv-path PATH] [--model-dir weights/] [--alpha 0.4]

# Train/test MMR recommender with various parameters
uv run python models/recommender_mmr.py [OPTIONS]
  --csv-path PATH              # Data file path
  --model-dir weights/         # Model storage directory (default: weights/)
  --alpha FLOAT                # ALS confidence parameter
  --user-id INT                # Test user ID
  --n-recommendations INT      # Number of recommendations
  --lambda-param FLOAT         # MMR diversity parameter (0=diverse, 1=relevant)
  --candidate-pool-size INT    # MMR candidate pool size
  --no-mmr                     # Disable MMR reranking
  --train                      # Force model retraining
```

## Architecture

### Core Components

**Data Layer:**
- `user_artist_plays.csv`: Main dataset with columns [user_id, artist, play_count, gender, age]
- `data_generator.py`: Synthetic data generation with demographics (gender: Male/Female/Other, age: 5-year categories)

**Model Layer:**
- `models/recommender.py`: Base MusicRecommender class using Implicit ALS
- `models/recommender_mmr.py`: Extended MusicRecommenderMMR with MMR reranking capabilities
- Model persistence in `weights/` directory with CSV-name-based file naming

**Frontend Layer:**
- `app.py`: **Main integrated application** with all features (MMR + demographics)
- `app_enhanced.py`: Legacy - demographic filtering only
- `app_mmr.py`: Legacy - MMR parameter controls only  
- `app_enhanced_mmr.py`: Legacy - full-featured version

### Data Processing Pipeline

1. **Data Loading**: Polars DataFrames for efficient processing
2. **Sparse Matrix Creation**: CSR format for ALS algorithm
3. **Model Training**: Implicit ALS with configurable alpha parameter
4. **Recommendation Generation**: 
   - Standard: Relevance-based ranking
   - MMR: Balanced relevance-diversity using cosine similarity

### Key Design Patterns

**Model Versioning**: Models saved with CSV basename and alpha parameter (`{csv_basename}_alpha_{alpha:.1f}.pkl`, `{csv_basename}_mmr_alpha_{alpha:.1f}.pkl`)

**Configurable Paths**: All file paths (CSV, model directory) configurable via command line arguments

**Organized Structure**: Models in `models/` directory, weights in `weights/` directory

**MMR Algorithm**: 
- Calculates artist similarity using learned embeddings
- Iteratively selects items maximizing: `λ × relevance - (1-λ) × max_similarity`
- No forced relevance-first selection - all positions determined by MMR score

**Frontend Architecture**:
- Streamlit caching for model loading performance
- Side-by-side comparison views for standard vs MMR recommendations
- Dynamic parameter adjustment with real-time results
- **app.py as main version**: All features integrated, use for new development
- Legacy apps preserved for reference

### Data Schema

```
user_artist_plays.csv:
- user_id: int (1-1000)
- artist: str (20 unique artists)  
- play_count: int (1-500)
- gender: str (Male/Female/Other)
- age: str (5-year categories: "15-19", "20-24", etc.)
```

### Package Dependencies

Uses `uv` for dependency management. Key packages:
- `streamlit`: Web interface
- `polars`: High-performance data processing  
- `implicit`: ALS recommendation algorithm
- `scikit-learn`: MMR similarity calculations
- `numpy`, `scipy`: Numerical computing

## Important Notes

- Always use `uv run` prefix for Python commands in this environment
- **Use app.py as the main application** - it contains all integrated features
- Models are automatically cached and reused unless `--train` flag is specified
- MMR requires higher candidate pool size than final recommendation count
- Demographic filtering works only with generated data containing gender/age columns
- Model files are automatically named based on CSV filename for multi-dataset support
- Port conflicts: Use different ports (8501-8504) for running multiple Streamlit apps simultaneously
- **Development workflow**: Create `app_test.py` for experiments, then integrate into `app.py`