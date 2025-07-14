# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
音楽推薦システム - 協調フィルタリングとMMR（Maximal Marginal Relevance）を統合した包括的な推薦システムです。Implicit ALSアルゴリズムを基にした推薦エンジンに、関連性と多様性のバランスを調整するMMRアルゴリズムと人口統計学フィルタリングを組み合わせています。

## Core Architecture

### Data Flow
```
CSV Data → Polars DataFrame → ALS Training → Model (cached in weights/) → MMR Re-ranking → Streamlit UI
```

### Key Components

#### 1. **Recommendation Models** (`models/matrix_factorization/`)
- **MusicRecommender**: Base collaborative filtering using Implicit ALS
- **MusicRecommenderMMR**: Extended with MMR (Maximal Marginal Relevance) for diversity
- **BaseRecommender**: Abstract base class with common functionality

#### 2. **Evaluation System** (`src/evaluation/`)
- **ModelEvaluator**: Main orchestrator for model evaluation
- **EvaluationMetrics**: Precision@K, Recall@K, NDCG@K, Hit Rate, Coverage metrics
- **TimeSeriesDataSplitter**: Splits data chronologically for realistic evaluation
- **ResultsManager**: Handles CSV-based result persistence with duplicate detection

#### 3. **Streamlit Application** (`app.py`)
- Interactive web interface with user search (ID or artist-based)
- Real-time MMR parameter adjustment (λ: 0=diversity, 1=relevance)
- Demographic filtering (gender/age) and side-by-side recommendation comparison
- **Critical**: Uses callback functions (`on_change`) instead of forms for cross-platform stability

### Data Schema
CSV files must follow this structure:
```csv
user_id,artist,play_count,gender,age,interaction_date,genre
1,Taylor Swift,41,Male,25-29,20210427,Pop
```

## Common Development Commands

### Environment Setup
```bash
# Install dependencies (uv recommended)
uv add polars implicit scikit-learn streamlit

# Generate sample data
python data_generator.py
```

### Running Applications
```bash
# Start Streamlit web interface
uv run streamlit run app.py

# Evaluate models with custom parameters
python evaluate_models.py --csv-path data/user_artist_plays.csv --k 5

# Show evaluation results
python evaluate_models.py --show-summary
python evaluate_models.py --compare-models
python evaluate_models.py --show-best

# Run individual models
python models/matrix_factorization/music_recommender_mmr.py --csv-path data/user_artist_plays.csv --user-id 1 --lambda-param 0.5
```

### Testing & Quality Assurance
```bash
# Type checking
mypy models/ src/ *.py

# Model caching check (weights should be created)
ls weights/

# Evaluation results verification
head results/evaluation_results.csv
```

## Critical Implementation Notes

### Streamlit Cross-Platform Compatibility
- **Never use `st.form()`** - causes state reset issues on Windows
- **Always use explicit widget keys** and `on_change` callbacks for ALL widgets
- **Mandatory callback pattern**: Apply to selectbox, multiselect, and all interactive widgets
- **Separate debug UI** into sidebar to avoid widget interference
- **Session state management**: Initialize with defensive type checking
- **Windows-first design**: Windows environment has stricter widget state management than Mac

### Model Training & Caching
- Models auto-train on first run and cache to `weights/` directory
- Cache key format: `{dataset_name}_mmr_alpha_{alpha_value}.pkl`
- Use `@st.cache_resource` for Streamlit integration
- Large datasets may require memory optimization

### Evaluation System
- Results auto-saved to `results/evaluation_results.csv` 
- Duplicate configuration detection prevents redundant evaluations
- JSON parameter serialization for reproducibility
- Time-series splitting maintains chronological order for realistic evaluation

### MMR Algorithm
- Lambda parameter: 0.0 (max diversity) to 1.0 (max relevance)
- Uses cosine similarity on ALS-generated artist embeddings
- Candidate pool size affects diversity vs computation trade-off
- Optimal λ=0.7 found through extensive evaluation

## Performance Optimization

### Large Dataset Handling
- Polars DataFrame for memory-efficient data processing
- Sparse matrix operations for user-item interactions
- Chunked evaluation for memory constraints
- Artist similarity matrix caching

### Model Configuration
Current optimal settings from evaluation:
```json
{
  "alpha": 0.4,
  "factors": 50, 
  "iterations": 20,
  "regularization": 0.1,
  "lambda_param": 0.7
}
```

## Streamlit Widget State Management Patterns

### Required Callback Pattern for Cross-Platform Compatibility

```python
# Session state initialization
if 'selected_value' not in st.session_state:
    st.session_state.selected_value = 'default'

# Callback function
def on_value_change():
    if 'widget_key' in st.session_state:
        st.session_state.selected_value = st.session_state.widget_key

# Widget with callback
options = ['option1', 'option2', 'option3']
index = options.index(st.session_state.selected_value) if st.session_state.selected_value in options else 0

selected = st.selectbox(
    "Label:",
    options,
    index=index,
    key="widget_key",
    on_change=on_value_change
)
```

### Why This Pattern is Essential

**Problem**: On Windows, Streamlit widgets reset to default values during page reruns triggered by other widget interactions.

**Root Cause**: 
- Windows has stricter widget state management than Mac
- Page reruns completely reinitialize widgets without `on_change` callbacks
- `default` parameter alone is insufficient for state persistence

**Solution**: Immediate synchronization via callbacks ensures state persistence across all environments.

### Widget Types Requiring This Pattern
- `st.selectbox` - dropdown selections
- `st.multiselect` - multiple selections  
- `st.radio` - radio button groups
- `st.slider` - numeric sliders
- `st.text_input` - text fields (when state persistence needed)

### Critical: Demographics Filter Issue Resolution
The same pattern that fixed artist selection was applied to gender/age filters:

**Before (problematic)**:
```python
selected_gender = st.selectbox("Gender:", options, key="gender_filter")
```

**After (Windows-compatible)**:
```python
def on_gender_change():
    if 'gender_selectbox' in st.session_state:
        st.session_state.selected_gender = st.session_state.gender_selectbox

index = options.index(st.session_state.selected_gender) if st.session_state.selected_gender in options else 0
selected_gender = st.selectbox("Gender:", options, index=index, key="gender_selectbox", on_change=on_gender_change)
```