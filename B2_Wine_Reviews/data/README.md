# Task B2: Predicting Wine Critic Scores from Tasting Notes and Metadata

## Dataset Source
- **Repository:** Kaggle (`zynicide/wine-reviews`) / Hugging Face (`james-burton/wine_reviews`)
- **Dataset File:** `winemag-data-130k-v2.csv`
- **URL:** [https://www.kaggle.com/datasets/zynicide/wine-reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews)

## Dataset Characteristics
- **Total Reviews:** ~130,000 professional sommelier reviews
- **Features:**
  - Free text: `description` (sensory tasting note, ~40 words)
  - Numeric: `price` (bottle price in USD, partially missing)
  - Categorical metadata: `country`, `province`, `variety`, `winery`
- **Target:** `points` (WineEnthusiast scale from 80 to 100)
- **Leakage Prevention:** Exact duplicate reviews across (description, points) are systematically purged *before* dataset splitting to prevent memorization and test leakage.

## Download Instructions
Downloaded directly via the `datasets` library or Kaggle API:
```python
from datasets import load_dataset

wine_dataset = load_dataset('james-burton/wine_reviews')
```
Raw tabular datasets are excluded from Git via `.gitignore`.
