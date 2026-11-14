# Task B3: Predicting Star Ratings of E-Commerce Clothing Reviews

## Dataset Source
- **Repository:** Kaggle (`nicapotato/womens-ecommerce-clothing-reviews`) / Hugging Face (`saattrupdan/womens-clothing-ecommerce-reviews`)
- **Title:** Women's E-Commerce Clothing Reviews
- **URL:** [https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

## Dataset Characteristics
- **Total Reviews:** 23,486 reviews across women's apparel
- **Features:**
  - Free text: `review_text` (customer commentary)
  - Numeric metadata: `age`, `positive_feedback_count`
  - Categorical departments: `division_name`, `department_name`, `class_name`
- **Target:** `rating` (Ordinal 1–5 stars, heavily skewed towards 5 stars)

## Data Leakage Rule
**CRITICAL:** The feature `Recommended IND` is strictly excluded from all model inputs. In e-commerce platforms, the recommendation boolean flag is submitted simultaneously with the star rating, making it unavailable during forward inference. Including it introduces massive label leakage.

## Download Instructions
Loaded directly via Hugging Face `datasets` or Kaggle:
```python
from datasets import load_dataset

dataset = load_dataset('saattrupdan/womens-clothing-ecommerce-reviews')
```
Raw tabular datasets are excluded from Git via `.gitignore`.
