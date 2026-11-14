# Task C2: Abstractive Summarisation of BBC News Articles (XSum)

## Dataset Source
- **Repository:** Hugging Face Datasets
- **Dataset:** `EdinburghNLP/xsum`
- **URL:** [https://huggingface.co/datasets/EdinburghNLP/xsum](https://huggingface.co/datasets/EdinburghNLP/xsum)
- **Paper Reference:** Narayan, S., Cohen, S. B., & Lapata, M. (2018). *Don't Give Me the Details, Just the Summary! Topic-Conditioned Abstractive Summarization*. EMNLP 2018.

## Dataset Characteristics
- **Total Corpus:** 226,711 BBC news articles paired with professionally written, one-sentence abstractive summaries.
  - Official Train: 204,045 articles
  - Official Validation: 11,332 articles
  - Official Test: 11,334 articles
- **Sampled Subset (Per Course Assignment Specification):**
  - Sampled Train: 20,000 articles
  - Sampled Validation: 1,000 articles
  - Sampled Test: 1,000 articles
  - Fixed Seed: 42
- **Key Properties:**
  - Extreme summarisation setting: highly abstractive one-sentence summaries written by BBC journalists.
  - High proportion of novel unigrams and bigrams (summary n-grams not found in the original document text).
  - Minimal lead bias compared to standard broadcast or wire news datasets (e.g., CNN/DailyMail).

## Download & Reproducibility Instructions
To load and partition the dataset programmatically:
```python
from datasets import load_dataset

# Load official XSum splits
xsum = load_dataset("EdinburghNLP/xsum")

# Draw deterministic subsets with fixed seed 42
train_sample = xsum["train"].shuffle(seed=42).select(range(20000))
val_sample = xsum["validation"].shuffle(seed=42).select(range(1000))
test_sample = xsum["test"].shuffle(seed=42).select(range(1000))
```

All raw arrow and parquet cache files reside in `~/.cache/huggingface/` and are excluded from Git version control per course size policies (< 50 MB).
