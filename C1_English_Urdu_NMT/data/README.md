# Task C1: English-to-Urdu Neural Machine Translation (OPUS-100)

## Dataset Source
- **Repository:** Hugging Face Datasets
- **Dataset:** `Helsinki-NLP/opus-100`
- **Config:** `en-ur`
- **URL:** [https://huggingface.co/datasets/Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- **Pretrained Benchmark Model:** `Helsinki-NLP/opus-mt-en-ur` (MarianMT)

## Dataset Characteristics
- **Total Corpus:** $\approx 750,000$ English-Urdu sentence pairs
- **Validation Split:** 2,000 sentence pairs
- **Test Split:** 2,000 sentence pairs
- **Cleaning Filters Applied:**
  1. Deduplication and removal of empty/whitespace lines.
  2. Removal of pairs with length ratio $> 3$.
  3. Removal of sentences with $> 30$ tokens.
  4. Removal of targets lacking valid Urdu script (`\u0600` to `\u06FF`).
- **Cleaned Working Subset:** 50,000 to 100,000 training pairs.

## Download Instructions
Downloaded directly using Hugging Face `datasets`:
```python
from datasets import load_dataset

opus_en_ur = load_dataset("Helsinki-NLP/opus-100", "en-ur")
train_data = opus_en_ur["train"]
val_data = opus_en_ur["validation"]
test_data = opus_en_ur["test"]
```
All raw parquet and arrow cache files are excluded from Git via `.gitignore`.
