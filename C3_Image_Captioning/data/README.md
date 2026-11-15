# Task C3: Image Captioning with a CNN Encoder and an LSTM Decoder (Flickr8k)

## Dataset Source
- **Repository:** Hugging Face Datasets
- **Dataset:** `jxie/flickr8k` (Mirror of Flickr8k benchmark, Hodosh et al., 2013)
- **URL:** [https://huggingface.co/datasets/jxie/flickr8k](https://huggingface.co/datasets/jxie/flickr8k)
- **Paper Reference:** Hodosh, M., Young, P., & Hockenmaier, J. (2013). *Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics*. Journal of Artificial Intelligence Research (JAIR), 47, 853–899.

## Dataset Characteristics
- **Total Corpus:** 8,091 color photographs depicting diverse everyday human activities, animals, and outdoor scenes.
- **Captions:** Exactly 5 reference captions per image (totaling 40,455 captions), crowdsourced via Amazon Mechanical Turk.
- **Split Strategy (Enforcing Zero Cross-Set Leakage):**
  - Partitioning is strictly performed at the **image level** so that all 5 reference captions for any image remain exclusively within one split.
  - **Train Split:** 6,000 images (30,000 captions)
  - **Validation Split:** 1,000 images (5,000 captions)
  - **Test Split:** 1,091 images (5,455 captions)

## Preprocessing & Tokenization Pipeline
1. **Case Normalization:** All text converted to lowercase.
2. **Punctuation Stripping:** Punctuation characters stripped via regex `r'[^a-z0-9\s]'`.
3. **Special Delimiters:** `<start>` (SOS) and `<end>` (EOS) tokens attached to each caption prefix.
4. **Vocabulary Thresholding:** Words occurring fewer than 5 times in the training split are replaced with `<unk>` (lowering vocabulary from ~8,500 down to ~2,800 tokens, eliminating sparse long-tail tokens).
5. **Feature Extraction:** Images are processed through a frozen, ImageNet-pretrained CNN (ResNet-50) with global average pooling, caching 2048-dimensional feature representations.

## Download & Reproducibility Instructions
To load and inspect the dataset programmatically:
```python
from datasets import load_dataset

flickr8k = load_dataset("jxie/flickr8k")
print(flickr8k)

# Split into official partitions
train_data = flickr8k["train"].select(range(6000))
val_data = flickr8k["validation"].select(range(1000))
test_data = flickr8k["test"].select(range(1091))
```

All raw image archives and cached feature arrays are excluded from Git tracking via `.gitignore` per course size policies (< 50 MB).
