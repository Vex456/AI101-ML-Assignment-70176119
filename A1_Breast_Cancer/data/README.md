# Task A1: Breast Cancer Diagnosis from Cell-Nucleus Measurements

## Dataset Source
- **Repository:** UCI Machine Learning Repository (ID: 17) & scikit-learn built-in dataset
- **Official Title:** Breast Cancer Wisconsin (Diagnostic)
- **URL:** [https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Dataset Characteristics
- **Total Samples:** 569 instances
- **Features:** 30 real-valued continuous nuclear morphometric descriptors computed from digitized fine-needle aspirate (FNA) images of breast masses.
- **Classes:** 
  - Malignant: 212 samples (37.26%) — mapped to positive class (1)
  - Benign: 357 samples (62.74%) — mapped to negative class (0)
- **Feature Groups:** For each of the 10 fundamental nuclear characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), the mean, standard error (SE), and "worst" (largest) value are recorded.

## Access & Loading
The dataset is loaded directly via `scikit-learn`:
```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
X = data.frame.drop(columns=['target'])
# Target mapping: 1 for malignant, 0 for benign
y = (data.frame['target'] == 0).astype(int)
```
No external data download is required.
