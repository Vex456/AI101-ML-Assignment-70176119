# Task A3: Dry Bean Variety Recognition from Shape Features

## Dataset Source
- **Repository:** UCI Machine Learning Repository (ID: 602)
- **Title:** Dry Bean Dataset
- **URL:** [https://archive.ics.uci.edu/dataset/602/dry+bean+dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Reference:** M. Koklu and I. A. Ozkan, "Multiclass classification of dry beans using computer vision and machine learning techniques," *Computers and Electronics in Agriculture*, vol. 174, p. 105507, 2020.

## Dataset Characteristics
- **Total Samples:** 13,611 grains
- **Features:** 16 continuous numeric shape descriptors:
  - 12 dimensional features: Area, Perimeter, Major Axis Length, Minor Axis Length, Aspect Ratio, Eccentricity, Convex Area, Equiv Diameter, Extent, Solidity, Roundness, Compactness.
  - 4 shape factors: ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.
- **Classes:** 7 registered dry bean varieties with natural class imbalance:
  - Dermason: 3,546 (26.05%)
  - Sira: 2,636 (19.37%)
  - Seker: 2,027 (14.89%)
  - Horoz: 1,928 (14.16%)
  - Cali: 1,630 (11.98%)
  - Barbunya: 1,322 (9.71%)
  - Bombay: 522 (3.84%)

## Download Instructions
Downloaded directly via the official `ucimlrepo` library:
```python
from ucimlrepo import fetch_ucirepo

dry_bean = fetch_ucirepo(id=602)
X = dry_bean.data.features
y = dry_bean.data.targets
```
Raw tabular datasets are excluded from Git via `.gitignore`.
