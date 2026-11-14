# Task A2: Concrete Compressive Strength Dataset

## Dataset Source
- **Repository:** UCI Machine Learning Repository
- **Dataset ID:** 165
- **URL:** [https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- **Donor:** Prof. I-Cheng Yeh, Department of Civil Engineering, Chung-Hua University, Hsin Chu, Taiwan.

## Dataset Characteristics
- **Instances:** 1,030
- **Input Features:** 8 numerical attributes (raw mix proportions in $\text{kg/m}^3$ and curing age in days)
- **Target:** Concrete compressive strength (MPa)

## Features Description
1. `Cement` ($\text{kg/m}^3$)
2. `Blast Furnace Slag` ($\text{kg/m}^3$)
3. `Fly Ash` ($\text{kg/m}^3$)
4. `Water` ($\text{kg/m}^3$)
5. `Superplasticizer` ($\text{kg/m}^3$)
6. `Coarse Aggregate` ($\text{kg/m}^3$)
7. `Fine Aggregate` ($\text{kg/m}^3$)
8. `Age` (days, 1 to 365)
9. Target: `Concrete compressive strength` (MPa)

## Download Instructions
The dataset is retrieved programmatically via `ucimlrepo` or directly from the UCI archive:
```python
from ucimlrepo import fetch_ucirepo

concrete = fetch_ucirepo(id=165)
X = concrete.data.features
y = concrete.data.targets
```
Raw dataset files should **not** be committed to GitHub as per course rules.
