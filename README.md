# AI101 Machine Learning Assignment: Training Models on Three Data Modalities

**The University of Lahore**  
**Department of Intelligent Systems**  
**Course:** AI-101 Introduction to Artificial Intelligence (Fall 2026)  
**Instructor:** Dr. Muhammad Awais (AI Architect)

---

## Student Information

| Field | Details |
| :--- | :--- |
| **Student Name** | Muhammad Zaryab Attiq |
| **Roll Number / SAP ID** | 70176119 |
| **Section** | Section A |
| **GitHub Repository** | `https://github.com/Vex456/AI101-ML-Assignment-70176119` |

---

## Complete Submission Portfolio: All 9 Tasks Across 3 Pools

Per course guidelines (Section 11 & Page 19 of the assignment specification), to qualify for the **maximum grade (30 marks) plus the full 5 bonus marks**, all 9 tasks across all three pools have been fully implemented, rigorously evaluated, and documented with individual 2-page double-column IEEE conference reports and executed Jupyter notebooks.

### 1. Graded Tasks Portfolio (Primary Deliverables)

| Pool | Task ID | Task Description | Champion Model & Architecture | Primary Metric Score | Notebook Deliverable | Report PDF (IEEE Double-Column) |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: |
| **Pool A (Numerical)** | **A2** | Predicting Concrete Compressive Strength | Gradient Boosted Decision Trees (GBDT) + Log/Hydration Features | **$R^2 = 0.919$**, RMSE = 4.55 MPa, MAE = 2.92 MPa | [`A2_notebook.ipynb`](./A2_Concrete_Strength/A2_notebook.ipynb) | [`70176119_A2_Report.pdf`](./A2_Concrete_Strength/report/70176119_A2_Report.pdf) |
| **Pool B (Mixed)** | **B1** | Detecting Fraudulent Job Postings | Early Multimodal Fusion (TF-IDF + Metadata) + GBDT | **Fraud $F_1 = 0.835$**, PR-AUC = 0.907, ROC-AUC = 0.992 | [`B1_notebook.ipynb`](./B1_Fake_Job_Postings/B1_notebook.ipynb) | [`70176119_B1_Report.pdf`](./B1_Fake_Job_Postings/report/70176119_B1_Report.pdf) |
| **Pool C (Seq2Seq)** | **C1** | English-to-Urdu Neural Machine Translation | Pretrained MarianMT (74M Params) / BiGRU + Bahdanau (Beam $k=4$) | MarianMT: **22.76 BLEU** (chrF = 44.23); BiGRU: **3.18 BLEU** | [`C1_notebook.ipynb`](./C1_English_Urdu_NMT/C1_notebook.ipynb) | [`70176119_C1_Report.pdf`](./C1_English_Urdu_NMT/report/70176119_C1_Report.pdf) |

### 2. Bonus Tasks Portfolio (+5 Bonus Marks)

| Pool | Task ID | Task Description | Champion Model & Architecture | Primary Metric Score | Notebook Deliverable | Report PDF (IEEE Double-Column) |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: |
| **Pool A (Numerical)** | **A1** | Breast Cancer Diagnosis Classification | L2-Regularized Logistic Regression + RobustScaler | **$F_1 = 0.976$**, ROC-AUC = 0.994, Recall = 97.6% | [`A1_notebook.ipynb`](./A1_Breast_Cancer/A1_notebook.ipynb) | [`70176119_A1_Report.pdf`](./A1_Breast_Cancer/report/70176119_A1_Report.pdf) |
| **Pool A (Numerical)** | **A3** | Dry Bean Multi-Class Classification | Radial Basis Function (RBF) Support Vector Machine | **Macro $F_1 = 0.938$**, Accuracy = 93.6% (7 Classes) | [`A3_notebook.ipynb`](./A3_Dry_Bean/A3_notebook.ipynb) | [`70176119_A3_Report.pdf`](./A3_Dry_Bean/report/70176119_A3_Report.pdf) |
| **Pool B (Mixed)** | **B2** | Predicting Wine Critic Review Scores | Ridge Regression + Word/Char TF-IDF + Origin Target Encoding | **$R^2 = 0.448$**, RMSE = 2.31 pts; Variety Top-1 Acc = 74.2% | [`B2_notebook.ipynb`](./B2_Wine_Reviews/B2_notebook.ipynb) | [`70176119_B2_Report.pdf`](./B2_Wine_Reviews/report/70176119_B2_Report.pdf) |
| **Pool B (Mixed)** | **B3** | Clothing Review Star Rating Classification | Cost-Sensitive Softmax GBDT + Sentiment/Review-Length Features | **Macro $F_1 = 0.658$**, Quadratic Weighted Kappa = 0.812 | [`B3_notebook.ipynb`](./B3_Clothing_Reviews/B3_notebook.ipynb) | [`70176119_B3_Report.pdf`](./B3_Clothing_Reviews/report/70176119_B3_Report.pdf) |
| **Pool C (Seq2Seq)** | **C2** | Abstractive BBC News Summarisation (XSum) | Fine-Tuned T5-Small Transformer + Beam Search ($k=4$, Trigram Block) | **ROUGE-1 = 25.07**, ROUGE-2 = 6.92, ROUGE-L = 20.87 | [`C2_notebook.ipynb`](./C2_News_Summarisation/C2_notebook.ipynb) | [`70176119_C2_Report.pdf`](./C2_News_Summarisation/report/70176119_C2_Report.pdf) |
| **Pool C (Seq2Seq)** | **C3** | Image Captioning with CNN and LSTM | ResNet-50 Penultimate Features + Canonical Merge LSTM Architecture | **BLEU-1 = 17.19%**, BLEU-2 = 10.19%, BLEU-4 = 1.74% | [`C3_notebook.ipynb`](./C3_Image_Captioning/C3_notebook.ipynb) | [`70176119_C3_Report.pdf`](./C3_Image_Captioning/report/70176119_C3_Report.pdf) |

---

## Key Experimental Findings & Methodological Breakthroughs

### Pool A: Numerical & Tabular Data
- **Task A1 (Breast Cancer Diagnosis):** Strict outer 5-fold cross-validation demonstrated that scaling must be fitted exclusively on training folds to prevent high-dimensional variance leakage. Logistic Regression with calibrated thresholding achieved zero false negatives (100% sensitivity) on held-out folds, critical for medical screening.
- **Task A2 (Concrete Strength Regression):** Standard linear regression on raw mixture percentages reached only $R^2 = 0.628$. Formulating domain-informed features based on Abram's Law ($w/c$ ratio) and logarithmic curing kinetics ($\log(\text{Age})$) boosted linear performance to $R^2 = 0.833$ (-33% RMSE). GBDT captured complex nonlinear hydration thresholds, achieving **$R^2 = 0.919$**.
- **Task A3 (Dry Bean Variety):** Morphological aspect ratio and roundness features cleanly separated elongated varieties (Horoz, Barbunya), while compact circular varieties (Dermason, Seker) overlapped heavily. Kernelized RBF SVMs with standardized feature inputs resolved nonlinear boundaries, achieving **93.8% Macro $F_1$**.

### Pool B: Text + Metadata (Mixed & Multimodal Data)
- **Task B1 (Fraudulent Job Postings):** Metadata alone yielded low precision (0.232) because early-stage legitimate startups frequently omit company logos. Text TF-IDF alone achieved high precision (0.860) but missed structural fraud patterns. Multimodal early concatenation preserved high precision while expanding recall, achieving **PR-AUC = 0.907** and **Fraud $F_1 = 0.835$**.
- **Task B2 (Wine Critic Review Scores):** Regression on sensory adjectives demonstrated that word n-grams capture qualitative notes ('crisp', 'velvety'), while character n-grams capture morphological wine varietals. Ridge regression achieved **$R^2 = 0.448$**, outperforming raw metadata baselines.
- **Task B3 (Clothing Review Ratings):** Star ratings exhibit an ordinal spectrum with severe positive skew (56% 5-star ratings). Standard multi-class cross-entropy ignores ordinal distance, penalizing a 5-star vs 4-star error equally with a 5-star vs 1-star error. Cost-sensitive weighting and sentiment polarity features achieved a **Quadratic Weighted Kappa of 0.812**.

### Pool C: Sequence-to-Sequence Modeling
- **Task C1 (English-to-Urdu Machine Translation):** On the low-resource parallel OPUS-100 corpus, autoregressive greedy decoding suffered from syntactic myopia. Beam Search ($k=4$) yielded a **+52.2% relative BLEU gain** (3.18 vs. 2.09), successfully reordering English SVO verbs into Urdu SOV sentence-final positions via Bahdanau additive attention.
- **Task C2 (Abstractive News Summarisation on XSum):** The Lead-1 heuristic fails on BBC XSum (15.87 ROUGE-1) due to extreme abstractiveness (83.16% novel bigrams in professional headlines). BiGRU models trained from scratch collapsed into phrase-level repetition (100% repetition rate). Fine-tuned T5-Small with beam search ($k=4$) and trigram blocking achieved **25.07 ROUGE-1** and **20.87 ROUGE-L**, while maintaining a low 6% repetition rate.
- **Task C3 (Image Captioning on Flickr8k):** Freezing ResNet-50 visual features offline accelerated training by $>25\times$. Evaluating greedy search vs beam search revealed that while beam search is standard in translation, in captioning on small datasets it aggressively prunes descriptive paths, collapsing onto safe templates ('a man', 'a dog') with only 10 to 18 distinct words utilized out of a 2,541-word vocabulary (0.7% vocabulary utilization).

---

## Setup and Reproducibility

### 1. Environment Installation
```bash
# Clone repository
git clone https://github.com/Vex456/AI101-ML-Assignment-70176119.git
cd AI101-ML-Assignment-70176119

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or .\.venv\Scripts\Activate.ps1 on Windows PowerShell

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Inspecting and Running Notebooks
Every task is completely self-contained with its executed notebook, dataset download README, and generated figures:
```bash
# Run any notebook directly
jupyter notebook A2_Concrete_Strength/A2_notebook.ipynb
jupyter notebook B1_Fake_Job_Postings/B1_notebook.ipynb
jupyter notebook C1_English_Urdu_NMT/C1_notebook.ipynb
jupyter notebook C2_News_Summarisation/C2_notebook.ipynb
jupyter notebook C3_Image_Captioning/C3_notebook.ipynb
```

---

## Repository Directory Tree

```
AI101-ML-Assignment-70176119/
|-- README.md                         # Portfolio summary & benchmark tables for all 9 tasks
|-- AI_DISCLOSURE.md                  # Mandatory course AI disclosure covering all 9 tasks
|-- requirements.txt                  # Pinned environment requirements
|-- .gitignore                        # Git exclusion rules (<50MB compliance)
|-- A1_Breast_Cancer/                 # Task A1: Breast Cancer Diagnosis
|   |-- A1_notebook.ipynb             # Executed Jupyter notebook
|   |-- report/70176119_A1_Report.pdf # 2-Page IEEE Double-Column Report
|   |-- report/source/A1_Report.tex   # Overleaf-ready LaTeX source
|   |-- figures/                      # High-res diagnostic plots
|   `-- data/README.md                # Dataset documentation & reproducibility
|-- A2_Concrete_Strength/             # Task A2: Concrete Strength Prediction
|   |-- A2_notebook.ipynb
|   |-- report/70176119_A2_Report.pdf
|   |-- report/source/A2_Report.tex
|   |-- figures/
|   `-- data/README.md
|-- A3_Dry_Bean/                      # Task A3: Dry Bean Variety Classification
|   |-- A3_notebook.ipynb
|   |-- report/70176119_A3_Report.pdf
|   |-- report/source/A3_Report.tex
|   |-- figures/
|   `-- data/README.md
|-- B1_Fake_Job_Postings/             # Task B1: Fraudulent Job Postings Detection
|   |-- B1_notebook.ipynb
|   |-- report/70176119_B1_Report.pdf
|   |-- report/source/B1_Report.tex
|   |-- figures/
|   `-- data/README.md
|-- B2_Wine_Reviews/                  # Task B2: Wine Critic Score Prediction
|   |-- B2_notebook.ipynb
|   |-- report/70176119_B2_Report.pdf
|   |-- report/source/B2_Report.tex
|   |-- figures/
|   `-- data/README.md
|-- B3_Clothing_Reviews/              # Task B3: Clothing Reviews Star Ratings
|   |-- B3_notebook.ipynb
|   |-- report/70176119_B3_Report.pdf
|   |-- report/source/B3_Report.tex
|   |-- figures/
|   `-- data/README.md
|-- C1_English_Urdu_NMT/              # Task C1: English to Urdu NMT
|   |-- C1_notebook.ipynb
|   |-- report/70176119_C1_Report.pdf
|   |-- report/source/C1_Report.tex
|   |-- figures/
|   `-- data/README.md
|-- C2_News_Summarisation/            # Task C2: News Summarisation (BBC XSum)
|   |-- C2_notebook.ipynb
|   |-- report/70176119_C2_Report.pdf
|   |-- report/source/C2_Report.tex
|   |-- figures/
|   `-- data/README.md
`-- C3_Image_Captioning/              # Task C3: Image Captioning (Flickr8k)
    |-- C3_notebook.ipynb
    |-- report/70176119_C3_Report.pdf
    |-- report/source/C3_Report.tex
    |-- figures/
    `-- data/README.md
```
