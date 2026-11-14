# AI101 Machine Learning Assignment: Training Models on Three Data Modalities

**The University of Lahore**  
**Department of Intelligent Systems**  
**Course:** AI-101 Introduction to Artificial Intelligence (Fall 2026)  
**Instructor:** Dr. Muhammad Awais (AI Architect)

---

## Student Information

| Field | Details |
| :--- | :--- |
| **Student Name** | [Student Name] |
| **Roll Number** | [Roll Number] |
| **Section** | [Section] |
| **GitHub Repository** | `https://github.com/[username]/AI101-ML-Assignment-[RollNo]` |

---

## Graded Tasks Attempted

| Pool | Task ID | Task Description | Primary Metric | Notebook Link | Report PDF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pool A (Numerical)** | **A2** | Predicting Concrete Compressive Strength | $R^2 = 0.919$ (GBDT) | [`A2_Concrete_Strength/A2_notebook.ipynb`](./A2_Concrete_Strength/A2_notebook.ipynb) | [`A2_Concrete_Strength/report/A2_Report.pdf`](./A2_Concrete_Strength/report/A2_Report.pdf) |
| **Pool B (Mixed)** | **B1** | Detecting Fraudulent Job Postings | $F_1 = 0.835$, PR-AUC = 0.907 | [`B1_Fake_Job_Postings/B1_notebook.ipynb`](./B1_Fake_Job_Postings/B1_notebook.ipynb) | [`B1_Fake_Job_Postings/report/B1_Report.pdf`](./B1_Fake_Job_Postings/report/B1_Report.pdf) |
| **Pool C (Seq2Seq)** | **C1** | English-to-Urdu Neural Machine Translation | MarianMT: 22.76 BLEU; Beam ($k=4$): 3.18 BLEU | [`C1_English_Urdu_NMT/C1_notebook.ipynb`](./C1_English_Urdu_NMT/C1_notebook.ipynb) | [`C1_English_Urdu_NMT/report/C1_Report.pdf`](./C1_English_Urdu_NMT/report/C1_Report.pdf) |

---

## Key Experimental Results Summary

### Pool A: Task A2 (Concrete Compressive Strength)
- **Problem Formulation:** Continuous regression estimating 28-day compressive strength (MPa) from 8 mixture components.
- **Key Breakthrough:** Curing kinetics follow a logarithmic curve ($\log(\text{Age})$) and hydration follows Abrams' water-to-cement ratio ($w/c$). Incorporating domain features into linear regression improved $R^2$ from 0.628 to 0.833 (-33% RMSE).
- **Champion Model:** Gradient Boosted Decision Trees (GBDT) achieved **$R^2 = 0.919$**, **$\text{RMSE} = 4.555\text{ MPa}$**, and **$\text{MAE} = 2.917\text{ MPa}$**.

### Pool B: Task B1 (Fraudulent Job Postings Detection)
- **Problem Formulation:** Highly imbalanced binary classification (4.84% fraud rate across 17,880 postings) fusing unstructured text and structured metadata.
- **Key Finding (Multimodal Ablation):**
  - *Structured Metadata Only:* Low precision (0.232) and $F_1 = 0.322$ due to false alarms on legitimate early-stage startups lacking company logos.
  - *Text TF-IDF Only:* High precision (0.860) and $F_1 = 0.803$ (PR-AUC = 0.902).
  - *Combined Multimodal:* Preserves high precision while enhancing discrimination, achieving **$\text{PR-AUC} = 0.907$**, **$\text{ROC-AUC} = 0.992$**, and **$\text{Fraud } F_1 = 0.835$** (GBDT).

### Pool C: Task C1 (English-to-Urdu Neural Machine Translation)
- **Problem Formulation:** Sequence-to-sequence translation between English (SVO) and Urdu (SOV) on OPUS-100 parallel corpus.
- **Key Finding (Decoding & Attention):**
  - Beam Search ($k=4$) delivered a **+52.2% relative BLEU improvement** over Greedy Decoding (3.18 vs. 2.09; chrF = 15.81 vs. 14.97) by mitigating autoregressive search myopia.
  - Bahdanau Additive Attention successfully bridged long-range SVO $\to$ SOV word order reordering (attending to English verbs at sentence-end during Urdu verb generation).
  - Pretrained MarianMT reached **22.76 BLEU** (chrF = 44.23) via SentencePiece subword tokenization and 74M Transformer parameters.

---

## Interactive Pedagogical Learning Lab

An interactive visualizer is included in this repository to demonstrate the underlying machine learning mechanisms (hydraulic press simulation for concrete kinetics, dual-stream scam job ablation scanner, and English-to-Urdu neural attention alignment):

👉 Open [`Interactive_Learning_Hub.html`](./Interactive_Learning_Hub.html) in any modern web browser.

---

## How to Set Up and Run the Code

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/[username]/AI101-ML-Assignment-[RollNo].git
cd AI101-ML-Assignment-[RollNo]

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or .\.venv\Scripts\Activate.ps1 on Windows PowerShell

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Running the Notebooks
Each task is self-contained inside its dedicated folder:
```bash
jupyter notebook A2_Concrete_Strength/A2_notebook.ipynb
jupyter notebook B1_Fake_Job_Postings/B1_notebook.ipynb
jupyter notebook C1_English_Urdu_NMT/C1_notebook.ipynb
```

---

## Repository Structure

```
AI101-ML-Assignment-[RollNo]/
|-- README.md                    # Project overview & submission table
|-- AI_DISCLOSURE.md             # Mandatory course AI disclosure
|-- requirements.txt             # Exact library requirements
|-- .gitignore                   # Ignore data/, checkpoints, large binaries
|-- Interactive_Learning_Hub.html# Interactive learning lab
|-- A2_Concrete_Strength/        # Task A2: Concrete Compressive Strength
|   |-- A2_notebook.ipynb        # Fully executed notebook
|   |-- report/                  # 2-Page IEEE Conference Report PDF
|   |   |-- source/              # LaTeX source files (.tex)
|   |   `-- A2_Report.pdf
|   |-- figures/                 # High-resolution generated charts
|   `-- data/README.md           # Dataset download instructions
|-- B1_Fake_Job_Postings/        # Task B1: Fraudulent Job Postings Detection
|   |-- B1_notebook.ipynb
|   |-- report/
|   |   |-- source/
|   |   `-- B1_Report.pdf
|   |-- figures/
|   `-- data/README.md
`-- C1_English_Urdu_NMT/         # Task C1: English to Urdu NMT
    |-- C1_notebook.ipynb
    |-- report/
    |   |-- source/
    |   `-- C1_Report.pdf
    |-- figures/
    `-- data/README.md
```
