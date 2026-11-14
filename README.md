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

| Pool | Task ID | Task Description | Notebook Link | Report PDF |
| :--- | :--- | :--- | :--- | :--- |
| **Pool A (Numerical)** | **A2** | Predicting Concrete Compressive Strength | [`A2_Concrete_Strength/A2_notebook.ipynb`](./A2_Concrete_Strength/A2_notebook.ipynb) | [`A2_Concrete_Strength/report/[RollNo]_A2_Report.pdf`](./A2_Concrete_Strength/report/) |
| **Pool B (Mixed)** | **B1** | Detecting Fraudulent Job Postings | [`B1_Fake_Job_Postings/B1_notebook.ipynb`](./B1_Fake_Job_Postings/B1_notebook.ipynb) | [`B1_Fake_Job_Postings/report/[RollNo]_B1_Report.pdf`](./B1_Fake_Job_Postings/report/) |
| **Pool C (Seq2Seq)** | **C1** | English-to-Urdu Neural Machine Translation | [`C1_English_Urdu_NMT/C1_notebook.ipynb`](./C1_English_Urdu_NMT/C1_notebook.ipynb) | [`C1_English_Urdu_NMT/report/[RollNo]_C1_Report.pdf`](./C1_English_Urdu_NMT/report/) |

---

## Interactive Learning Hub

An interactive visualizer is included in this repository to demonstrate the underlying machine learning mechanisms (hydraulic press simulation for concrete kinetics, dual-stream scam job ablation scanner, and English-to-Urdu neural attention alignment):

👉 Open [`Interactive_Learning_Hub.html`](./Interactive_Learning_Hub.html) in any modern browser.

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

# Install dependencies
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
|   |   |-- source/              # LaTeX source files (.tex, .bib)
|   |   `-- [RollNo]_A2_Report.pdf
|   |-- figures/                 # High-resolution generated charts
|   `-- data/README.md           # Dataset download instructions
|-- B1_Fake_Job_Postings/        # Task B1: Fraudulent Job Postings Detection
|   |-- B1_notebook.ipynb
|   |-- report/
|   |-- figures/
|   `-- data/README.md
`-- C1_English_Urdu_NMT/         # Task C1: English to Urdu NMT
    |-- C1_notebook.ipynb
    |-- report/
    |-- figures/
    `-- data/README.md
```
