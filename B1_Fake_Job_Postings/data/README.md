# Task B1: Real or Fake Job Postings (EMSCAD) Dataset

## Dataset Source
- **Repository:** Kaggle
- **Slug:** `shivamb/real-or-fake-fake-jobposting-prediction`
- **URL:** [https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Origin:** Employment Scam Aegean Dataset (EMSCAD), University of the Aegean.

## Dataset Characteristics
- **Instances:** 17,880 job advertisements
- **Class Balance:** 17,014 legitimate postings (95.2%), 866 fraudulent postings (4.8%)
- **Attributes:** 17 input features (free-text fields, categorical metadata, binary indicator flags)
- **Target:** `fraudulent` ($0 = \text{real}$, $1 = \text{fraud}$)

## Modalities
1. **Unstructured Text:** `title`, `company_profile`, `description`, `requirements`, `benefits`
2. **Categorical Metadata:** `employment_type`, `required_experience`, `required_education`, `industry`, `function`, `location`
3. **Binary Flags:** `telecommuting`, `has_company_logo`, `has_questions`

## Download Instructions
Download the dataset using the Kaggle API or web interface:
```bash
# Requires Kaggle API token (~/.kaggle/kaggle.json)
kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction --unzip -p data/
```
The raw `fake_job_postings.csv` file exceeds Git repository size limits and is excluded via `.gitignore`.
