# AI Use Disclosure

This disclosure covers all tasks submitted as part of the **AI-101 Introduction to Artificial Intelligence** Machine Learning Assignment (*Training Machine Learning Models on Three Data Modalities*), Fall 2026, Department of Intelligent Systems, The University of Lahore.

---

### Disclosure Statement

I used **Antigravity (powered by Google DeepMind Gemini)** for:
1. Architectural scaffolding and mathematical formulation of the learning problems.
2. Generating interactive pedagogical visual simulations (`Interactive_Learning_Hub.html`).
3. Scaffolding boilerplate code for data loading, scikit-learn pipelines, and diagnostic plots.
4. Polishing the wording, academic structure, and LaTeX formatting for the IEEE two-column reports.

The following parts were generated or substantially assisted by AI:
- Initial notebook cell structuring and cross-validation pipelines (`A2_Concrete_Strength`, `B1_Fake_Job_Postings`, `C1_English_Urdu_NMT`).
- Mathematical formulation in Section 2 (*Problem Statement*) and Section 5 (*Methodology*) of each report.
- Interactive learning visualizations in `Interactive_Learning_Hub.html`.

### Verification Methodology
I verified all content by:
- Re-running every notebook from top to bottom with fixed random seeds (`seed = 42`) in the local environment and cross-verifying outputs.
- Checking every single reported metric in Table 1, Table 2, and Table 3 against raw notebook execution variables to ensure 100% concordance with zero fabrication.
- Reviewing all data transformation steps to verify that every scaler, encoder, and vectorizer is fitted strictly on the training fold (zero test leakage).

All experiments, results, and analysis in this report were produced and checked by me, and I take full responsibility for this work.
