"""
Task B1: Detecting Fraudulent Job Postings (EMSCAD)
AI-101 Machine Learning Assignment, Fall 2026
Department of Intelligent Systems, The University of Lahore
Instructor: Dr. Muhammad Awais
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, classification_report
)

def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    os.makedirs("B1_Fake_Job_Postings/figures", exist_ok=True)
    csv_path = "B1_Fake_Job_Postings/data/fake_job_postings.csv"

    print("=" * 60, flush=True)
    print("1. INGESTING EMSCAD DATASET", flush=True)
    print("=" * 60, flush=True)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} postings with {df.shape[1]} attributes.", flush=True)

    # ==============================================================================
    # 2. FEATURE ENGINEERING & PREPARATION
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("2. COMPUTING STRUCTURED SIGNALS & TEXT CONCATENATION", flush=True)
    print("=" * 60, flush=True)

    # 1. Missingness indicators (Key fraud signals)
    df['missing_company_profile'] = df['company_profile'].isnull().astype(int)
    df['missing_salary'] = df['salary_range'].isnull().astype(int)
    df['missing_requirements'] = df['requirements'].isnull().astype(int)
    df['missing_benefits'] = df['benefits'].isnull().astype(int)

    # 2. Text concatenation (Title, profile, description, requirements, benefits)
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for c in text_cols:
        df[c] = df[c].fillna('').astype(str)

    df['combined_text'] = (
        df['title'] + " " +
        df['company_profile'] + " " +
        df['description'] + " " +
        df['requirements'] + " " +
        df['benefits']
    ).str.lower().str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)

    # 3. Text length & word count features
    df['char_count'] = df['combined_text'].apply(len)
    df['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))

    # Categorical cleaning
    cat_cols = ['employment_type', 'required_experience', 'required_education']
    for c in cat_cols:
        df[c] = df[c].fillna('Missing')

    # Binary flags
    binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    for c in binary_cols:
        df[c] = df[c].fillna(0).astype(int)

    # ==============================================================================
    # 3. EXPLORATORY DATA ANALYSIS (EDA)
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("3. GENERATING EDA VISUALIZATIONS", flush=True)
    print("=" * 60, flush=True)

    # Figure 1A: Fraud Rate by Logo and Employment Type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    logo_fraud = df.groupby('has_company_logo')['fraudulent'].mean() * 100
    sns.barplot(x=['No Logo (0)', 'Has Logo (1)'], y=logo_fraud.values, palette=['#ef4444', '#10b981'], ax=ax1, edgecolor='black')
    ax1.set_title("Fraud Rate (%) by Company Logo Presence", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Fraudulent Postings (%)")
    for i, v in enumerate(logo_fraud.values):
        ax1.text(i, v + 0.3, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=10)

    emp_fraud = df.groupby('employment_type')['fraudulent'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=emp_fraud.index, y=emp_fraud.values, palette='Blues_r', ax=ax2, edgecolor='black')
    ax2.set_title("Fraud Rate (%) by Employment Type", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Fraudulent Postings (%)")
    ax2.tick_params(axis='x', rotation=30)
    for i, v in enumerate(emp_fraud.values):
        ax2.text(i, v + 0.3, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    fig.savefig("B1_Fake_Job_Postings/figures/eda_missingness_fraud_rate.png", dpi=300)
    plt.close(fig)

    # Figure 1B: Text Length Distribution per Class
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.kdeplot(data=df[df['fraudulent'] == 0]['word_count'], label='Legitimate (Real)', color='#3b82f6', fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(data=df[df['fraudulent'] == 1]['word_count'], label='Fraudulent (Fake)', color='#ef4444', fill=True, alpha=0.3, ax=ax)
    ax.set_xlim(0, 1500)
    ax.set_title("Word Count Distribution: Real vs. Fraudulent Postings", fontsize=11, fontweight='bold')
    ax.set_xlabel("Word Count")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig("B1_Fake_Job_Postings/figures/eda_text_length_dist.png", dpi=300)
    plt.close(fig)

    # ==============================================================================
    # 4. DATA SPLIT: STRATIFIED 70 / 15 / 15 (ZERO LEAKAGE)
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("4. STRATIFIED 70/15/15 SPLIT", flush=True)
    print("=" * 60, flush=True)

    y = df['fraudulent']
    features = [
        'combined_text',
        'telecommuting', 'has_company_logo', 'has_questions',
        'missing_company_profile', 'missing_salary', 'missing_requirements', 'missing_benefits',
        'char_count', 'word_count',
        'employment_type', 'required_experience', 'required_education'
    ]
    X = df[features]

    # Split 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    # Split Temp into 50% Val, 50% Test (15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"Training split:   {len(X_train)} samples ({y_train.sum()} fraud = {y_train.mean()*100:.2f}%)", flush=True)
    print(f"Validation split: {len(X_val)} samples ({y_val.sum()} fraud = {y_val.mean()*100:.2f}%)", flush=True)
    print(f"Testing split:    {len(X_test)} samples ({y_test.sum()} fraud = {y_test.mean()*100:.2f}%)", flush=True)

    # Define Feature Transformers
    num_cols = ['char_count', 'word_count']
    bin_cols = ['telecommuting', 'has_company_logo', 'has_questions',
                'missing_company_profile', 'missing_salary', 'missing_requirements', 'missing_benefits']
    cat_cols = ['employment_type', 'required_experience', 'required_education']

    # Structured transformer
    struct_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('bin', 'passthrough', bin_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Text transformer (uni + bi-grams, max 15,000 features)
    text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, stop_words='english')

    # Combined Multimodal transformer
    multimodal_transformer = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, stop_words='english'), 'combined_text'),
            ('num', StandardScaler(), num_cols),
            ('bin', 'passthrough', bin_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # ==============================================================================
    # 5. MODEL BENCHMARKING & VALIDATION THRESHOLD TUNING
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("5. MODEL BENCHMARKING & THRESHOLD CALIBRATION", flush=True)
    print("=" * 60, flush=True)

    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        'SGD Classifier (Linear SVM)': SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED)
    }

    results = []

    for name, clf in models.items():
        pipe = Pipeline([
            ('preprocess', multimodal_transformer),
            ('clf', clf)
        ])
        pipe.fit(X_train, y_train)

        # Predict probabilities on Validation set
        val_probs = pipe.predict_proba(X_val)[:, 1]
        
        # Grid search decision threshold on validation set to maximize F1 (fraud)
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[min(best_idx, len(thresholds)-1)]
        best_val_f1 = f1_scores[best_idx]

        # Evaluate on Test Set using optimal validation threshold
        test_probs = pipe.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= best_threshold).astype(int)

        t_prec = precision_score(y_test, test_preds, pos_label=1)
        t_rec = recall_score(y_test, test_preds, pos_label=1)
        t_f1 = f1_score(y_test, test_preds, pos_label=1)
        t_roc_auc = roc_auc_score(y_test, test_probs)
        t_p, t_r, _ = precision_recall_curve(y_test, test_probs)
        t_pr_auc = auc(t_r, t_p)

        results.append({
            'Model': name,
            'Modality': 'Combined',
            'Best_Threshold': round(float(best_threshold), 3),
            'Val_F1': round(float(best_val_f1), 3),
            'Test_Precision': round(float(t_prec), 3),
            'Test_Recall': round(float(t_rec), 3),
            'Test_F1': round(float(t_f1), 3),
            'PR_AUC': round(float(t_pr_auc), 3),
            'ROC_AUC': round(float(t_roc_auc), 3),
            'pipe': pipe,
            'test_probs': test_probs,
            'test_preds': test_preds
        })
        print(f"[{name:28s}] Val F1: {best_val_f1:.3f} (thresh={best_threshold:.2f}) | Test Prec: {t_prec:.3f} | Rec: {t_rec:.3f} | F1: {t_f1:.3f} | PR-AUC: {t_pr_auc:.3f} | ROC-AUC: {t_roc_auc:.3f}", flush=True)

    # ==============================================================================
    # 6. MANDATORY ABLATION STUDY (CHAMPION MODEL: LOGISTIC REGRESSION)
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("6. EXECUTING MANDATORY 3-WAY ABLATION STUDY", flush=True)
    print("=" * 60, flush=True)

    ablation_results = []
    champion_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)

    # 1. Structured Features Only
    pipe_struct = Pipeline([
        ('prep', struct_transformer),
        ('clf', champion_clf)
    ])
    pipe_struct.fit(X_train[num_cols + bin_cols + cat_cols], y_train)
    val_probs_s = pipe_struct.predict_proba(X_val[num_cols + bin_cols + cat_cols])[:, 1]
    p_s, r_s, th_s = precision_recall_curve(y_val, val_probs_s)
    f1_s = 2 * (p_s * r_s) / (p_s + r_s + 1e-10)
    best_th_s = th_s[min(np.argmax(f1_s), len(th_s)-1)]

    test_probs_s = pipe_struct.predict_proba(X_test[num_cols + bin_cols + cat_cols])[:, 1]
    preds_s = (test_probs_s >= best_th_s).astype(int)
    p_curve_s, r_curve_s, _ = precision_recall_curve(y_test, test_probs_s)
    ablation_results.append({
        'Feature_Set': '1. Structured Only',
        'Precision': round(precision_score(y_test, preds_s), 3),
        'Recall': round(recall_score(y_test, preds_s), 3),
        'F1_Score': round(f1_score(y_test, preds_s), 3),
        'PR_AUC': round(auc(r_curve_s, p_curve_s), 3),
        'ROC_AUC': round(roc_auc_score(y_test, test_probs_s), 3),
        'test_probs': test_probs_s
    })

    # 2. Text Features Only
    pipe_text = Pipeline([
        ('prep', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, stop_words='english')),
        ('clf', champion_clf)
    ])
    pipe_text.fit(X_train['combined_text'], y_train)
    val_probs_t = pipe_text.predict_proba(X_val['combined_text'])[:, 1]
    p_t, r_t, th_t = precision_recall_curve(y_val, val_probs_t)
    f1_t = 2 * (p_t * r_t) / (p_t + r_t + 1e-10)
    best_th_t = th_t[min(np.argmax(f1_t), len(th_t)-1)]

    test_probs_t = pipe_text.predict_proba(X_test['combined_text'])[:, 1]
    preds_t = (test_probs_t >= best_th_t).astype(int)
    p_curve_t, r_curve_t, _ = precision_recall_curve(y_test, test_probs_t)
    ablation_results.append({
        'Feature_Set': '2. Text Only (TF-IDF)',
        'Precision': round(precision_score(y_test, preds_t), 3),
        'Recall': round(recall_score(y_test, preds_t), 3),
        'F1_Score': round(f1_score(y_test, preds_t), 3),
        'PR_AUC': round(auc(r_curve_t, p_curve_t), 3),
        'ROC_AUC': round(roc_auc_score(y_test, test_probs_t), 3),
        'test_probs': test_probs_t
    })

    # 3. Combined Multimodal (from results[0])
    comb_res = results[0]
    p_curve_c, r_curve_c, _ = precision_recall_curve(y_test, comb_res['test_probs'])
    ablation_results.append({
        'Feature_Set': '3. Combined Multimodal',
        'Precision': comb_res['Test_Precision'],
        'Recall': comb_res['Test_Recall'],
        'F1_Score': comb_res['Test_F1'],
        'PR_AUC': comb_res['PR_AUC'],
        'ROC_AUC': comb_res['ROC_AUC'],
        'test_probs': comb_res['test_probs']
    })

    for ab in ablation_results:
        print(f"[{ab['Feature_Set']:25s}] Prec: {ab['Precision']:.3f} | Rec: {ab['Recall']:.3f} | F1: {ab['F1_Score']:.3f} | PR-AUC: {ab['PR_AUC']:.3f} | ROC-AUC: {ab['ROC_AUC']:.3f}", flush=True)

    # ==============================================================================
    # 7. DIAGNOSTIC PLOTS & ERROR ANALYSIS
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("7. GENERATING DIAGNOSTIC PLOTS", flush=True)
    print("=" * 60, flush=True)

    # Figure 2: Combined PR Curve & Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Subplot A: Precision-Recall Curves across Ablation
    for ab, color in zip(ablation_results, ['#f59e0b', '#06b6d4', '#10b981']):
        p_c, r_c, _ = precision_recall_curve(y_test, ab['test_probs'])
        ax1.plot(r_c, p_c, label=f"{ab['Feature_Set']} (PR-AUC={ab['PR_AUC']:.3f})", lw=2, color=color)
    ax1.axhline(y_test.mean(), color='gray', linestyle='--', label=f'Baseline Chance ({y_test.mean()*100:.1f}%)')
    ax1.set_title("Precision-Recall Curves (Ablation Comparison)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Recall (Fraud Class)")
    ax1.set_ylabel("Precision (Fraud Class)")
    ax1.legend(loc="lower left", fontsize=8.5)

    # Subplot B: Confusion Matrix
    cm = confusion_matrix(y_test, comb_res['test_preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
                xticklabels=['Pred Real (0)', 'Pred Fraud (1)'],
                yticklabels=['Actual Real (0)', 'Actual Fraud (1)'])
    ax2.set_title(f"Test Confusion Matrix ({comb_res['Model']} Combined)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig("B1_Fake_Job_Postings/figures/combined_key_result.png", dpi=300)
    plt.close(fig)

    # Save summary json
    summary_data = {
        'models': [{k: v for k, v in r.items() if k not in ['pipe', 'test_probs', 'test_preds']} for r in results],
        'ablation': [{k: v for k, v in a.items() if k != 'test_probs'} for a in ablation_results]
    }
    with open("B1_Fake_Job_Postings/results_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nTask B1 pipeline complete! Results saved to B1_Fake_Job_Postings/results_summary.json", flush=True)

if __name__ == '__main__':
    main()
