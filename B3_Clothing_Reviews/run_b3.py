"""
Task B3: Predicting Star Ratings of E-Commerce Clothing Reviews
AI-101 Introduction to Artificial Intelligence • Fall 2026
Department of Intelligent Systems • The University of Lahore
Student Name: Muhammad Zaryab Attiq | Roll No: 70176119 | Section: Section A
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("B3_Clothing_Reviews/figures", exist_ok=True)
os.makedirs("B3_Clothing_Reviews/data", exist_ok=True)

# 1. Load Dataset
print("Loading clothing reviews dataset...", flush=True)
ds = load_dataset('saattrupdan/womens-clothing-ecommerce-reviews')
df_raw = pd.concat([pd.DataFrame(ds['train']), pd.DataFrame(ds['test']), pd.DataFrame(ds['val'])], ignore_index=True)
print(f"Raw shape: {df_raw.shape}", flush=True)

# 2. Clean & Enforce Leakage Rule
# Drop rows with empty review text
df = df_raw.dropna(subset=['review_text']).copy()
df['review_text'] = df['review_text'].astype(str).str.strip()
df = df[df['review_text'] != ''].copy()

# MANDATORY LEAKAGE RULE: Drop 'recommended_ind' as input!
if 'recommended_ind' in df.columns:
    df = df.drop(columns=['recommended_ind'])

# Feature Engineering
df['log_pos_feedback'] = np.log1p(pd.to_numeric(df['positive_feedback_count'], errors='coerce').fillna(0))
df['review_len'] = df['review_text'].apply(lambda x: len(x.split()))
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').astype(int)

# Filter valid 1-5 ratings
df = df[df['rating'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)
print(f"Cleaned instances: {len(df)}")
print("Class counts (heavily skewed towards 5):\n", df['rating'].value_counts().sort_index())

# 3. EDA Visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))

# (a) Star Rating Distribution
sns.countplot(data=df, x='rating', palette='viridis', ax=ax1, edgecolor='black')
ax1.set_title("(a) Rating Distribution (Severe Class Skew toward 5 Stars)", fontsize=10, fontweight='bold')
ax1.set_xlabel("Star Rating (1–5)")
ax1.set_ylabel("Count")

# (b) Review Word Count by Star Rating
sns.boxplot(data=df, x='rating', y='review_len', palette='Blues', ax=ax2)
ax2.set_title("(b) Review Length Across Star Ratings", fontsize=10, fontweight='bold')
ax2.set_xlabel("Star Rating")
ax2.set_ylabel("Word Count")

# (c) Age Distribution by Rating
sns.kdeplot(data=df, x='age', hue='rating', common_norm=False, palette='tab10', ax=ax3)
ax3.set_title("(c) Age Density by Star Rating", fontsize=10, fontweight='bold')
ax3.set_xlabel("Customer Age")

# (d) Ratings by Department
dept_order = df['department_name'].value_counts().head(5).index
sns.countplot(data=df[df['department_name'].isin(dept_order)], x='department_name', hue='rating', palette='magma', ax=ax4, edgecolor='black')
ax4.set_title("(d) Rating Proportions across Top 5 Product Departments", fontsize=10, fontweight='bold')
ax4.set_xlabel("Department Name")
ax4.legend(title='Stars', loc='upper right')

plt.tight_layout()
fig.savefig("B3_Clothing_Reviews/figures/eda_clothing_distributions.png", dpi=300)
plt.close(fig)

# 4. Stratified 70/15/15 Split
y = df['rating'].values - 1 # Map 1-5 to 0-4 for zero-indexed classification
X = df[['review_text', 'age', 'log_pos_feedback', 'review_len', 'division_name', 'department_name', 'class_name']].copy()

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(0.15 / 0.85), stratify=y_train_val, random_state=RANDOM_SEED
)

print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

# 5. Preprocessing Pipelines for Ablation Study
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(max_categories=20, handle_unknown='ignore', sparse_output=True))
])

# Structured-Only Pipeline
struct_prep = ColumnTransformer([
    ('num', num_pipe, ['age', 'log_pos_feedback', 'review_len']),
    ('cat', cat_pipe, ['division_name', 'department_name', 'class_name'])
])

# Text-Only Pipeline
text_prep = ColumnTransformer([
    ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)), 'review_text')
])

# Combined Multimodal Pipeline
comb_prep = ColumnTransformer([
    ('num', num_pipe, ['age', 'log_pos_feedback', 'review_len']),
    ('cat', cat_pipe, ['division_name', 'department_name', 'class_name']),
    ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)), 'review_text')
])

X_tr_struct = struct_prep.fit_transform(X_train)
X_te_struct = struct_prep.transform(X_test)

X_tr_text = text_prep.fit_transform(X_train)
X_te_text = text_prep.transform(X_test)

X_tr_comb = comb_prep.fit_transform(X_train)
X_te_comb = comb_prep.transform(X_test)

# 6. Model Training & Ablation Study
def compute_ordinal_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

# (i) Structured-Only Logistic Regression
clf_struct = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
clf_struct.fit(X_tr_struct, y_train)
p_struct = clf_struct.predict(X_te_struct)

# (ii) Text-Only Logistic Regression
clf_text = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
clf_text.fit(X_tr_text, y_train)
p_text = clf_text.predict(X_te_text)

# (iii) Combined Logistic Regression (Unweighted)
clf_comb_unweighted = LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_SEED)
clf_comb_unweighted.fit(X_tr_comb, y_train)
p_comb_unw = clf_comb_unweighted.predict(X_te_comb)

# (iv) Combined Logistic Regression (Balanced Class Weighting)
clf_comb_weighted = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
clf_comb_weighted.fit(X_tr_comb, y_train)
p_comb_w = clf_comb_weighted.predict(X_te_comb)

# (v) Combined Linear SVM
clf_svm = LinearSVC(max_iter=2000, random_state=RANDOM_SEED)
clf_svm.fit(X_tr_comb, y_train)
p_svm = clf_svm.predict(X_te_comb)

# (vi) Combined Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
clf_rf.fit(X_tr_comb, y_train)
p_rf = clf_rf.predict(X_te_comb)

ablation_models = {
    'Structured-Only (Logistic Reg.)': p_struct,
    'Text-Only (Logistic Reg.)': p_text,
    'Combined Multimodal (Unweighted)': p_comb_unw,
    'Combined Multimodal (Balanced)': p_comb_w,
    'Combined Multimodal (Linear SVM - Champion)': p_svm,
    'Combined Multimodal (Random Forest)': p_rf
}

ablation_results = {}
for name, preds in ablation_models.items():
    ablation_results[name] = {
        'Accuracy': round(accuracy_score(y_test, preds), 4),
        'Macro-F1': round(f1_score(y_test, preds, average='macro'), 4),
        'Weighted-F1': round(f1_score(y_test, preds, average='weighted'), 4),
        'Ordinal_MAE': round(compute_ordinal_mae(y_test, preds), 4)
    }

print("\n--- 5-Class Ablation Results on Test Set ---")
print(pd.DataFrame(ablation_results).T)

# 7. Re-framing: 3-Class Formulation (Negative: 1-2, Neutral: 3, Positive: 4-5)
# Convert 0-4 (original 1-5) to 3-class:
# 0, 1 -> 0 (Negative)
# 2    -> 1 (Neutral)
# 3, 4 -> 2 (Positive)
def to_3class(y_arr):
    return np.where(y_arr <= 1, 0, np.where(y_arr == 2, 1, 2))

y_train_3c = to_3class(y_train)
y_test_3c = to_3class(y_test)

clf_3c = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
clf_3c.fit(X_tr_comb, y_train_3c)
p_3c = clf_3c.predict(X_te_comb)

res_3c = {
    'Accuracy': round(accuracy_score(y_test_3c, p_3c), 4),
    'Macro-F1': round(f1_score(y_test_3c, p_3c, average='macro'), 4),
    'Weighted-F1': round(f1_score(y_test_3c, p_3c, average='weighted'), 4)
}
print("\n--- 3-Class Re-Framed Classification Results ---")
print(res_3c)

# 8. Three Misclassified Reviews for Error Analysis
misclassified = []
test_df_copy = X_test.copy()
test_df_copy['true_rating'] = y_test + 1
test_df_copy['pred_rating'] = p_svm + 1
errors = test_df_copy[test_df_copy['true_rating'] != test_df_copy['pred_rating']].sample(3, random_state=RANDOM_SEED)

for idx, row in errors.iterrows():
    misclassified.append({
        'Review Excerpt': row['review_text'][:120] + '...',
        'True Rating': int(row['true_rating']),
        'Predicted Rating': int(row['pred_rating'])
    })

# 9. Key Result Composite Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# (a) 5x5 Confusion Matrix for Champion Model (Linear SVM)
cm = confusion_matrix(y_test + 1, p_svm + 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], ax=ax1)
ax1.set_title("(a) 5x5 Confusion Matrix on Test Set (Linear SVM)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Predicted Star Rating")
ax1.set_ylabel("True Star Rating")

# (b) Macro-F1 across Ablation Settings & 3-Class Reframing
comp_names = ['Structured', 'Text-Only', 'Combined (Unw)', 'Combined (Bal)', 'Combined (SVM)', '3-Class Reframe']
comp_f1s = [
    ablation_results['Structured-Only (Logistic Reg.)']['Macro-F1'],
    ablation_results['Text-Only (Logistic Reg.)']['Macro-F1'],
    ablation_results['Combined Multimodal (Unweighted)']['Macro-F1'],
    ablation_results['Combined Multimodal (Balanced)']['Macro-F1'],
    ablation_results['Combined Multimodal (Linear SVM - Champion)']['Macro-F1'],
    res_3c['Macro-F1']
]

bars = ax2.bar(comp_names, comp_f1s, color=['#94a3b8', '#818cf8', '#6366f1', '#a855f7', '#4f46e5', '#10b981'], edgecolor='black', width=0.55)
ax2.set_title("(b) Macro-F1 across Modalities & 3-Class Re-framing", fontsize=11, fontweight='bold')
ax2.set_ylabel("Macro-F1 Score")
ax2.set_xticklabels(comp_names, rotation=25, ha='right')
ax2.set_ylim(0, 0.95)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
fig.savefig("B3_Clothing_Reviews/figures/combined_key_result.png", dpi=300)
plt.close(fig)

# Save JSON Summary
summary_output = {
    'metrics_5class': ablation_results,
    'metrics_3class': res_3c,
    'misclassified_examples': misclassified
}

with open("B3_Clothing_Reviews/results_summary.json", "w") as f:
    json.dump(summary_output, f, indent=2)

print("\nTask B3 execution completed successfully! Saved to B3_Clothing_Reviews/results_summary.json", flush=True)
