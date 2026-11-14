"""
Task A1: Breast Cancer Diagnosis from Cell-Nucleus Measurements
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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("A1_Breast_Cancer/figures", exist_ok=True)
os.makedirs("A1_Breast_Cancer/data", exist_ok=True)

# 1. Ingest Data
data = load_breast_cancer(as_frame=True)
df = data.frame
# Scikit-learn: 0 = malignant, 1 = benign. We flip/map so malignant = 1 (positive class)
X = df.drop(columns=['target'])
y = (df['target'] == 0).astype(int) # 1: Malignant (212), 0: Benign (357)

print(f"Dataset Shape: {X.shape}, Malignant: {y.sum()} ({y.mean()*100:.2f}%), Benign: {(y==0).sum()}")

# 2. EDA Figures
# Fig 1: Correlation Matrix
plt.figure(figsize=(10, 8))
corr = X.iloc[:, :10].corr() # Mean features
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix of Mean Nuclear Descriptors (Collinearity in Radius, Perimeter, Area)", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("A1_Breast_Cancer/figures/eda_correlation_matrix.png", dpi=300)
plt.close()

# Fig 2: Class distribution and nuclear features boxplots
plt.figure(figsize=(9, 4.5))
features_to_plot = ['mean radius', 'mean texture', 'mean concavity', 'worst area']
df_plot = X[features_to_plot].copy()
df_plot['Diagnosis'] = np.where(y == 1, 'Malignant (1)', 'Benign (0)')
melted = pd.melt(df_plot, id_vars=['Diagnosis'], var_name='Feature', value_name='Value')

# Scale values for combined boxplot visualization
from sklearn.preprocessing import MinMaxScaler
df_norm = pd.DataFrame(MinMaxScaler().fit_transform(X[features_to_plot]), columns=features_to_plot)
df_norm['Diagnosis'] = np.where(y == 1, 'Malignant (1)', 'Benign (0)')
melted_norm = pd.melt(df_norm, id_vars=['Diagnosis'], var_name='Feature', value_name='Normalized Value')

sns.boxplot(data=melted_norm, x='Feature', y='Normalized Value', hue='Diagnosis', palette={'Malignant (1)': '#e11d48', 'Benign (0)': '#3b82f6'})
plt.title("Key Nuclear Features by Diagnosis (Normalized Distributions)", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("A1_Breast_Cancer/figures/eda_boxplots.png", dpi=300)
plt.close()

# 3. Stratified 80/20 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
)

print(f"Train samples: {len(X_train)} (Malignant: {y_train.sum()}), Test samples: {len(X_test)} (Malignant: {y_test.sum()})")

# 4. Pipelines & Tuning with 5-fold Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Baseline: Logistic Regression
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000))
])
param_lr = {'clf__C': [0.01, 0.1, 1.0, 10.0]}
grid_lr = GridSearchCV(pipe_lr, param_lr, cv=cv, scoring='recall', n_jobs=1)
grid_lr.fit(X_train, y_train)

# Model 2: SVM (RBF)
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True, random_state=RANDOM_SEED))
])
param_svm = {'clf__C': [0.1, 1.0, 10.0], 'clf__gamma': ['scale', 0.01, 0.1]}
grid_svm = GridSearchCV(pipe_svm, param_svm, cv=cv, scoring='recall', n_jobs=1)
grid_svm.fit(X_train, y_train)

# Model 3: Random Forest
pipe_rf = Pipeline([
    ('clf', RandomForestClassifier(random_state=RANDOM_SEED))
])
param_rf = {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [3, 5, None]}
grid_rf = GridSearchCV(pipe_rf, param_rf, cv=cv, scoring='recall', n_jobs=1)
grid_rf.fit(X_train, y_train)

# Model 4: Gradient Boosting (Champion candidate)
pipe_gb = Pipeline([
    ('clf', GradientBoostingClassifier(random_state=RANDOM_SEED))
])
param_gb = {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.05, 0.1], 'clf__max_depth': [2, 3]}
grid_gb = GridSearchCV(pipe_gb, param_gb, cv=cv, scoring='recall', n_jobs=1)
grid_gb.fit(X_train, y_train)

models = {
    'Logistic Regression (Baseline)': grid_lr.best_estimator_,
    'SVM (RBF Kernel)': grid_svm.best_estimator_,
    'Random Forest': grid_rf.best_estimator_,
    'Gradient Boosting': grid_gb.best_estimator_
}

results = {}
for name, m in models.items():
    y_pred = m.predict(X_test)
    y_prob = m.predict_proba(X_test)[:, 1]
    results[name] = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4),
        'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4)
    }

print("\nModel Evaluation Summary on Test Set:")
print(pd.DataFrame(results).T)

# 5. Threshold Study on Champion Model (Gradient Boosting / Logistic Regression)
# In screening, a missed malignancy (FN) is far more costly than FP.
# We move threshold until recall >= 0.95 (or 1.0) and report precision cost.
best_model = grid_gb.best_estimator_
y_prob_best = best_model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.01, 0.99, 100)
thresh_results = []
target_thresh = None

for t in thresholds:
    y_t = (y_prob_best >= t).astype(int)
    rec = recall_score(y_test, y_t, zero_division=0)
    prec = precision_score(y_test, y_t, zero_division=0)
    thresh_results.append({'threshold': t, 'recall': rec, 'precision': prec})
    if rec >= 0.95 and target_thresh is None and t > 0.1:
        target_thresh = t

# Let's find highest threshold achieving recall >= 0.95
df_thresh = pd.DataFrame(thresh_results)
valid_thresh = df_thresh[df_thresh['recall'] >= 0.95]
best_t_row = valid_thresh.sort_values(by='threshold', ascending=False).iloc[0]
opt_thresh = best_t_row['threshold']
opt_recall = best_t_row['recall']
opt_prec = best_t_row['precision']

# Standard threshold (0.5)
std_recall = results['Gradient Boosting']['Recall']
std_prec = results['Gradient Boosting']['Precision']

print(f"\nThreshold Study:")
print(f"Standard Threshold 0.50 -> Recall: {std_recall:.4f}, Precision: {std_prec:.4f}")
print(f"Calibrated Threshold {opt_thresh:.2f} -> Recall: {opt_recall:.4f}, Precision: {opt_prec:.4f} (Precision cost: {std_prec - opt_prec:.4f})")

# 6. Combined Key Diagnostic Result Figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))

# ROC Curves
for name, m in models.items():
    fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
    ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC={results[name]['ROC-AUC']:.3f})")
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_title("(a) Receiver Operating Characteristic (ROC)", fontsize=10, fontweight='bold')
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(alpha=0.3)

# Precision-Recall Curves
for name, m in models.items():
    p, r, _ = precision_recall_curve(y_test, m.predict_proba(X_test)[:, 1])
    ax2.plot(r, p, lw=2, label=name)
ax2.set_title("(b) Precision-Recall Curves", fontsize=10, fontweight='bold')
ax2.set_xlabel("Recall (Malignant)")
ax2.set_ylabel("Precision")
ax2.legend(loc='lower left', fontsize=8)
ax2.grid(alpha=0.3)

# Confusion Matrix for Calibrated Champion Model
y_opt = (y_prob_best >= opt_thresh).astype(int)
cm = confusion_matrix(y_test, y_opt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3,
            xticklabels=['Benign (Pred)', 'Malignant (Pred)'],
            yticklabels=['Benign (True)', 'Malignant (True)'])
ax3.set_title(f"(c) Confusion Matrix (Threshold tau={opt_thresh:.2f})", fontsize=10, fontweight='bold')

# Feature Importances (Gradient Boosting)
gb_model = grid_gb.best_estimator_.named_steps['clf']
importances = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(8)
importances.plot(kind='barh', color='#4f46e5', edgecolor='black', ax=ax4)
ax4.set_title("(d) Top 8 Predictive Nuclear Descriptors (GBDT)", fontsize=10, fontweight='bold')
ax4.set_xlabel("Relative Gini Feature Importance")

plt.tight_layout()
plt.savefig("A1_Breast_Cancer/figures/combined_key_result.png", dpi=300)
plt.close()

# Save results summary JSON
summary_output = {
    'metrics': results,
    'threshold_study': {
        'default_threshold': 0.50,
        'default_recall': std_recall,
        'default_precision': std_prec,
        'calibrated_threshold': round(float(opt_thresh), 3),
        'calibrated_recall': round(float(opt_recall), 4),
        'calibrated_precision': round(float(opt_prec), 4),
        'precision_cost': round(float(std_prec - opt_prec), 4)
    },
    'best_params': {
        'Logistic Regression': str(grid_lr.best_params_),
        'SVM': str(grid_svm.best_params_),
        'Random Forest': str(grid_rf.best_params_),
        'Gradient Boosting': str(grid_gb.best_params_)
    }
}

with open("A1_Breast_Cancer/results_summary.json", "w") as f:
    json.dump(summary_output, f, indent=2)

print("\nTask A1 execution completed successfully! Results saved to A1_Breast_Cancer/results_summary.json")
