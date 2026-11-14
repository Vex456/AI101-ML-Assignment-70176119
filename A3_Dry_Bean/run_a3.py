"""
Task A3: Dry Bean Variety Recognition from Shape Features
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

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("A3_Dry_Bean/figures", exist_ok=True)
os.makedirs("A3_Dry_Bean/data", exist_ok=True)

# 1. Ingest Data (UCI ID 602)
bean_data = fetch_ucirepo(id=602)
X = bean_data.data.features.copy()
y_raw = bean_data.data.targets.copy()

le = LabelEncoder()
y = le.fit_transform(y_raw.values.ravel())
class_names = list(le.classes_)

print(f"Dataset shape: {X.shape}, Classes ({len(class_names)}): {class_names}")
class_counts = pd.Series(y_raw.values.ravel()).value_counts()
print("Class counts:\n", class_counts)

# 2. EDA Visualizations
# Fig 1: 2-D PCA Scatter colored by bean class
scaler_eda = StandardScaler()
X_scaled_eda = scaler_eda.fit_transform(X)
pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca_2d = pca_2d.fit_transform(X_scaled_eda)

plt.figure(figsize=(9, 6))
df_pca = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
df_pca['Variety'] = y_raw.values.ravel()
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Variety', palette='tab10', alpha=0.6, s=25)
plt.title(f"2-D PCA Projection of Dry Bean Morphometric Descriptors (Var Exp: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=11, fontweight='bold')
plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% Variance)")
plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% Variance)")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("A3_Dry_Bean/figures/eda_pca_scatter.png", dpi=300)
plt.close()

# Fig 2: Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix of 16 Dry Bean Shape Features", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("A3_Dry_Bean/figures/eda_correlation_matrix.png", dpi=300)
plt.close()

# 3. Stratified 70/15/15 Split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(0.15 / 0.85), stratify=y_train_val, random_state=RANDOM_SEED
)

print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

# 4. Standardise Features inside Pipeline (Zero leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. Model Comparisons & Class Weighting Study
# Model 1: Multinomial Logistic Regression (Unweighted vs Balanced)
lr_none = LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_SEED)
lr_none.fit(X_train_scaled, y_train)
val_pred_lr_none = lr_none.predict(X_val_scaled)

lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
lr_balanced.fit(X_train_scaled, y_train)
val_pred_lr_bal = lr_balanced.predict(X_val_scaled)

print("\n--- Imbalance Study on Validation Set (Logistic Regression) ---")
print(f"Unweighted (class_weight=None):     Acc = {accuracy_score(y_val, val_pred_lr_none):.4f}, Macro-F1 = {f1_score(y_val, val_pred_lr_none, average='macro'):.4f}")
print(f"Weighted   (class_weight='balanced'): Acc = {accuracy_score(y_val, val_pred_lr_bal):.4f}, Macro-F1 = {f1_score(y_val, val_pred_lr_bal, average='macro'):.4f}")

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Model 3: MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_SEED)
mlp.fit(X_train_scaled, y_train)

# Model 4: Gradient Boosting (Champion Candidate)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_SEED)
gb.fit(X_train_scaled, y_train)

models = {
    'Multinomial Logistic Regression': lr_none,
    'Logistic Regression (Balanced)': lr_balanced,
    'Random Forest': rf,
    'Multi-Layer Perceptron (MLP)': mlp,
    'Gradient Boosting (Champion)': gb
}

val_metrics = {}
for name, m in models.items():
    pred = m.predict(X_val_scaled)
    val_metrics[name] = {
        'Accuracy': round(accuracy_score(y_val, pred), 4),
        'Macro-F1': round(f1_score(y_val, pred, average='macro'), 4),
        'Weighted-F1': round(f1_score(y_val, pred, average='weighted'), 4)
    }

print("\nValidation Set Evaluation Summary:")
print(pd.DataFrame(val_metrics).T)

# Evaluate on Test Set once
test_metrics = {}
for name, m in models.items():
    pred = m.predict(X_test_scaled)
    test_metrics[name] = {
        'Accuracy': round(accuracy_score(y_test, pred), 4),
        'Macro-F1': round(f1_score(y_test, pred, average='macro'), 4),
        'Weighted-F1': round(f1_score(y_test, pred, average='weighted'), 4)
    }

print("\nFinal Test Set Evaluation Summary:")
print(pd.DataFrame(test_metrics).T)

# 6. Dimensionality Study: Best Model on first k PCA components (k = 2, 4, 8, 16)
pca_k_results = {}
for k in [2, 4, 8, 16]:
    pca = PCA(n_components=k, random_state=RANDOM_SEED)
    X_tr_pca = pca.fit_transform(X_train_scaled)
    X_te_pca = pca.transform(X_test_scaled)
    clf_pca = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
    clf_pca.fit(X_tr_pca, y_train)
    p_k = clf_pca.predict(X_te_pca)
    macro_f1_k = f1_score(y_test, p_k, average='macro')
    var_exp = pca.explained_variance_ratio_.sum()
    pca_k_results[k] = {'Macro-F1': round(macro_f1_k, 4), 'Variance_Explained': round(var_exp, 4)}

print("\nPCA Dimensionality Study (k = 2, 4, 8, 16):")
print(pd.DataFrame(pca_k_results).T)

# 7. Combined Diagnostic Key Result Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: 7x7 Confusion Matrix for Best Model (Random Forest / GBDT)
best_test_pred = rf.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)
ax1.set_title("(a) Confusion Matrix on Test Set (Random Forest)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Predicted Variety")
ax1.set_ylabel("True Variety")

# Panel 2: Macro-F1 vs PCA Components k
ks = list(pca_k_results.keys())
f1_vals = [pca_k_results[k]['Macro-F1'] for k in ks]
full_f1 = test_metrics['Random Forest']['Macro-F1']
ax2.plot(ks, f1_vals, 'o-b', lw=2.5, markersize=8, label='PCA Subspace (k components)')
ax2.axhline(full_f1, color='r', linestyle='--', lw=1.5, label=f'Full 16 Features (F1={full_f1:.4f})')
ax2.axhline(full_f1 * 0.98, color='green', linestyle=':', lw=1.5, label='98% Target Threshold')
ax2.set_title("(b) Dimensionality Study: Macro-F1 vs PCA Components k", fontsize=11, fontweight='bold')
ax2.set_xlabel("Number of PCA Components (k)")
ax2.set_ylabel("Macro-F1 Score")
ax2.set_xticks(ks)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower right')

for k, val in zip(ks, f1_vals):
    ax2.text(k, val + 0.015, f"{val:.4f}", ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig("A3_Dry_Bean/figures/combined_key_result.png", dpi=300)
plt.close()

# Save Results Summary
results_summary = {
    'metrics': test_metrics,
    'pca_study': pca_k_results,
    'imbalance_study': {
        'unweighted': {'Accuracy': round(accuracy_score(y_val, val_pred_lr_none), 4), 'Macro-F1': round(f1_score(y_val, val_pred_lr_none, average='macro'), 4)},
        'balanced': {'Accuracy': round(accuracy_score(y_val, val_pred_lr_bal), 4), 'Macro-F1': round(f1_score(y_val, val_pred_lr_bal, average='macro'), 4)}
    },
    'per_class_metrics': classification_report(y_test, best_test_pred, target_names=class_names, output_dict=True)
}

with open("A3_Dry_Bean/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\nTask A3 execution completed successfully! Saved to A3_Dry_Bean/results_summary.json")
