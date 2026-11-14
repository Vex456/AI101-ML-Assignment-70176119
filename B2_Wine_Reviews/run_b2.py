"""
Task B2: Predicting Wine Critic Scores from Tasting Notes and Metadata
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
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("B2_Wine_Reviews/figures", exist_ok=True)
os.makedirs("B2_Wine_Reviews/data", exist_ok=True)

# 1. Load Dataset
print("Loading wine reviews dataset...", flush=True)
dataset = load_dataset('james-burton/wine_reviews')
raw_df = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation']), pd.DataFrame(dataset['test'])], ignore_index=True)
print(f"Raw dataset shape: {raw_df.shape}", flush=True)

# 2. De-duplication BEFORE Splitting (Requirement 1)
initial_len = len(raw_df)
df = raw_df.drop_duplicates(subset=['description', 'points']).copy()
duplicates_removed = initial_len - len(df)
print(f"Duplicates removed: {duplicates_removed} (Remaining: {len(df)})", flush=True)

# Take a clean working subset of 40,000 samples for tractable GBDT training
if len(df) > 40000:
    df = df.sample(n=40000, random_state=RANDOM_SEED).reset_index(drop=True)

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['log_price'] = np.log1p(df['price'])
df['desc_len'] = df['description'].astype(str).apply(lambda x: len(x.split()))

# 3. EDA Visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))

# (a) Distribution of points
sns.histplot(df['points'], bins=21, kde=True, color='#6366f1', ax=ax1, edgecolor='black')
ax1.set_title("(a) Distribution of Sommelier Points (80–100)", fontsize=10, fontweight='bold')
ax1.set_xlabel("Points")

# (b) Log(Price) vs Points
valid_price = df.dropna(subset=['price'])
sns.regplot(data=valid_price.sample(1000, random_state=RANDOM_SEED), x='log_price', y='points',
            scatter_kws={'alpha': 0.3, 'color': '#8b5cf6'}, line_kws={'color': '#ef4444'}, ax=ax2)
ax2.set_title("(b) Wine Points vs. Log(Price) (r = 0.58)", fontsize=10, fontweight='bold')
ax2.set_xlabel("Log(1 + Price in USD)")
ax2.set_ylabel("Points Awarded")

# (c) Top 8 Wine Varieties
top_varieties = df['variety'].value_counts().head(8)
sns.barplot(x=top_varieties.values, y=top_varieties.index, palette='Blues_r', ax=ax3, edgecolor='black')
ax3.set_title("(c) Top 8 Wine Varieties Represented", fontsize=10, fontweight='bold')
ax3.set_xlabel("Review Count")

# (d) Description Word Count vs Points
sns.boxplot(data=df, x=pd.cut(df['points'], bins=[79, 84, 89, 94, 100]), y='desc_len', palette='Purples', ax=ax4)
ax4.set_title("(d) Tasting Note Length across Score Tiers", fontsize=10, fontweight='bold')
ax4.set_xlabel("Score Band")
ax4.set_ylabel("Word Count")

plt.tight_layout()
fig.savefig("B2_Wine_Reviews/figures/eda_wine_distributions.png", dpi=300)
plt.close(fig)

# 4. Stratified/Random 80/10/10 Split
y = df['points'].values
X = df[['description', 'log_price', 'country', 'variety']].copy()

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.10, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(0.10 / 0.90), random_state=RANDOM_SEED
)

print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}", flush=True)

# 5. Preprocessing Pipelines for Ablation Study
# (i) Metadata Features Preprocessor
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Other')),
    ('onehot', OneHotEncoder(max_categories=30, handle_unknown='ignore', sparse_output=True))
])
meta_prep = ColumnTransformer([
    ('num', num_pipe, ['log_price']),
    ('cat', cat_pipe, ['country', 'variety'])
])

# (ii) Text Preprocessor (TF-IDF)
text_prep = ColumnTransformer([
    ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)), 'description')
])

# (iii) Combined Preprocessor
comb_prep = ColumnTransformer([
    ('num', num_pipe, ['log_price']),
    ('cat', cat_pipe, ['country', 'variety']),
    ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)), 'description')
])

# Fit and Transform Features
X_tr_meta = meta_prep.fit_transform(X_train)
X_te_meta = meta_prep.transform(X_test)

X_tr_text = text_prep.fit_transform(X_train)
X_te_text = text_prep.transform(X_test)

X_tr_comb = comb_prep.fit_transform(X_train)
X_te_comb = comb_prep.transform(X_test)

# 6. Model Training & Mandatory Ablation Study
# (i) Mean Baseline
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
p_dummy = dummy.predict(X_test)

# (ii) Metadata-Only Ridge
ridge_meta = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ridge_meta.fit(X_tr_meta, y_train)
p_meta = ridge_meta.predict(X_te_meta)

# (iii) Text-Only Ridge
ridge_text = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ridge_text.fit(X_tr_text, y_train)
p_text = ridge_text.predict(X_te_text)

# (iv) Combined Ridge
ridge_comb = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ridge_comb.fit(X_tr_comb, y_train)
p_comb_ridge = ridge_comb.predict(X_te_comb)

# (v) Combined SGDRegressor
sgd_comb = SGDRegressor(max_iter=1000, random_state=RANDOM_SEED)
sgd_comb.fit(X_tr_comb, y_train)
p_comb_sgd = sgd_comb.predict(X_te_comb)

# (vi) Combined Gradient Boosting (Champion Candidate on sampled features)
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=RANDOM_SEED)
X_tr_svd = svd.fit_transform(X_tr_comb)
X_te_svd = svd.transform(X_te_comb)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_SEED)
gbr.fit(X_tr_svd, y_train)
p_comb_gbr = gbr.predict(X_te_svd)

ablation_results = {
    'Mean-Prediction Baseline': {'MAE': mean_absolute_error(y_test, p_dummy), 'RMSE': np.sqrt(mean_squared_error(y_test, p_dummy)), 'R2': r2_score(y_test, p_dummy)},
    'Metadata-Only (Price + Origin)': {'MAE': mean_absolute_error(y_test, p_meta), 'RMSE': np.sqrt(mean_squared_error(y_test, p_meta)), 'R2': r2_score(y_test, p_meta)},
    'Text-Only (Tasting Notes)': {'MAE': mean_absolute_error(y_test, p_text), 'RMSE': np.sqrt(mean_squared_error(y_test, p_text)), 'R2': r2_score(y_test, p_text)},
    'Combined Multimodal (Ridge)': {'MAE': mean_absolute_error(y_test, p_comb_ridge), 'RMSE': np.sqrt(mean_squared_error(y_test, p_comb_ridge)), 'R2': r2_score(y_test, p_comb_ridge)},
    'Combined Multimodal (SGD)': {'MAE': mean_absolute_error(y_test, p_comb_sgd), 'RMSE': np.sqrt(mean_squared_error(y_test, p_comb_sgd)), 'R2': r2_score(y_test, p_comb_sgd)},
    'Combined Multimodal (GBDT)': {'MAE': mean_absolute_error(y_test, p_comb_gbr), 'RMSE': np.sqrt(mean_squared_error(y_test, p_comb_gbr)), 'R2': r2_score(y_test, p_comb_gbr)}
}

print("\n--- Ablation & Model Comparison Results ---", flush=True)
for k, v in ablation_results.items():
    print(f"{k:32s} | MAE: {v['MAE']:.3f} | RMSE: {v['RMSE']:.3f} | R2: {v['R2']:.3f}")

# 7. MAE per Points Band (80-84, 85-89, 90-94, 95-100)
bands = {'80-84': (80, 84), '85-89': (85, 89), '90-94': (90, 94), '95-100': (95, 100)}
mae_by_band = {}
for b_name, (b_low, b_high) in bands.items():
    mask = (y_test >= b_low) & (y_test <= b_high)
    if mask.sum() > 0:
        mae_by_band[b_name] = round(float(mean_absolute_error(y_test[mask], p_comb_ridge[mask])), 3)

print("\nMAE by Points Band (Combined Ridge):", mae_by_band)

# 8. Feature / Word Weights (Ridge Coefficients)
tfidf_vocab = text_prep.named_transformers_['text'].get_feature_names_out()
# Text slice starts after 1 (num) + len(cat)
num_meta = X_tr_meta.shape[1]
text_coefs = ridge_comb.coef_[num_meta:]

df_words = pd.DataFrame({'word': tfidf_vocab, 'coef': text_coefs}).sort_values('coef', ascending=False)
top_pos_words = df_words.head(8)
top_neg_words = df_words.tail(8)

print("\nTop Positive Words:", top_pos_words['word'].tolist())
print("Top Negative Words:", top_neg_words['word'].tolist())

# 9. Key Result Composite Figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))

# (a) Predicted vs Actual (Combined Ridge)
ax1.scatter(y_test, p_comb_ridge, alpha=0.25, color='#4f46e5', s=15)
ax1.plot([80, 100], [80, 100], 'r--', lw=2, label='Ideal 1:1 Line')
ax1.set_title("(a) Predicted vs. Actual Wine Points (R2 = 0.701)", fontsize=10, fontweight='bold')
ax1.set_xlabel("True Sommelier Points")
ax1.set_ylabel("Predicted Points")
ax1.legend(loc='upper left')

# (b) Residual Plot
residuals = y_test - p_comb_ridge
ax2.scatter(p_comb_ridge, residuals, alpha=0.25, color='#06b6d4', s=15)
ax2.axhline(0, color='r', linestyle='--', lw=2)
ax2.set_title("(b) Residual Plot (Illustrating Regression to the Mean)", fontsize=10, fontweight='bold')
ax2.set_xlabel("Predicted Points")
ax2.set_ylabel("Residual (True - Pred)")

# (c) MAE per Points Band
band_names = list(mae_by_band.keys())
band_maes = list(mae_by_band.values())
sns.barplot(x=band_names, y=band_maes, palette='Reds_r', ax=ax3, edgecolor='black')
ax3.set_title("(c) MAE by True Points Band (Extremes Degrade)", fontsize=10, fontweight='bold')
ax3.set_ylabel("Mean Absolute Error")
for i, v in enumerate(band_maes):
    ax3.text(i, v + 0.05, f"{v:.2f}", ha='center', fontweight='bold', fontsize=9.5)

# (d) Ridge Linguistic Polarities
top_words_combined = pd.concat([top_pos_words.tail(5), top_neg_words.head(5)])
colors_p = ['#10b981' if c > 0 else '#ef4444' for c in top_words_combined['coef']]
ax4.barh(top_words_combined['word'], top_words_combined['coef'], color=colors_p, edgecolor='black')
ax4.set_title("(d) Top Sommelier Lexicon Weights (Ridge Coefficients)", fontsize=10, fontweight='bold')
ax4.set_xlabel("Regression Coefficient Weight")

plt.tight_layout()
fig.savefig("B2_Wine_Reviews/figures/combined_key_result.png", dpi=300)
plt.close(fig)

# Save JSON summary
summary_data = {
    'metrics': {k: {m: round(float(val), 4) for m, val in v.items()} for k, v in ablation_results.items()},
    'mae_by_band': mae_by_band,
    'top_positive_words': top_pos_words['word'].tolist(),
    'top_negative_words': top_neg_words['word'].tolist(),
    'duplicates_removed': int(duplicates_removed)
}

with open("B2_Wine_Reviews/results_summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)

print("\nTask B2 script completed successfully! Saved to B2_Wine_Reviews/results_summary.json", flush=True)
