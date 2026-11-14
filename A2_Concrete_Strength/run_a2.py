"""
Task A2: Predicting Concrete Compressive Strength
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

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ucimlrepo import fetch_ucirepo

def main():
    # Set styles and random seed
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    sns.set_palette("deep")

    os.makedirs("A2_Concrete_Strength/figures", exist_ok=True)
    os.makedirs("A2_Concrete_Strength/data", exist_ok=True)

    print("=" * 60, flush=True)
    print("1. INGESTING DATASET (UCI ID 165)", flush=True)
    print("=" * 60, flush=True)
    concrete = fetch_ucirepo(id=165)
    X_raw = concrete.data.features.copy()
    y_raw = concrete.data.targets.copy()

    # Standardize column names for clean code
    X_raw.columns = [
        'cement', 'blast_furnace_slag', 'fly_ash', 'water',
        'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age'
    ]
    y_raw.columns = ['strength']
    df = pd.concat([X_raw, y_raw], axis=1)

    print(f"Total samples: {len(df)}", flush=True)
    print(f"Features: {list(X_raw.columns)}", flush=True)
    print(f"Missing values: {df.isnull().sum().sum()}", flush=True)
    print(df.describe().round(2), flush=True)

    # ==============================================================================
    # 2. EXPLORATORY DATA ANALYSIS & VISUALIZATION
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("2. COMPUTING EDA & SAVING FIGURES", flush=True)
    print("=" * 60, flush=True)

    # Figure 1A: Feature Histograms & Age Skewness
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#1f77b4')
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("")
    plt.tight_layout()
    fig.savefig("A2_Concrete_Strength/figures/eda_histograms.png", dpi=300)
    plt.close(fig)

    # Figure 1B: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("Correlation Matrix of Ingredients, Age, and Strength", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig("A2_Concrete_Strength/figures/eda_correlation_matrix.png", dpi=300)
    plt.close(fig)

    # Figure 1C: Scatters - Strength vs Age & Strength vs Water/Cement Ratio
    df['water_cement_ratio'] = df['water'] / df['cement']
    df['log_age'] = np.log(df['age'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df, x='age', y='strength', hue='cement', palette='viridis', alpha=0.7, ax=ax1)
    ax1.set_title("Compressive Strength vs. Curing Age (Days)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Age (days)")
    ax1.set_ylabel("Strength (MPa)")

    sns.scatterplot(data=df, x='water_cement_ratio', y='strength', hue='age', palette='magma', alpha=0.7, ax=ax2)
    ax2.set_title("Compressive Strength vs. Water/Cement Ratio (Abrams' Law)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Water / Cement Ratio")
    ax2.set_ylabel("Strength (MPa)")
    plt.tight_layout()
    fig.savefig("A2_Concrete_Strength/figures/eda_scatter_age_wc.png", dpi=300)
    plt.close(fig)

    # ==============================================================================
    # 3. DATA SPLITTING & ZERO LEAKAGE SETUP
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("3. SPLITTING DATA (80% Train, 20% Test) & PIPELINE PREPARATION", flush=True)
    print("=" * 60, flush=True)

    X_raw_feat = df[['cement', 'blast_furnace_slag', 'fly_ash', 'water',
                     'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']]
    X_eng_feat = df[['cement', 'blast_furnace_slag', 'fly_ash', 'water',
                     'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age',
                     'water_cement_ratio', 'log_age']]
    y = df['strength']

    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw_feat, y, test_size=0.20, random_state=RANDOM_SEED
    )
    X_tr_eng, X_te_eng, _, _ = train_test_split(
        X_eng_feat, y, test_size=0.20, random_state=RANDOM_SEED
    )

    print(f"Training instances: {len(X_tr_raw)} ({len(X_tr_raw)/len(df)*100:.1f}%)", flush=True)
    print(f"Testing instances:  {len(X_te_raw)} ({len(X_te_raw)/len(df)*100:.1f}%)", flush=True)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # ==============================================================================
    # 4. MODEL TRAINING, TUNING & ABLATION EXPERIMENTS
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("4. RUNNING MODEL SUITE & 5-FOLD CV HYPERPARAMETER TUNING", flush=True)
    print("=" * 60, flush=True)

    results = []

    def evaluate_model(name, model, X_train, X_test, feature_set_name, best_params="N/A"):
        cv_scores = np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=1))
        cv_rmse_mean = cv_scores.mean()
        cv_rmse_std = cv_scores.std()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        res = {
            'model': name,
            'features': feature_set_name,
            'params': str(best_params),
            'cv_rmse_mean': cv_rmse_mean,
            'cv_rmse_std': cv_rmse_std,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'model_obj': model,
            'y_pred': y_pred
        }
        results.append(res)
        print(f"[{feature_set_name:10s}] {name:28s} | CV RMSE: {cv_rmse_mean:6.3f} +/- {cv_rmse_std:5.3f} | Test MAE: {test_mae:6.3f} | Test RMSE: {test_rmse:6.3f} | Test R2: {test_r2:6.3f}", flush=True)
        return res

    # 1. Baseline: Linear Regression
    pipe_lr_raw = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    evaluate_model("Linear Regression", pipe_lr_raw, X_tr_raw, X_te_raw, "Raw (8)")

    pipe_lr_eng = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    evaluate_model("Linear Regression", pipe_lr_eng, X_tr_eng, X_te_eng, "Engineered")

    # 2. Polynomial + Ridge Regression
    param_grid_poly = {
        'poly__degree': [2],
        'ridge__alpha': [0.1, 1.0, 10.0, 100.0]
    }

    grid_poly_raw = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(include_bias=False)), ('ridge', Ridge(random_state=RANDOM_SEED))]),
        param_grid_poly, cv=cv, scoring='neg_mean_squared_error', n_jobs=1
    )
    grid_poly_raw.fit(X_tr_raw, y_train)
    evaluate_model("Polynomial + Ridge", grid_poly_raw.best_estimator_, X_tr_raw, X_te_raw, "Raw (8)", grid_poly_raw.best_params_)

    grid_poly_eng = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(include_bias=False)), ('ridge', Ridge(random_state=RANDOM_SEED))]),
        param_grid_poly, cv=cv, scoring='neg_mean_squared_error', n_jobs=1
    )
    grid_poly_eng.fit(X_tr_eng, y_train)
    evaluate_model("Polynomial + Ridge", grid_poly_eng.best_estimator_, X_tr_eng, X_te_eng, "Engineered", grid_poly_eng.best_params_)

    # 3. Random Forest Regressor
    param_grid_rf = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, 15],
        'rf__min_samples_leaf': [1, 2]
    }

    grid_rf_raw = GridSearchCV(
        Pipeline([('rf', RandomForestRegressor(random_state=RANDOM_SEED))]),
        param_grid_rf, cv=cv, scoring='neg_mean_squared_error', n_jobs=1
    )
    grid_rf_raw.fit(X_tr_raw, y_train)
    evaluate_model("Random Forest", grid_rf_raw.best_estimator_, X_tr_raw, X_te_raw, "Raw (8)", grid_rf_raw.best_params_)

    grid_rf_eng = GridSearchCV(
        Pipeline([('rf', RandomForestRegressor(random_state=RANDOM_SEED))]),
        param_grid_rf, cv=cv, scoring='neg_mean_squared_error', n_jobs=1
    )
    grid_rf_eng.fit(X_tr_eng, y_train)
    evaluate_model("Random Forest", grid_rf_eng.best_estimator_, X_tr_eng, X_te_eng, "Engineered", grid_rf_eng.best_params_)

    # 4. Gradient Boosting Regressor
    param_grid_gbt = {
        'gbt__n_estimators': [100, 200],
        'gbt__learning_rate': [0.05, 0.1],
        'gbt__max_depth': [3, 5]
    }

    grid_gbt_eng = GridSearchCV(
        Pipeline([('gbt', GradientBoostingRegressor(random_state=RANDOM_SEED))]),
        param_grid_gbt, cv=cv, scoring='neg_mean_squared_error', n_jobs=1
    )
    grid_gbt_eng.fit(X_tr_eng, y_train)
    champion_gbt = grid_gbt_eng.best_estimator_
    evaluate_model("Gradient Boosting", champion_gbt, X_tr_eng, X_te_eng, "Engineered", grid_gbt_eng.best_params_)

    # 5. MLPRegressor
    pipe_mlp_eng = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, alpha=0.01, random_state=RANDOM_SEED))
    ])
    evaluate_model("MLPRegressor", pipe_mlp_eng, X_tr_eng, X_te_eng, "Engineered", "{hidden: (64, 32), alpha: 0.01}")

    # ==============================================================================
    # 5. DIAGNOSTIC PLOTS & ERROR RESIDUAL ANALYSIS
    # ==============================================================================
    print("\n" + "=" * 60, flush=True)
    print("5. GENERATING TEST DIAGNOSTIC PLOTS", flush=True)
    print("=" * 60, flush=True)

    champion_res = sorted([r for r in results if r['features'] == 'Engineered'], key=lambda x: x['test_rmse'])[0]
    print(f"Champion Model: {champion_res['model']} with Test RMSE = {champion_res['test_rmse']:.3f} MPa and R2 = {champion_res['test_r2']:.3f}", flush=True)

    y_pred_champ = champion_res['y_pred']
    residuals = y_test - y_pred_champ

    # Figure 2A: Predicted vs Actual
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred_champ, alpha=0.7, edgecolors='k', linewidth=0.5, color='#2ca02c')
    min_val = min(y_test.min(), y_pred_champ.min()) - 2
    max_val = max(y_test.max(), y_pred_champ.max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (1:1)')
    ax.set_title(f"Predicted vs. Actual Strength ({champion_res['model']})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Actual Strength (MPa)", fontsize=10)
    ax.set_ylabel("Predicted Strength (MPa)", fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig("A2_Concrete_Strength/figures/predicted_vs_actual.png", dpi=300)
    plt.close(fig)

    # Figure 2B: Residual Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred_champ, residuals, alpha=0.7, edgecolors='k', linewidth=0.5, color='#d62728')
    ax.axhline(0, color='black', linestyle='--', lw=2)
    ax.set_title(f"Residual Plot ({champion_res['model']})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Strength (MPa)", fontsize=10)
    ax.set_ylabel("Residual (Actual - Predicted) MPa", fontsize=10)
    plt.tight_layout()
    fig.savefig("A2_Concrete_Strength/figures/residual_plot.png", dpi=300)
    plt.close(fig)

    # Figure 2C: Combined Figure for IEEE Report Figure 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(y_test, y_pred_champ, alpha=0.7, edgecolors='k', linewidth=0.5, color='#1f77b4')
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity line')
    ax1.set_title(f"Predicted vs. Actual ({champion_res['model']})", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Actual Compressive Strength (MPa)")
    ax1.set_ylabel("Predicted Compressive Strength (MPa)")
    ax1.legend()

    ax2.scatter(y_pred_champ, residuals, alpha=0.7, edgecolors='k', linewidth=0.5, color='#ff7f0e')
    ax2.axhline(0, color='black', linestyle='--', lw=2)
    ax2.set_title(f"Residuals vs. Fitted Values", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Predicted Compressive Strength (MPa)")
    ax2.set_ylabel("Residual (MPa)")
    plt.tight_layout()
    fig.savefig("A2_Concrete_Strength/figures/combined_key_result.png", dpi=300)
    plt.close(fig)

    # Figure 3: Feature Importances
    importances = None
    if 'gbt' in champion_res['model_obj'].named_steps:
        importances = champion_res['model_obj'].named_steps['gbt'].feature_importances_
    elif 'rf' in champion_res['model_obj'].named_steps:
        importances = champion_res['model_obj'].named_steps['rf'].feature_importances_

    if importances is not None:
        feat_series = pd.Series(importances, index=X_eng_feat.columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        feat_series.plot(kind='barh', color='#38bdf8', ax=ax, edgecolor='black')
        ax.set_title("Gini Feature Importances (Engineered Feature Set)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Relative Importance")
        plt.tight_layout()
        fig.savefig("A2_Concrete_Strength/figures/feature_importances.png", dpi=300)
        plt.close(fig)

    # Save summary
    summary_records = []
    for r in results:
        summary_records.append({
            'Model': r['model'],
            'Features': r['features'],
            'Hyperparameters': r['params'],
            'CV_RMSE_Mean': round(r['cv_rmse_mean'], 3),
            'CV_RMSE_Std': round(r['cv_rmse_std'], 3),
            'Test_MAE': round(r['test_mae'], 3),
            'Test_RMSE': round(r['test_rmse'], 3),
            'Test_R2': round(r['test_r2'], 3)
        })

    with open("A2_Concrete_Strength/results_summary.json", "w") as f:
        json.dump(summary_records, f, indent=2)

    print("\nTask A2 processing successfully finished! Exact metrics saved to A2_Concrete_Strength/results_summary.json", flush=True)

if __name__ == '__main__':
    main()
