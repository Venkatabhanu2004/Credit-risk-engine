import os
import math
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from credit_risk.logger import logger

warnings.filterwarnings(action='ignore')

# Ensure visual artifacts directory exists
OUTPUT_DIR = "artifacts/visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. UNIVARIATE ANALYSIS - NUMERICAL (df.hist)
# ============================================================

def plot_univariate_numerical(df, save_filename='Univariate Analysis - Numerical Variables Distribution.png'):
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    logger.info("Generating Univariate Numerical Distributions")

    df_num = df.select_dtypes(include=['int64', 'float64'])
    df_num = df_num[[col for col in df_num.columns if 'id' not in col.lower()]]

    plt.figure(figsize=(15, 12))
    df_num.hist(figsize=(15, 12), bins=30, edgecolor='black', grid=False)
    plt.suptitle('Univariate Numerical Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

    print(f"Saved: {save_path}")
    logger.info(f"Saved numerical univariate chart to {save_path}")


# ============================================================
# 2. UNIVARIATE ANALYSIS - CATEGORICAL (Donut Composition)
# ============================================================

def plot_univariate_categorical(df, save_filename='Univariate Analysis - Categorical Variables Composition.png'):
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    logger.info("Generating Univariate Categorical Composition")

    df_cat = df.select_dtypes(include=['object', 'category'])
    cat_columns = [col for col in df_cat.columns if 'id' not in col.lower()]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(cat_columns[:9]):
        counts = df_cat[col].value_counts()
        
        axes[i].pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor='white')
        )
        axes[i].set_title(col, fontsize=12, fontweight='bold')

    for j in range(len(cat_columns[:9]), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Univariate Categorical Composition', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

    print(f"Saved: {save_path}")
    logger.info(f"Saved categorical univariate chart to {save_path}")


# ============================================================
# 3. BIVARIATE ANALYSIS - NUMERICAL VS TARGET (Boxplots)
# ============================================================

def plot_bivariate_numerical(df, target_col='Loan_Default', save_filename='BI-Variate Analysis - Numerical Variables vs Target Variable Distribution.png'):
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    logger.info(f"Generating Bivariate Boxplots against {target_col}")

    df_num = df.select_dtypes(include=['int64', 'float64'])
    num_cols = [c for c in df_num.columns if 'id' not in c.lower() and c != target_col]

    n_features = len(num_cols)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.8))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.boxplot(
            data=df,
            x=target_col,
            y=col,
            ax=ax,
            palette='Set2',
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "6"}
        )
        ax.set_title(f'{col} by {target_col}', fontsize=11, fontweight='bold')
        ax.set_xlabel(target_col, fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Bivariate Analysis: Numerical Features vs. {target_col}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

    print(f"Saved: {save_path}")
    logger.info(f"Saved numerical bivariate boxplots to {save_path}")


# ============================================================
# 4. BIVARIATE ANALYSIS - CATEGORICAL VS TARGET (Stacked Bar)
# ============================================================

def plot_bivariate_categorical(df, target_col='Loan_Default', save_filename='BI-Variate Analysis - Categorical Variables vs Target Variable Distribution.png'):
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    logger.info(f"Generating Bivariate Stacked Bars against {target_col}")

    df_cat = df.select_dtypes(include=['object', 'category'])
    cat_columns = [col for col in df_cat.columns if 'id' not in col.lower() and col != target_col]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(cat_columns[:9]):
        ax = axes[i]
        cross_tab_prop = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        
        cross_tab_prop.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            colormap='coolwarm',
            edgecolor='black',
            alpha=0.85
        )
        
        ax.set_title(f'{col} vs. {target_col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=20)
        ax.legend(title=target_col, loc='upper right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    for j in range(len(cat_columns[:9]), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Bivariate Analysis: Categorical Features vs. {target_col} (Proportions)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

    print(f"Saved: {save_path}")
    logger.info(f"Saved categorical bivariate stacked bars to {save_path}")


# ============================================================
# 5. MASTER EXECUTION FUNCTION
# ============================================================

def run_all_eda(df, target_col='Loan_Default'):
    """
    Executes all 4 EDA steps and writes charts to artifacts/visualisations/
    """
    logger.info("Starting Full EDA Execution")
    plot_univariate_numerical(df)
    plot_univariate_categorical(df)
    plot_bivariate_numerical(df, target_col=target_col)
    plot_bivariate_categorical(df, target_col=target_col)
    logger.info("Full EDA Execution Completed Successfully")