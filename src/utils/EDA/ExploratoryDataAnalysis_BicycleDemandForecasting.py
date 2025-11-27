"""
EDA Script for Bicycle Demand Forecasting
Author: AI Assistant
Date: 2025-11-27
Description: Exploratory Data Analysis for merged_with_weather.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set default matplotlib style
plt.style.use('default')
plt.rc('axes', unicode_minus=False)

# Project root directory
script_dir = Path(__file__).resolve().parent  # EDA directory
project_root = script_dir.parent.parent.parent  # PROJECT-Bicyle-Demand-Forecasting directory

# Load data (using absolute path)
data_path = project_root / 'Data' / 'processed' / 'join' / 'merged_with_weather.csv'

# Check if file exists
if not data_path.exists():
    print(f"Error: File not found: {data_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    raise FileNotFoundError(f"Data file does not exist: {data_path}")

df = pd.read_csv(data_path)

print("=" * 80)
print("1. Basic Data Information")
print("=" * 80)
print(f"\nData shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
print("\n" + "=" * 80)
print("2. Column Information")
print("=" * 80)
print(df.info())

print("\n" + "=" * 80)
print("3. Descriptive Statistics")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("4. Missing Values Check")
print("=" * 80)
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage (%)': missing_percentage
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

if missing_df['Missing Count'].sum() == 0:
    print("No missing values!")

print("\n" + "=" * 80)
print("5. Column Classification by Data Type")
print("=" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"\nDatetime columns ({len(datetime_cols)}): {datetime_cols}")

print("\n" + "=" * 80)
print("6. Duplicate Data Check")
print("=" * 80)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates:,}")
if duplicates > 0:
    print(f"Duplicate ratio: {(duplicates / len(df)) * 100:.2f}%")

print("\n" + "=" * 80)
print("7. Sample Data")
print("=" * 80)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

# Visualization
print("\n" + "=" * 80)
print("8. Generating Visualizations...")
print("=" * 80)

# Output directory (based on project root)
output_dir = project_root / 'notebooks'
output_dir.mkdir(exist_ok=True)

# 8-1. Numeric variable distributions
if len(numeric_cols) > 0:
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eda_numeric_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Numeric distributions saved: {output_dir / 'eda_numeric_distributions.png'}")
    plt.close()

# 8-2. Correlation heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=1)
    plt.title('Correlation Heatmap of Numeric Variables', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Correlation heatmap saved: {output_dir / 'eda_correlation_heatmap.png'}")
    plt.close()

# 8-3. Categorical variable distributions
if len(categorical_cols) > 0:
    n_cols = min(3, len(categorical_cols))
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for idx, col in enumerate(categorical_cols):
        if idx < len(axes):
            value_counts = df[col].value_counts()
            axes[idx].bar(range(len(value_counts)), value_counts.values, alpha=0.7)
            axes[idx].set_xticks(range(len(value_counts)))
            axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Remove empty subplots
    for idx in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eda_categorical_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Categorical distributions saved: {output_dir / 'eda_categorical_distributions.png'}")
    plt.close()

# 8-4. Box plots (outlier detection)
if len(numeric_cols) > 0:
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            axes[idx].boxplot(df[col].dropna(), vert=True)
            axes[idx].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(col)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Remove empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eda_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"✓ Boxplots saved: {output_dir / 'eda_boxplots.png'}")
    plt.close()

print("\n" + "=" * 80)
print("9. EDA Complete!")
print("=" * 80)
print(f"All visualization files have been saved to '{output_dir}' folder.")
print("\nGenerated files:")
print("  - eda_numeric_distributions.png: Numeric variable distributions")
print("  - eda_correlation_heatmap.png: Correlation heatmap")
print("  - eda_categorical_distributions.png: Categorical variable distributions")
print("  - eda_boxplots.png: Boxplots (outlier detection)")
