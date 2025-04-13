import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
import pymysql
import os

# Print environment variable to verify
print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")

# Create SQLAlchemy engine using the Cloud SQL Proxy
try:
    # Connect through the proxy running on localhost:3307
    engine = create_engine(
        "mysql+pymysql://root:0000@127.0.0.1:3307/aise3010finalproject-db"
    )
    print("Engine created successfully!")
except Exception as e:
    print(f"Error creating SQLAlchemy engine: {e}")
    exit(1)

# Query to join tables
query = """
SELECT 
    d.m_id, d.Type, d.sex, d.age,
    b.fibros, b.activity,
    i.got, i.gpt, i.alb, i.tbil, i.dbil, i.che, i.ttt, i.ztt, i.tcho, i.tp,
    inf.dur
FROM dispat d
LEFT JOIN rel11 r11 ON d.m_id = r11.m_id
LEFT JOIN Bio b ON r11.b_id = b.b_id
LEFT JOIN rel12 r12 ON d.m_id = r12.m_id
LEFT JOIN indis i ON r12.in_id = i.in_id
LEFT JOIN rel13 r13 ON d.m_id = r13.m_id
LEFT JOIN inf inf ON r13.a_id = inf.a_id
"""

# Load data into a pandas DataFrame
try:
    df = pd.read_sql(query, engine)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error executing query: {e}")
    engine.dispose()
    exit(1)
finally:
    engine.dispose()  # Ensure the engine is closed

# --- Exploratory Data Analysis ---

# 1. Summary Statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# 2. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Distribution of the target variable (dispat.Type)
plt.figure(figsize=(8, 6))
sns.countplot(x='Type', data=df)
plt.title('Distribution of Target Variable (Type)')
plt.xlabel('Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('target_distribution.png')  # Save figure to file
plt.close()

# 4. Numerical feature distributions
numerical_cols = ['age', 'fibros', 'activity', 'got', 'gpt', 'alb', 'tbil', 
                  'dbil', 'che', 'ttt', 'ztt', 'tcho', 'tp', 'dur']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('numerical_distributions.png')  # Save figure to file
plt.close()

# 5. Box plots of numerical features vs. Type
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(x='Type', y=col, data=df)
    plt.title(f'{col} vs Type')
    plt.xlabel('Type')
    plt.ylabel(col)
plt.tight_layout()
plt.savefig('boxplots.png')  # Save figure to file
plt.close()

# 6. Correlation heatmap for numerical features
plt.figure(figsize=(12, 8))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')  # Save figure to file
plt.close()

# 7. Categorical feature (sex) vs. Type
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', hue='Type', data=df)
plt.title('Sex vs Type')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Type')
plt.tight_layout()
plt.savefig('sex_vs_type.png')  # Save figure to file
plt.close()

# 8. Pairplot for a subset of important features
subset_cols = ['age', 'fibros', 'got', 'gpt', 'Type']
sns.pairplot(df[subset_cols], hue='Type', diag_kind='kde')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.savefig('pairplot.png')  # Save figure to file
plt.close()

print("All visualizations completed and saved as PNG files!")