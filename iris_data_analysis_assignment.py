# iris_data_analysis_assignment.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display first few rows
print(df.head())

# Data types and missing values
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Basic statistics
print(df.describe())

# Grouping by species
grouped = df.groupby("species").mean()
print("\nGroup-wise mean:\n", grouped)

# --- Visualizations ---
sns.set(style="whitegrid")

# Bar chart - average petal length by species
df.groupby('species')['petal length (cm)'].mean().plot(kind='bar')
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram - sepal width distribution
df['sepal width (cm)'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter plot - sepal length vs. petal length
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal Length vs Petal Length")
plt.tight_layout()
plt.show()

# Line chart - simulated trend (not real time series)
df[['sepal length (cm)']].plot(kind='line', title='Simulated Sepal Length Trend')
plt.ylabel("Sepal Length (cm)")
plt.tight_layout()
plt.show()
