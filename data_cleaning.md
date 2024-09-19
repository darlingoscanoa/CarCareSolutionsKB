# Data Cleaning

Data cleaning is a crucial step in preparing our app usage data for analysis. It involves identifying and correcting errors, handling missing values, and dealing with outliers to ensure the quality and reliability of our data.

## Importance of Data Cleaning

Clean data is essential for:
- Accurate analysis and insights
- Reliable machine learning models
- Informed decision-making

# Data Cleaning with Iris Dataset

Data cleaning is a crucial step in preparing data for analysis. In this guide, we'll use the famous Iris dataset to demonstrate key data cleaning techniques in Google Colab.

## Setting Up the Environment

First, let's set up our Google Colab environment and import the necessary libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                  columns=iris['feature_names'] + ['target'])

# Display the first few rows
print(df.head())

![image](https://github.com/user-attachments/assets/3896c504-0716-4c2a-ada3-67d186635abc)

# Key Data Cleaning Techniques

## 1. Handling Missing Data
Although the Iris dataset is clean, let's simulate some missing data to demonstrate cleaning techniques:

# Introduce some missing values
df.loc[np.random.choice(df.index, 10), 'sepal length (cm)'] = np.nan

# Check for missing values
print(df.isnull().sum())

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

![image](https://github.com/user-attachments/assets/790183bb-e86c-4649-a66a-77f69b7e09fa)

