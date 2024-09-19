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

```
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
```

![image](https://github.com/user-attachments/assets/3896c504-0716-4c2a-ada3-67d186635abc)

# Key Data Cleaning Techniques

## 1. Handling Missing Data
Handling missing data is crucial in data analysis and machine learning. Missing values can lead to biased results, reduced statistical power, and incorrect conclusions. It's essential to address this issue as ignoring it can significantly impact the accuracy of your models. While handling missing data can be time-consuming, it's a necessary step that can prevent costly errors in decision-making processes based on the analysis results.

Although the Iris dataset is clean, let's simulate some missing data to demonstrate cleaning techniques:

## a) Identifying Missing Data
Identifying missing data is the first step in handling it. This process involves checking for null or NaN values in your dataset. Visualizing missing data can help you understand the pattern of missingness, which can inform your strategy for dealing with it.

```
# Introduce some missing values
df.loc[np.random.choice(df.index, 10), 'sepal length (cm)'] = np.nan

# Check for missing values
print(df.isnull().sum())

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()
```

![image](https://github.com/user-attachments/assets/790183bb-e86c-4649-a66a-77f69b7e09fa)

## b) Dealing with Missing Data
Once missing data is identified, there are several strategies to deal with it. Common approaches include dropping rows with missing values or imputing missing values with measures like mean, median, or mode. The choice of method depends on the nature of your data and the specific requirements of your analysis.

```
# Drop rows with missing values
df_cleaned = df.dropna()

# Fill missing values with mean
df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean(), inplace=True)

print("After cleaning:")
print(df.isnull().sum())
```
![image](https://github.com/user-attachments/assets/ce810d1f-9fdb-4588-a1c9-c801549e12c7)

## 2. Handling Outliers
Outlier detection and treatment are important steps in data cleaning. Outliers can significantly skew statistical analyses and machine learning models, leading to inaccurate results. While some outliers may be genuine extreme values, others could be due to data entry errors or measurement issues. Properly handling outliers can improve model performance and the reliability of your analyses. However, be cautious not to remove legitimate extreme values that might be important for your specific use case.

## a) Identifying Outliers
Identifying outliers involves using statistical methods or visualization techniques to spot data points that are significantly different from other observations. Box plots and z-scores are common methods for outlier detection.

```
# Using box plots to identify outliers
plt.figure(figsize=(12, 6))
df.boxplot(column=iris['feature_names'])
plt.title('Box Plot to Identify Outliers')
plt.xticks(rotation=45)
plt.show()

# Using Z-score to identify outliers
from scipy import stats

z_scores = np.abs(stats.zscore(df[iris['feature_names']]))
print("Potential outliers (Z-score > 3):")
print(df[(z_scores > 3).any(axis=1)])
```
![image](https://github.com/user-attachments/assets/cfd7d3a2-6fc5-4b3c-9fdb-9d32e86f78b6)

## b) Dealing with Outliers
Once outliers are identified, you need to decide how to handle them. Common approaches include removing outliers, capping them at a certain value (winsorization), or transforming the data. The choice depends on the nature of your data and the requirements of your analysis.

```
# Example: Removing outliers using IQR for 'petal length (cm)'
Q1 = df['petal length (cm)'].quantile(0.25)
Q3 = df['petal length (cm)'].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[(df['petal length (cm)'] >= Q1 - 1.5*IQR) & 
                    (df['petal length (cm)'] <= Q3 + 1.5*IQR)]

print(f"Shape before removing outliers: {df.shape}")
print(f"Shape after removing outliers: {df_no_outliers.shape}")
```
![image](https://github.com/user-attachments/assets/f4cc8e56-ea95-4c35-9b19-d2f1f58308bb)

## 3. Data Validation
Data validation is a critical step in ensuring the integrity and consistency of your dataset. It helps identify and correct errors, inconsistencies, and inaccuracies in the data. While it can be time-consuming, especially for large datasets, it's an essential process that can prevent serious issues down the line. Proper data validation can save time and resources by catching errors early and improving the overall quality of your analyses and models.

Ensure data consistency and accuracy:

```
# Check for duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Validate data types
print("\nData types:")
print(df.dtypes)

# Check value ranges
print("\nValue ranges:")
for column in iris['feature_names']:
    print(f"{column}: {df[column].min()} to {df[column].max()}")

# Check target distribution
print("\nTarget distribution:")
print(df['target'].value_counts())
```

![image](https://github.com/user-attachments/assets/95b8b947-d104-47c1-b4d7-5ece4be80416)

## 4. Feature Correlation
Analyzing feature correlation is an important step in understanding the relationships between variables in your dataset. It can help identify redundant features, which can be useful for feature selection or dimensionality reduction. Correlation analysis can also reveal potential issues like multicollinearity, which can affect the performance of certain machine learning models. Visualizing correlations can provide insights that might not be apparent from raw data alone.

Visualize correlations between features:

```
# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[iris['feature_names']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
```

![image](https://github.com/user-attachments/assets/facda1d9-f4ba-413a-ae29-01e5096099c0)

## Conclusion

By applying these data cleaning techniques to the Iris dataset, we've demonstrated how to handle missing data, identify and deal with outliers, validate data, and visualize correlations. These skills are crucial for preparing data for further analysis and machine learning tasks.
Remember that data cleaning is an iterative process, and the specific techniques used may vary depending on the dataset and the goals of your analysis. References:

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179-188.

McKinney, W. (2017). Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython (2nd ed.). O'Reilly Media.

Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (2nd ed.). O'Reilly Media.
