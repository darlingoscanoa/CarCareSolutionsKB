# Data Preprocessing with Iris Dataset

Data preprocessing is a crucial step in preparing data for machine learning models. This guide demonstrates key preprocessing techniques using the Iris dataset in Google Colab.

## Setting Up the Environment

First, let's set up our Google Colab environment and import the necessary libraries:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                  columns=iris['feature_names'] + ['target'])

print(df.head())
```
![image](https://github.com/user-attachments/assets/e08930ea-d9e9-4944-b5d8-81b72c359ee0)

# Key Data Preprocessing Techniques

## 1. Data Normalization and Standardization
Normalization and standardization help to scale features to a common range.

Normalization and standardization are essential preprocessing techniques that adjust the scale of features to ensure that they contribute equally to the analysis. Normalization rescales features to a range of [0, 1], which is particularly useful for algorithms that rely on distance calculations, such as k-nearest neighbors. Standardization transforms features to have a mean of 0 and a standard deviation of 1, making them suitable for algorithms that assume normally distributed data, like linear regression. By applying these techniques, we can improve model performance and convergence speed.

## Min-Max Normalization

```
# Apply Min-Max scaling
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[iris['feature_names']]), 
                             columns=iris['feature_names'])

# Visualize before and after
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
df[iris['feature_names']].boxplot(ax=ax1)
ax1.set_title('Before Normalization')
df_normalized.boxplot(ax=ax2)
ax2.set_title('After Min-Max Normalization')
plt.show()
```
![image](https://github.com/user-attachments/assets/542f6aa2-bea3-4b07-9650-566bc19d2689)

## Z-score Standardization
Z-score standardization, also known as standard scaling, is a crucial preprocessing technique that transforms data to have a mean of 0 and a standard deviation of 1. This method is particularly useful for comparing variables with different units or scales, identifying outliers, and preparing data for machine learning algorithms that assume normally distributed features. Z-scores also facilitate the interpretation of how many standard deviations a data point is from the mean, making it valuable for statistical analysis and ensuring that all features contribute equally to model training.


```
# Apply Z-score standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df[iris['feature_names']]), 
                               columns=iris['feature_names'])

# Visualize standardized data
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_standardized)
plt.title('Features After Z-score Standardization')
plt.show()
```

![image](https://github.com/user-attachments/assets/207e03d0-e479-474c-aeff-4581e7a2056b)

## 2. Handling Categorical Data
Handling categorical data is vital in machine learning because many algorithms require numerical input. Categorical variables can represent different classes or groups and must be transformed into a format that algorithms can interpret. One-hot encoding is a common technique that converts categorical variables into a binary format, allowing models to utilize this information effectively without introducing ordinal relationships that do not exist. Properly encoding categorical data enhances model accuracy and performance.

The Iris dataset doesn't have categorical features, but let's demonstrate encoding:

```
# Create a categorical column for demonstration
df['size'] = pd.cut(df['petal length (cm)'], bins=3, labels=['small', 'medium', 'large'])

# One-hot encoding
encoder = OneHotEncoder(sparse=False)
size_encoded = encoder.fit_transform(df[['size']])
size_encoded_df = pd.DataFrame(size_encoded, columns=encoder.get_feature_names(['size']))

print(size_encoded_df.head())
```
![image](https://github.com/user-attachments/assets/a7a53b82-49da-4eb2-a681-b8c3c0b90170)

## 3. Feature Selection
Feature selection is an important step in preprocessing that helps improve model performance by retaining only the most relevant features while removing redundant or irrelevant ones. This process reduces overfitting, decreases training time, and enhances model interpretability. By selecting the best features based on statistical tests or model-based methods, we can ensure that our model focuses on the most informative aspects of the data.

Select the most important features for our model:

```
# Separate features and target
X = df[iris['feature_names']]
y = df['target']

# Apply SelectKBest
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()

print("Selected features:", selected_features)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=iris['feature_names'], y=selector.scores_)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('F-score')
plt.show()
```
![image](https://github.com/user-attachments/assets/dc0308f3-0728-4b23-b2a1-4abbfd33f0dd)

## 4. Handling Imbalanced Data
Handling imbalanced data is crucial for building robust machine learning models. When one class significantly outnumbers another in classification tasks, it can lead to biased predictions where the model favors the majority class. Techniques such as resampling or using different evaluation metrics help ensure that all classes are adequately represented during training. Addressing class imbalance improves the model's ability to generalize and perform well across all classes.

The Iris dataset is balanced, but let's demonstrate how to check for imbalance:

```
# Check class distribution
class_distribution = df['target'].value_counts()
print("Class distribution:\n", class_distribution)

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Class Distribution in Iris Dataset')
plt.show()
```

![image](https://github.com/user-attachments/assets/e83fe8f4-b5d7-4c4c-8b7e-b427a21e59be)

## 5. Train-Test Split
The train-test split is a fundamental step in validating machine learning models. By dividing the dataset into training and testing subsets, we can train our model on one portion of the data while evaluating its performance on unseen data. This practice helps prevent overfitting and ensures that our model generalizes well to new inputs. A common split ratio is 80/20 or 70/30 for training and testing sets.

Prepare data for model training:

```
# Split the data
X = df[selected_features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
```

![image](https://github.com/user-attachments/assets/6a0fa22b-50ab-43b0-9757-472bc893869a)

## 6. Handling Missing Values
Handling missing values is critical in data preprocessing because missing data can lead to biased estimates and reduced statistical power if not addressed properly. Imputation techniques replace missing values with meaningful substitutes (e.g., mean or mode), allowing us to retain more data points for analysis without discarding entire rows or columns. This process enhances the quality of our dataset and improves the reliability of our machine learning models.

The Iris dataset doesn't have missing values, but let's demonstrate imputation for both numeric and categorical data:

```
# Introduce some missing values for demonstration
df_missing = df.copy()
df_missing.loc[np.random.choice(df.index, 10), 'sepal length (cm)'] = np.nan
df_missing.loc[np.random.choice(df.index, 10), 'size'] = np.nan

# Separate numeric and categorical columns
numeric_columns = df_missing.select_dtypes(include=[np.number]).columns
categorical_columns = df_missing.select_dtypes(exclude=[np.number]).columns

# Impute missing values for numeric data
numeric_imputer = SimpleImputer(strategy='mean')
df_imputed_numeric = pd.DataFrame(numeric_imputer.fit_transform(df_missing[numeric_columns]), 
                                  columns=numeric_columns)

# Impute missing values for categorical data
categorical_imputer = SimpleImputer(strategy='most_frequent')
df_imputed_categorical = pd.DataFrame(categorical_imputer.fit_transform(df_missing[categorical_columns]), 
                                      columns=categorical_columns)

# Combine imputed dataframes
df_imputed = pd.concat([df_imputed_numeric, df_imputed_categorical], axis=1)

print("Before imputation:")
print(df_missing.isnull().sum())
print("\nAfter imputation:")
print(df_imputed.isnull().sum())
```
![image](https://github.com/user-attachments/assets/07bcbabd-858a-44e5-82c9-9ea4e8571cc3)


# Conclusion

These preprocessing techniques are essential for preparing the Iris dataset (or any dataset) for machine learning models. We've covered normalization, standardization, handling categorical data, feature selection, checking for class imbalance, splitting the data into training and testing sets, and handling missing values in both numeric and categorical data.
Remember that the specific preprocessing steps may vary depending on your dataset and the requirements of your chosen machine learning algorithm.

# References

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (2nd ed.). O'Reilly Media.

Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling. Springer.

scikit-learn developers. (2023). scikit-learn: Machine Learning in Python. scikit-learn.org. https://scikit-learn.org/stable/



