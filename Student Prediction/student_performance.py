# Student Performance Prediction Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("StudentsPerformance.csv")
    print("Dataset loaded successfully!\n")
except FileNotFoundError:
    print("Error: 'StudentsPerformance.csv' not found.")
    print("Make sure the CSV file is in the same folder.")
    exit()

# -----------------------------
# Basic Information
# -----------------------------
print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Categorical Columns
# -----------------------------
categorical_cols = df.select_dtypes(include='object').columns

print("\nCategorical Columns:")
print(categorical_cols)

for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())

# -----------------------------
# Data Visualization
# -----------------------------

# Gender count plot
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=df)
plt.title("Gender Distribution")
plt.show()

# Test preparation course pie chart
count_test = df['test preparation course'].value_counts()
labels = count_test.index

plt.figure(figsize=(6, 6))
plt.pie(count_test, labels=labels, autopct='%1.1f%%')
plt.title("Test Preparation Course")
plt.legend(labels)
plt.show()

# -----------------------------
# Feature Engineering
# -----------------------------
# Create average score
df['average_score'] = (
    df['math score'] +
    df['reading score'] +
    df['writing score']
) / 3

print("\nAverage score column added!")

# Scatter plots
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x='average_score',
    y='math score',
    hue='gender',
    data=df
)
plt.title("Average Score vs Math Score")
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x='average_score',
    y='reading score',
    hue='gender',
    data=df
)
plt.title("Average Score vs Reading Score")
plt.show()

# -----------------------------
# Encoding Categorical Data
# -----------------------------
gender = {
    'male': 1,
    'female': 0
}

race = {
    'group A': 0,
    'group B': 1,
    'group C': 2,
    'group D': 3,
    'group E': 4
}

education = {
    "bachelor's degree": 0,
    'some college': 1,
    "master's degree": 2,
    "associate's degree": 3,
    "high school": 4,
    "some high school": 5
}

df['gender'] = df['gender'].map(gender)
df['race/ethnicity'] = df['race/ethnicity'].map(race)
df['parental level of education'] = (
    df['parental level of education'].map(education)
)

# Convert remaining categorical columns
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# Train-Test Split
# -----------------------------
X = df.drop(columns='average_score')
y = df['average_score']

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=0
)

# -----------------------------
# Model Training
# -----------------------------
model = RandomForestRegressor(random_state=0)

print("\nTraining model...")
model.fit(x_train, y_train)

# Predictions
predictions = model.predict(x_test)

# -----------------------------
# Model Evaluation
# -----------------------------
score = r2_score(y_test, predictions)

print("\nModel Accuracy (R2 Score):")
print(score)

# -----------------------------
# Save Model
# -----------------------------
pickle.dump(model, open('student_model.pkl', 'wb'))

print("\nModel saved as 'student_model.pkl'")
print("Project completed successfully!")