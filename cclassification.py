import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Step 1 — shape and dtypes of df_mh
df_mh = pd.read_csv('Student Mental health.csv')

# Rename columns for clarity
df_mh.columns = ['timestamp','age','gender','course','year','cgpa',
                  'married','depression','anxiety','panic_attack','treatment']

# Drop timestamp
df_mh.drop('timestamp', axis=1, inplace=True)

# Clean CGPA ranges like '3.50 - 4.00' → take midpoint
def parse_cgpa(x):
    try:
        parts = str(x).split(' - ')
        return (float(parts[0]) + float(parts[1])) / 2
    except:
        return np.nan

df_mh['cgpa'] = df_mh['cgpa'].apply(parse_cgpa)

# Encode target: Depression Yes=1, No=0
df_mh['depression'] = df_mh['depression'].map({'Yes': 1, 'No': 0})

# Encode remaining categoricals
df_mh['gender'] = df_mh['gender'].map({'Male': 0, 'Female': 1})
for col in ['married','anxiety','panic_attack','treatment']:
    df_mh[col] = df_mh[col].map({'Yes': 1, 'No': 0})

# Year of study: extract number
df_mh['year'] = df_mh['year'].str.extract('(\d)').astype(float)

# One-hot encode course (many unique values)
df_mh = pd.get_dummies(df_mh, columns=['course'], drop_first=True)

# Handle missing
df_mh.fillna(df_mh.median(numeric_only=True), inplace=True)

print("Cleaned shape:", df_mh.shape)
print("Depression balance:\n", df_mh['depression'].value_counts())