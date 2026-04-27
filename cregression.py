import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

df_mat = pd.read_csv('student-mat.csv', sep=';')
df_por = pd.read_csv('student-por.csv', sep=';')
df = pd.concat([df_mat, df_por], ignore_index=True)

# Step 1 — shape and dtypes
print(df.shape, df.dtypes.value_counts())

# Step 2 — missing values
print(df.isnull().sum())  # this dataset is clean, document that explicitly

# Step 3 — remove rows where G3 == 0 (no-shows, not real grades)
df = df[df['G3'] > 0]

# Step 4 — drop G1 and G2 (intermediate grades, would leak the target)
df_reg = df.drop(columns=['G1', 'G2'])

# Step 5 — encode binary yes/no columns
binary_cols = ['schoolsup','famsup','paid','activities','nursery',
               'higher','internet','romantic']
for col in binary_cols:
    df_reg[col] = df_reg[col].map({'yes': 1, 'no': 0})

# Step 6 — one-hot encode remaining categoricals
df_reg = pd.get_dummies(df_reg, drop_first=True)

print("Cleaned shape:", df_reg.shape)
df_reg.head()