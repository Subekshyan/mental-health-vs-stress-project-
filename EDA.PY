import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Student Performance — EDA', fontsize=16, fontweight='bold')

# 1. Grade distribution
axes[0,0].hist(df_reg['G3'], bins=15, color='steelblue', edgecolor='white')
axes[0,0].set_title('Final grade (G3) distribution')
axes[0,0].set_xlabel('Grade')

# 2. Study time vs grade
sns.boxplot(data=df_reg, x='studytime', y='G3', ax=axes[0,1], palette='Blues')
axes[0,1].set_title('Study time vs final grade')

# 3. Absences vs grade
axes[0,2].scatter(df_reg['absences'], df_reg['G3'], alpha=0.4, color='coral')
axes[0,2].set_title('Absences vs final grade')
axes[0,2].set_xlabel('Absences')

# 4. Failures impact
sns.boxplot(data=df_reg, x='failures', y='G3', ax=axes[1,0], palette='Reds')
axes[1,0].set_title('Past failures vs final grade')

# 5. Internet access
sns.barplot(data=df_reg, x='internet', y='G3', ax=axes[1,1], palette='Set2')
axes[1,1].set_title('Internet access effect on grade')

# 6. Correlation heatmap (top 10 features)
corr = df_reg.corr()[['G3']].sort_values('G3', ascending=False).head(10)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,2])
axes[1,2].set_title('Top 10 correlated features')

plt.tight_layout()
plt.savefig('eda_regression.png', dpi=150)
plt.show()