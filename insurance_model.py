import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency


warnings.filterwarnings('ignore')


df = pd.read_csv('insurance.csv')

# EDA(Exploratory Data Analysis)

# print('Shape of the Data: ',df.shape)
# print('Data Info: ',df.info())
# print('Describe: ', df.describe())

# print('To check the Null values : ', df.isnull().sum())

numeric_columns = ['age', 'bmi', 'children', 'charges']
for cols in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[cols], kde=True, bins=20)


# To see the Other Fields
fig, axes = plt.subplots(1,3, figsize=(15,4))
sns.countplot(data=df, x='children', ax=axes[0])
sns.countplot(data=df, x='sex', ax=axes[1])
sns.countplot(data=df, x='smoker', ax=axes[2])

plt.tight_layout()
# plt.show()

# It will help to understand the Data (EDA)


# 2. Now for the box plots
for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    # plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

# Data Cleaning and Processing
df_cleaned = df.copy()
df_cleaned.head()

# It will Delete the duplicates
df_cleaned.drop_duplicates(inplace=True)
print(df_cleaned.shape)

# To check the null data values
df_cleaned.isnull().sum()

# It help to know all the datatypes
df_cleaned.dtypes

# It will help to count the data in the columns
df_cleaned['sex'].value_counts()

# Let's convert the object into numeric value so we will encode
# Label Encoding
df_cleaned['sex'] = df_cleaned['sex'].map({"male":0,"female":1})


df_cleaned['smoker'].value_counts()
df_cleaned['smoker'] = df_cleaned['smoker'].map({"no":0,"yes":1})
# print('After Label Encoding: \n',df_cleaned.head())

# To rename the Columns
df_cleaned.rename(columns={
    'sex': 'is_female',
    'smoker': 'is_smoker'
}, inplace=True)

# print('After Renamed: \n', df_cleaned.head())

# One Hot encoding
df_cleaned = pd.get_dummies(df_cleaned,columns=['region'],drop_first=True)
df_cleaned = df_cleaned.astype(int)

# print(df_cleaned.head())

# Feature Engineering and Extraction
# pd.cut assigns the value to it's corresponding groups
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=['Underweight','Normal','Overweight','Obese']
)

# It's time to do one hot encoding
df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
df_cleaned = df_cleaned.astype(int)

# Now Feature Scaling Comes

cols= ['age', 'bmi', 'children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
# print(df_cleaned.head())

# To Select the Major Features
# Checking with ehe Pearson Correlation Calculation

# This is the list of features that we will compare with charges
selected_features = [
    'age', 'bmi', 'children', 'is_female','is_smoker', 'region_northwest',
    'region_southeast','region_southwest', 'bmi_category_Normal','bmi_category_Overweight', 'bmi_category_Obese'
]

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation'])
correlation_df.sort_values(by='Pearson Correlation', ascending=False)

cat_features = [
    'is_female','is_smoker', 'region_northwest',
    'region_southeast','region_southwest',
    'bmi_category_Normal','bmi_category_Overweight',
    'bmi_category_Obese'
]


aplha = 0.05
df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'],q=4,labels=False)
chi2_results = {}
for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat , p_val, _,_ = chi2_contingency(contingency)
    decision = 'Reject Null(keep Feature)' if p_val < aplha else 'Accept Null(Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }
chi2_df  = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')
# print(chi2_df)

final_df = df_cleaned[['age','is_female','bmi', 'children', 'is_smoker', 'charges', 'region_southeast', 'bmi_category_Obese']]
print(final_df.head())