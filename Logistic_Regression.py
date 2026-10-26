import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

# Loading the dataset
train_df = pd.read_csv('titanic./train.csv')
test_df = pd.read_csv('titanic./test.csv')


# Set the visual style
sns.set_theme(style='whitegrid')

# Creat a figure with multiple subplots
fig,axes = plt.subplots(1,3, figsize=(18,5))

# Plot 1: Survival by Sex
sns.barplot(data=train_df, x='Sex', y='Survived', ax=axes[0], palette='viridis')
axes[0].set_title('Survival Rate by Gender')

# Plot 2: Survival by Pclass
sns.barplot(data=train_df, x='Pclass', y='Survived', ax=axes[1], palette='magma')
axes[1].set_title('Survival Rate by Passenger Class')

# Plot 3: Age Distribution of Survived vs Not Survived
sns.histplot(data=train_df, x='Age',hue='Survived', kde=True,ax=axes[2], element='step')
axes[2].set_title('Age Distribution by Survival')

plt.tight_layout()
# To see the data in the graph plot
# plt.show()

print(f'Train shape: {train_df.shape}')
print(train_df.head())

def preprocess_titanic(df):
    # Drop columns with too many missing values or irrelevant info
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Encode categorical features
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))

    return df

train_processed = preprocess_titanic(train_df)

# Separate features and target
X = train_processed.drop('Survived', axis=1)
y = train_processed['Survived']

# Split
X_train, X_val , y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=42)

# Impute and Scale
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train = scaler.fit_transform(imputer.fit_transform(X_train))
X_val = scaler.transform(imputer.transform(X_val))

# Train Logistic Regression
titanic_model = LogisticRegression()
titanic_model.fit(X_train, y_train)

# Evaluate
y_pred = titanic_model.predict(X_val)
print('Titanic Survival Classification Report (Validation Set): ')
print(classification_report(y_val, y_pred))
