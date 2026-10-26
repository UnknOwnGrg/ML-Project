import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# To read data from the file
df = pd.read_csv('CarPrice_Assignment.csv')

df.info()
print(df.head())

# 'price' is the target. Features like horsepower , curbweight, enginesize are typically strong predictors.
features = ['horsepower', 'curbweight', 'enginesize','highwaympg','wheelbase']

# Handle missing values if any
df_clean = df.dropna(subset=features + ['price'])

X = df_clean[features]
Y = df_clean['price']

print(df_clean[features+ ['price']].head())

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.histplot(df_clean['price'], kde=True)
plt.title('Car Price Distribution')

plt.subplot(1,3,2)
sns.scatterplot(data=df_clean, x='horsepower', y='price')
plt.title('Price vs Horsepower')

plt.subplot(1,3,3)
sns.scatterplot(data=df_clean, x='enginesize', y='price')
plt.title('Price vs Engine Size')

plt.tight_layout()
# plt.show()


# Splitting the data
X_train, X_test, y_train, y_test =  train_test_split(X,Y, test_size=0.2, random_state=42)

# Scaling the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)
score = r2_score(y_test , y_pred)

print(f'R2 score for Car Price Prediction: {score}')

# To see the scaler of X_train_scaled
print(X_train_scaled)