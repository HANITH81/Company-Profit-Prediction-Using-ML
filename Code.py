import pandas as pd        #pandas verison -> 2.0.1
import numpy as np         #numpy version ->  1.24.3
from matplotlib import pyplot as plt   #matplotlib version ->   3.7.1
from sklearn.model_selection import train_test_split     #sklearn version -> 1.2.1
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv(r'F:\Exposys\Data.csv')
X = df[['R&D Spend','Administration','Marketing Spend']]
Y = df['Profit']

print(df.head())
print(df.info())
print(df.describe)

#Finding the neccesarry things for analyzing data
# print(df['Administration'].describe())
# print(df['Marketing Spend'].describe())
# print(df.describe())
# plt.plot(df['R&D Spend'],df['Profit'])
# plt.show()
# r = df[df['Marketing Spend'] > 400000]
# print(r)
# c = df['Marketing Spend'].value_counts()
# print(c)
# plt.hist(df['R&D Spend'],bins=10)
# plt.show()
# print(np.mean(df['R&D Spend']))
# print(np.median(df['R&D Spend']))
# print(np.std(df['R&D Spend']))

# print('2:')
# print(np.mean(df['Administration']))
# print(np.median(df['Administration']))
# print(np.std(df['Administration']))

# print('3:')
# print(np.mean(df['Marketing Spend']))
# print(np.median(df['Marketing Spend']))
# print(np.std(df['Marketing Spend']))

# print(df.isna().sum())
# print(df.dtypes)

# Correlations

# col1,col2,col3,col4 = df.columns
# c12 = df[col1].corr(df[col2])
# c13 = df[col1].corr(df[col3])
# c23 = df[col2].corr(df[col3])
# print('C12 '+str(c12))
# print('C13 '+str(c13))
# print('C23 '+str(c23))

# No Reduntant data

# # Ploting
# plt.plot(df['R&D Spend'],df['Profit'])
# plt.show()
# plt.plot(df['Administration'], df['Profit'])
# plt.show()
# plt.plot(df['Marketing Spend'], df['Profit'])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=60)
# Construct different regression algorithms
linear_reg = LinearRegression()
decision_tree_reg = DecisionTreeRegressor()

# Train the models
linear_reg.fit(X_train, y_train)
decision_tree_reg.fit(X_train, y_train)

# Predict on the test set
linear_reg_pred = linear_reg.predict(X_test)
decision_tree_pred = decision_tree_reg.predict(X_test)

# Calculate different regression metrics
linear_reg_mse = mean_squared_error(y_test, linear_reg_pred)
linear_reg_mae = mean_absolute_error(y_test, linear_reg_pred)
linear_reg_r2 = r2_score(y_test, linear_reg_pred)

decision_tree_mse = mean_squared_error(y_test, decision_tree_pred)
decision_tree_mae = mean_absolute_error(y_test, decision_tree_pred)
decision_tree_r2 = r2_score(y_test, decision_tree_pred)

# Choose the best model based on the metrics
best_model = 'Linear Regression' if linear_reg_r2 > decision_tree_r2 else 'Decision Tree'

# Print the regression metrics
print("Linear Regression Metrics:")
print("MSE:", linear_reg_mse)
print("MAE:", linear_reg_mae)
print("R2 Score:", linear_reg_r2)

print("\nDecision Tree Metrics:")
print("MSE:", decision_tree_mse)
print("MAE:", decision_tree_mae)
print("R2 Score:", decision_tree_r2)

print("\nBest Model:", best_model)