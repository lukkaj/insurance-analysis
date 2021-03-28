#!/usr/bin/env python3
# Machine learning project 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy.stats import skew
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt
from sklearn.pipeline import make_pipeline
import operator

path_to_data = "~/Downloads/insurance.csv"
insurance_data = pd.read_csv(path_to_data)


# No missing values
print(insurance_data.isnull().sum())
df2 = pd.DataFrame(insurance_data)
# Dataframe with the needed columns
df = pd.DataFrame(insurance_data)
print(df.head())

# change sex and smoker to numeric value
df['sex'] = pd.get_dummies(df['sex'])
enc = preprocessing.LabelEncoder()
df['smoker'] = enc.fit_transform(df['smoker'])
df['region'] = enc.fit_transform(df['region'])


# Some explorative data-analysis
#df.hist(rwidth=0.9,figsize = (8,8))
#plt.show()
"""
f, axs = plt.subplots(ncols=5, figsize = (8,8))
sns.barplot(x = 'smoker', y = 'charges', data = df, ax = axs[0])
sns.barplot(x = 'region', y = 'charges', data = df, ax = axs[1])
sns.barplot(x = 'age', y = 'charges', data = df, ax = axs[2])
sns.barplot(x = 'bmi', y = 'charges', data = df, ax = axs[3])
sns.barplot(x = 'children', y = 'charges', data = df, ax = axs[4])
plt.show()
"""
print(df.describe())

# check if any correlations
#cor = df.corr()
#sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, annot=True)
#plt.show()

# checking for skewness
print("Skewness", skew(df['charges']))

# Charges are highly right skewed, which can be fixed with log transformation
#df['charges'] = np.log(df['charges'])
cor = df.corr()
sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, annot=True)
plt.show()

df.hist(rwidth=0.9,figsize = (8,8))
plt.show()

# separate to features and label 
x = df.drop(['charges'], axis = 1)
y = df['charges']

# split data to train, validation and test data
# split ratios are 0.6, 0.2 and 0.2 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, random_state =1)

# First linear regression model
lreg = LinearRegression()
lreg.fit(x_train, y_train)
# These are the weights 
print("Weights: ",lreg.coef_)
# Intercept of the model
print("Intercept: ",lreg.intercept_)
# R^2
print("model score: ",lreg.score(x_val, y_val))




# Performance evaluation

y_pred_val = lreg.predict(x_val)

print("R2: ", (r2_score(y_pred_val,y_val)))
# Plot output

plt.scatter(y_val, y_pred_val)
plt.xlabel("Actual labels")
plt.ylabel("Predicted labels")
plt.title("Insurance cost predictions")
z = np.polyfit(y_val, y_pred_val, 1)
p = np.poly1d(z)
plt.plot(y_val, p(y_val), color='red')
plt.show()

# Training error
y_pred_train = lreg.predict(x_train)
mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
rmae_train = sqrt(metrics.mean_absolute_error(y_train, y_pred_train))
mse_train = metrics.mean_squared_error(y_train, y_pred_train)
rmse_train = sqrt(metrics.mean_squared_error(y_train, y_pred_train))
print("Mean absolute error: ", mae_train)
print("Root mean absolute error: ", rmae_train)
print("Mean absolute error: ", mse_train)
print("Root mean squared error: ", rmse_train)


# Validation error
mae_val = metrics.mean_absolute_error(y_val, y_pred_val)
rmae_val = sqrt(metrics.mean_absolute_error(y_val, y_pred_val))
mse_val = metrics.mean_squared_error(y_val, y_pred_val)
rmse_val = sqrt(metrics.mean_squared_error(y_val, y_pred_val))
print("Mean absolute error: ", mae_val)
print("Root mean absolute error: ", rmae_val)
print("Mean absolute error: ", mse_val)
print("Root mean squared error: ", rmse_val)

# Validation error vs training error
plot1 = plt.bar(0, rmse_train, width = 0.5)
plot2 = plt.bar(0.6, rmse_val, width = 0.5)

plt.xlabel("Training and validation")
plt.ylabel("Root mean squared error")
plt.legend(["training", "validation"], loc="center left", bbox_to_anchor=(0.8,1.0))
plt.show()

plot1 = plt.bar(0, mae_train, width = 0.5)
plot2 = plt.bar(0.6, mae_val, width = 0.5)

plt.xlabel("Training and validation")
plt.ylabel("Root mean squared error")
plt.legend(["training", "validation"], loc="center left", bbox_to_anchor=(0.8,1.0))
plt.show()

plot1 = plt.bar(0, rmae_train, width = 0.5)
plot2 = plt.bar(0.6, rmae_val, width = 0.5)

plt.xlabel("Training and validation")
plt.ylabel("Root mean squared error")
plt.legend(["training", "validation"], loc="center left", bbox_to_anchor=(0.8,1.0))
plt.show()


# Next model is polynomial regression

score_pol = []
mae_pol = []
rmae_pol = []
mse_pol = []
rmse_pol = []
r_squared = []
optimal_mse_pol = 0


test_score_pol = []
test_mae_pol = []
test_rmae_pol = []
test_mse_pol = []
test_rmse_pol = []
test_r_squared = []
y_test_pred_poly = []
optimal_mse_pol_test = 0

x_rele = df.drop(['charges', 'sex', 'region'], axis=1)

x_train_pol, x_test_pol, y_train_pol, y_test_pol = train_test_split(x, y, test_size = 0.2, random_state =1)
x_train_pol, x_val_pol, y_train_pol, y_val_pol = train_test_split(x_train, y_train, test_size = 0.25, random_state =1)

for i in range(1,10):
    poly_reg = make_pipeline(PolynomialFeatures(degree = i), LinearRegression())
    poly_reg.fit(x_train_pol, y_train_pol)
    score_pol.append(poly_reg.score(x_train_pol, y_train_pol))
    r_squared.append(r2_score(poly_reg.predict(x_val_pol), y_val_pol))
    mae_pol.append(mean_absolute_error(poly_reg.predict(x_val_pol), y_val_pol))
    rmae_pol.append(sqrt(mean_absolute_error(poly_reg.predict(x_val_pol), y_val_pol)))
    mse_pol.append(mean_squared_error(poly_reg.predict(x_val_pol), y_val_pol))
    rmse_pol.append(sqrt(mean_squared_error(poly_reg.predict(x_val_pol), y_val_pol)))
    # running this test already I found that degree 3 polynomial matches data the best
    if(i == 3):
        optimal_mse_pol = mean_squared_error(poly_reg.predict(x_train_pol), y_train_pol)
        poly_reg.fit(x_train_pol, y_train_pol)
        test_score_pol.append(poly_reg.score(x_test_pol, y_test_pol))
        test_r_squared.append(r2_score(poly_reg.predict(x_test_pol), y_test_pol))
        test_mae_pol.append(mean_absolute_error(poly_reg.predict(x_test_pol), y_test_pol))
        test_rmae_pol.append(sqrt(mean_absolute_error(poly_reg.predict(x_test_pol), y_test_pol)))
        test_mse_pol.append(mean_squared_error(poly_reg.predict(x_test_pol), y_test_pol))
        test_rmse_pol.append(sqrt(mean_squared_error(poly_reg.predict(x_test_pol), y_test_pol)))
        y_test_pred_poly.append(poly_reg.predict(x_test_pol))
        optimal_mse_pol_test = mean_squared_error(poly_reg.predict(x_test_pol), y_test_pol)




print("R squared: ", r_squared)
print()
print("Mean absolute error: ", mae_pol)
print()
print("Root mean absolute error: ", rmae_pol)
print()
print("Mean squared error: ", mse_pol)
print()
print("Root mean squared error: ", rmse_pol)
print()
print("Training model score: ", score_pol)
print()

"""
From upper test the degree 3 polynomial fits the data best.
"""
print("R squared test data: ", test_r_squared)
print()
print("Mean absolute error test data: ", test_mae_pol)
print()
print("Root mean absolute error test data: ", test_rmae_pol)
print()
print("Mean squared error test data: ", test_mse_pol)
print()
print("Root mean squared error test data: ", test_rmse_pol)
print()
print("Test model score test data: ", test_score_pol)
print()

print(pd.DataFrame({"Known": y_test_pol, "Predicted": y_test_pred_poly[0]}))

value_pol_test = y_test_pred_poly[0].reshape(268,)

plt.scatter(y_test_pol, value_pol_test, color='blue')
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(y_test_pol, value_pol_test), key=sort_axis)
y_test_pol, value_pol_test = zip(*sorted_zip)
plt.plot(y_test_pol, value_pol_test, color='red')
#z = np.polyfit(y_test_pol, value_pol_test, 3)
#p = np.poly1d(z)
#plt.plot(y_test_pol, p(y_test_pol), color='red')
#plt.plot(y_test_pol, y_test_pred_poly[0], color='red')
plt.xlabel("Actual labels")
plt.ylabel("Predicted labels")
plt.title("Insurance cost predictions")
plt.show()




plot1 = plt.bar(0, optimal_mse_pol, width = 0.5)
plot2 = plt.bar(0.6, optimal_mse_pol_test, width = 0.5)

plt.xlabel("Training and test")
plt.ylabel("mean squared error")
plt.legend(["training", "test"], loc="center left", bbox_to_anchor=(0.8,1.0))
plt.show()




#print(pd.DataFrame({"Known": np.exp(y_test_pol), "Predicted": np.exp(y_test_pred_poly[0])}))
#print(pd.DataFrame({"Known": df2['charges']}))