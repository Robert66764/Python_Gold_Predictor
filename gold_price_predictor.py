#Importing the various libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#Collecting the gold dataset.
gold_data = pd.read_csv('gld_price_data.csv')

#Analyzing the data.
gold_data.head()
gold_data.tail()
gold_data.describe()
gold_data.info()

#Deleting the date column.
del gold_data['Date']

gold_data_correlation = gold_data.corr()

plt.figure(figsize = (8,8))
sns.heatmap(gold_data_correlation, cbar = True, square = True, cmap = 'Blues', fmt = '.1f', annot = True, annot_kws = {'size': 8})

#Check the distribution of the gold price.
sns.displot(gold_data['GLD'], color = 'yellow')

X = gold_data
Y = gold_data['GLD']

print(X)
print(Y)

#Splitting the data into training data and testing dataa
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 2)

#Model Training: Random Forest Regressor
regressor = RandomForestRegressor(n_estimators = 100)

#Training the model
regressor.fit(X_train,Y_train)

#Prediction on Test Data
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

#Calculating the Rsquared error to deduce whether the prediction is accruate
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squred error " + str(error_score))

#Comparing the actual values and predicted values in a plot.
Y_test = list(Y_test)

#Plotting the actual values against the predicted values. 
plt.plot(Y_test, color = 'yellow', label = 'Actual Value')
plt.plot(test_data_prediction, color = 'green', label = 'Predicited Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel("Number of Values")
plt.ylabel("GLD Price")
plt.legend()
plt.show()





