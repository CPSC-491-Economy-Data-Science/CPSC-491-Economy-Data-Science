import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split # split data into test/train
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error

# read in dataset
data = pd.read_csv('C:/Users/Grego/Documents/490 Class/UsEconCSVs/qrt_data_new.csv')
# remove the first column (column that was generated when creating csv files and is just redundant indexing)
data_final = data.drop(data.columns[0], axis=1)

# seperate the target valiable
Y = data_final['GDP']
# seperate the independent variables
X = data_final.drop(['GDP', 'perc_ch_GDP', 'DATE'], axis =1)

# split the data into testing and training data used for modeling and testing. Set the test/train split to 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# testing different learning rates for the Elastic-net algorithm to pick from, it will choose the best
alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
# create the model using the learning rates above and set cv to 5
elastic_cv = ElasticNetCV(alphas=alphas, cv=5)
#elastic_cv = ElasticNet()
# fit the model using the training data that we split above
model = elastic_cv.fit(X_train, y_train)
# create predictions using the testing data that was created above
ypred = model.predict(X_test) 
# test the model on Quarter 1, 2 2022 data
test = model.predict([[4.277553, 1.94, 1.826774, 207673.1, -0.4, 1859, 0.04961, 3.8],[5.041692, 2.93, 2.167097, 198846.2, -3.1, 1738, 0.0574, 3.6]])
# get the r-squared value of the model using testing data
score = model.score(X_test, y_test)
# get values used for seeing how well the model does
mse = mean_squared_error(y_test, ypred)
rmse = np.sqrt(mse)
print(score)
print(mse)
print(rmse)
print(test)
# plot the predicted values and the orginal values
x_ax = range(len(X_test))
plt.scatter(x_ax, y_test, color='blue', label='original')
plt.plot(x_ax, ypred, color='red', label='predicted')
plt.legend()
plt.show()

# the plots below were used for graphing data from r-studio

# data2 = pd.read_csv('C:/Users/Grego/Documents/490 Class/UsEconCSVs/1ypred.csv')
# data2_final = data2.drop(data.columns[0], axis=1)


# y_test2 = data2_final['perc_ch_GDP']
# ypred2 = data2_final['ypred']
# x_ax2 = data2_final['DATE']

# plt.scatter(x_ax2, y_test2, s=5, color='blue', label='original')
# plt.plot(x_ax2, ypred2, lw=0.8, color='red', label='predicted')
# plt.legend()
# plt.show()

# data3 = pd.read_csv('C:/Users/Grego/Documents/490 Class/UsEconCSVs/2ypred.csv')
# data3_final = data3.drop(data.columns[0], axis=1)


# y_test3 = data3_final['perc_ch_GDP']
# ypred3 = data3_final['ypred']
# x_ax3 = data3_final['DATE']

# plt.scatter(x_ax3, y_test3, s=5, color='blue', label='original')
# plt.plot(x_ax3, ypred3, lw=0.8, color='red', label='predicted')
# plt.legend()
# plt.show()

# data4 = pd.read_csv('C:/Users/Grego/Documents/490 Class/UsEconCSVs/3ypred.csv')
# data4_final = data4.drop(data.columns[0], axis=1)


# y_test4 = data4_final['perc_ch_GDP']
# ypred4 = data4_final['ypred']
# x_ax4 = data4_final['DATE']

# plt.scatter(x_ax4, y_test4, s=5, color='blue', label='original')
# plt.plot(x_ax4, ypred4, lw=0.8, color='red', label='predicted')
# plt.legend()
# plt.show()


# data5 = pd.read_csv('C:/Users/Grego/Documents/490 Class/UsEconCSVs/1ypredGDP.csv')
# data5_final = data5.drop(data.columns[0], axis=1)


# y_test5 = data5_final['GDP']
# ypred5 = data5_final['ypred']
# x_ax5 = data5_final['DATE']

# plt.scatter(x_ax5, y_test5, s=5, color='blue', label='original')
# plt.plot(x_ax5, ypred5, lw=0.8, color='red', label='predicted')
# plt.legend()
# plt.show()