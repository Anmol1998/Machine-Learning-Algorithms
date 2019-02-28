import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

#Dataset URL: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
data = pd.read_csv("day.csv")

#x_tmp = data.registered
x_tmp = data.values[:,14]
#y_tmp = data.cnt
y_tmp = data.values[:,-1]

x = []
y = []

for i in range(len(x_tmp)):
    x.append([x_tmp[i]])
    y.append([y_tmp[i]])

print(x_tmp)
print(y_tmp)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=1)

LinReg = linear_model.LinearRegression()

LinReg.fit(x_train, y_train)
y_pred = LinReg.predict(x_test)

print("The Linear Regression Coefficients:", LinReg.coef_)
print("The Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("The R Squared Score:", round(r2_score(y_test, y_pred), 2))

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, y_pred, color="red", linewidth=5)
plt.xlabel("Registered")
plt.ylabel("Count of users")

plt.show()

