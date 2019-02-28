from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

data = pd.read_csv("D:/Semester 6/ML/LAB/ML LAB-1/iris.csv")

x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data.species

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

LogReg = LogisticRegression()
LogReg.fit(x_train, y_train)

y_pred = LogReg.predict(x_test)

confusionMatrix = metrics.confusion_matrix(y_test, y_pred)
print("The confusion matrix:\n", confusionMatrix)
print("The Error rate:", 1-metrics.accuracy_score(y_test, y_pred))
print("The accuracy score:", metrics.accuracy_score(y_test, y_pred))
print("The precision score:", metrics.precision_score(y_test, y_pred,average=None))
print("The recall score:", metrics.recall_score(y_test, y_pred,average=None))
print("The F-Measure:", metrics.f1_score(y_test, y_pred,average=None))