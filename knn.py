
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

wine = pd.read_csv("D:\Semester 6\ML\LAB\ML LAB-2\wine.csv")

target=wine.Wine
data=wine.values[:,1:]

print(target.shape)
print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print("Confusion Martix:")
print(metrics.confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(metrics.classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
