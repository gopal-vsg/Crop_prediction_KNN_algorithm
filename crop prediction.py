import numpy as np
import  pandas as pd

import matplotlib.pyplot as plt
def scatter_plot(title, y_predict):
  plt.figure(figsize=(12,12))
  plt.xticks(rotation=90)
  plt.title(title,fontsize = 16)
  plt.xlabel('Actual',fontsize = 15)
  plt.ylabel('Predictions', fontsize =15)
  plt.scatter(y_test,y_predict)
  plt.show()

data = pd.read_csv('Crop_recommendation.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#print(y)
#y_decoded = le.inverse_transform(y)

#print(y_decoded)
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier(n_neighbors = 5, metric ='minkowski', p = 2)
cls.fit(x_train, y_train)
y_pred=cls.predict(x_test)


print("This model is based on the KNN classifiaction algorithm")
print("and the accuracy of this model is : ", accuracy_score(y_pred, y_test))
confusion_matrix(y_pred, y_test)
print(classification_report(y_test, y_pred))

scatter_plot('K-Nearest Neighbors Classifier', y_pred)

#taking the inputs form the user
print("please enter the required parameters!")
N = float(input("enter the nitrogen content in the soil : "))
P = float(input("enter the phosporous content in the soil : "))
K = float(input("enter the pottasium content in the soil : "))
temperature = float(input("enter the average temp of the region : "))
humidity = float(input("enter the average humidity of the region : "))
ph = float(input("enter the ph of the soil : "))
rainfall = float(input("enter the anual rainfall of the region(centimeters) : "))

#out put
y_ans = cls.predict([[N, P, K, temperature, humidity, ph, rainfall]])
print("The suitable crop is :",le.inverse_transform(y_ans)[0])
