from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.svm import SVC
import numpy as np
import json 


DATASET_PATH = '../dataset/dataset_threshold_100_shift_05_2.txt'

dataset = []
with open(DATASET_PATH) as f:
    dataset = json.load(f)

X = np.array(dataset[0])
Y = np.array(dataset[1])

tmp = []
for y in Y:
    tmp.append(np.argmax(y))
Y = tmp

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train, Y_train)
KNN_pred = KNN_model.predict(X_test)
print(classification_report(Y_test, KNN_pred))
print(accuracy_score(Y_test, KNN_pred))

SVC_model = SVC()
SVC_model.fit(X_train, Y_train)
SVC_pred = SVC_model.predict(X_test)
print(classification_report(Y_test, SVC_pred))
print(accuracy_score(Y_test, SVC_pred))
