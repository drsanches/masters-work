from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import utils


X, Y = utils.get_dataset('dataset_threshold_100_shift_05_2.txt')
Y = utils.convert_Y_to_class_numbers(Y)

# print(X.shape[1])
# X = utils.remove_feature(X, 17)
# X = utils.remove_feature(X, 16)
# X = utils.remove_feature(X, 15)
# X = utils.remove_feature(X, 14)
# X = utils.remove_feature(X, 13)
# X = utils.remove_feature(X, 12)
# X = utils.remove_feature(X, 11)
# X = utils.remove_feature(X, 10)
# X = utils.remove_feature(X, 9)
# X = utils.remove_feature(X, 8)
# X = utils.remove_feature(X, 7)
# X = utils.remove_feature(X, 6)
# # X = utils.remove_feature(X, 5)
# # X = utils.remove_feature(X, 4) # 0.02
# # X = utils.remove_feature(X, 3)
# # X = utils.remove_feature(X, 2)
# # X = utils.remove_feature(X, 1) # 0.02
# # X = utils.remove_feature(X, 0)
# print(X.shape[1])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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