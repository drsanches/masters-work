from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import utils


def test_fetures(X, Y):
    print('K-Neighbors')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    KNN_model = KNeighborsClassifier(n_neighbors=5)
    KNN_model.fit(X_train, Y_train)
    KNN_pred = KNN_model.predict(X_test)
    print("%.2f%%" % (accuracy_score(Y_test, KNN_pred)*100))
    for i in range(X.shape[1]):
        tmp = utils.remove_feature(X, i)
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
        KNN_model = KNeighborsClassifier(n_neighbors=5)
        KNN_model.fit(X_train, Y_train)
        KNN_pred = KNN_model.predict(X_test)
        # print(classification_report(Y_test, KNN_pred))
        print("%.2f%%" % (accuracy_score(Y_test, KNN_pred)*100), end=' ')

    print('\nSVM')
    X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
    SVC_model = SVC()
    SVC_model.fit(X_train, Y_train)
    SVC_pred = SVC_model.predict(X_test)
    print("%.2f%%" % (accuracy_score(Y_test, SVC_pred)*100))
    for i in range(X.shape[1]):
        tmp = utils.remove_feature(X, i)
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
        SVC_model = SVC()
        SVC_model.fit(X_train, Y_train)
        SVC_pred = SVC_model.predict(X_test)
        # print(classification_report(Y_test, SVC_pred))
        print("%.2f%%" % (accuracy_score(Y_test, SVC_pred)*100), end=' ')

    print('\nNaive Bayes Classifier')
    X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
    GNB_model = GaussianNB()
    GNB_model.fit(X_train, Y_train)
    GNB_pred = GNB_model.predict(X_test)
    print("%.2f%%" % (accuracy_score(Y_test, GNB_pred)*100))
    for i in range(X.shape[1]):
        tmp = utils.remove_feature(X, i)
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
        GNB_model = GaussianNB()
        GNB_model.fit(X_train, Y_train)
        GNB_pred = GNB_model.predict(X_test)
        # print(classification_report(Y_test, GNB_pred))
        print("%.2f%%" % (accuracy_score(Y_test, GNB_pred)*100), end=' ')
        

    print('\nDecision Tree')
    X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, Y_train)
    tree_pred = tree_model.predict(X_test)
    print("%.2f%%" % (accuracy_score(Y_test, tree_pred)*100))
    for i in range(X.shape[1]):
        tmp = utils.remove_feature(X, i)
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
        tree_model = DecisionTreeClassifier()
        tree_model.fit(X_train, Y_train)
        tree_pred = tree_model.predict(X_test)
        # print(classification_report(Y_test, tree_pred))
        print("%.2f%%" % (accuracy_score(Y_test, tree_pred)*100), end=' ')

    print('\nRandom Forest')
    X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
    forest_model = RandomForestClassifier()
    forest_model.fit(X_train, Y_train)
    forest_pred = forest_model.predict(X_test)
    print("%.2f%%" % (accuracy_score(Y_test, forest_pred)*100))
    for i in range(X.shape[1]):
        tmp = utils.remove_feature(X, i)
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
        forest_model = RandomForestClassifier()
        forest_model.fit(X_train, Y_train)
        forest_pred = forest_model.predict(X_test)
        # print(classification_report(Y_test, forest_pred))
        print("%.2f%%" % (accuracy_score(Y_test, forest_pred)*100), end=' ')

def  k_nearest_neighbors(X_train, Y_train, X_test, Y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    return accuracy_score(Y_test, pred)

def support_vector_machine(X_train, Y_train, X_test, Y_test):
    model = SVC()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    return accuracy_score(Y_test, pred)

def naive_bayes_classifier(X_train, Y_train, X_test, Y_test):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    return accuracy_score(Y_test, pred)

def decision_tree(X_train, Y_train, X_test, Y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    return accuracy_score(Y_test, pred)

def random_forest(X_train, Y_train, X_test, Y_test):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    return accuracy_score(Y_test, pred)

def plot_accuracy(X, Y, method):
    feature_num = X.shape[1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    fig, ax = plt.subplots()
    ax.set_title(method.__name__)
    ax.hlines(method(X_train, Y_train, X_test, Y_test) * 100, 0, feature_num - 1)
    acc = []
    for i in range(feature_num):
        tmp = utils.remove_feature(X, i)
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, Y, test_size=0.2, random_state=42)
        acc.append(method(X_train, Y_train, X_test, Y_test) * 100)
    ax.plot(range(feature_num), acc)
    points = [i + 0.5 for i in range(feature_num)]
    xticks = [i + 1 for i in range(feature_num)]
    plt.xticks(points, xticks)
    plt.show()


X, Y = utils.get_dataset('dataset_threshold_100.txt')
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

plot_accuracy(X, Y, k_nearest_neighbors)
plot_accuracy(X, Y, support_vector_machine)
plot_accuracy(X, Y, naive_bayes_classifier)
plot_accuracy(X, Y, decision_tree)
plot_accuracy(X, Y, random_forest)