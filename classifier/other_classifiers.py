from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 
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


def plot_accuracy(X_train, Y_train, X_test, Y_test, model):
    feature_num = X_train.shape[1]
    fig, ax = plt.subplots()
    ax.set_title(model)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    acc_for_all = accuracy_score(Y_test, pred)
    print(classification_report(Y_test, pred))
    print(confusion_matrix(Y_test, pred))
    ax.hlines(acc_for_all * 100, 0, feature_num - 1, label=str(acc_for_all))
    acc = []
    for i in range(feature_num):
        tmp_X_train = utils.remove_feature(X_train, i)
        tmp_X_test = utils.remove_feature(X_test, i)
        model.fit(tmp_X_train, Y_train)
        pred = model.predict(tmp_X_test)
        acc.append(accuracy_score(Y_test, pred) * 100)
    points = [i for i in range(feature_num)]
    xticks = [i + 1 for i in range(feature_num)]
    plt.xticks(points, xticks)
    ax.plot(range(feature_num), acc)
    ax.set_xlabel('feature number')
    ax.set_ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()


def plot_feature_importances(X_train, Y_train, X_test, Y_test, model):
    feature_num = X.shape[1]
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print(classification_report(Y_test, pred))
    print(confusion_matrix(Y_test, pred))
    fig, ax = plt.subplots()
    ax.set_title('Accuracy: ' + str(accuracy_score(Y_test, pred)))
    points = [i for i in range(feature_num)]
    xticks = [i + 1 for i in range(feature_num)]
    plt.xticks(points, xticks)
    ax.bar(range(feature_num), model.feature_importances_)
    plt.show()


def prepare_dataset(X_train, X_test):
    tmp_X_train = X_train
    tmp_X_test = X_test
    tmp_X_train = utils.remove_feature(tmp_X_train, 17)
    tmp_X_train = utils.remove_feature(tmp_X_train, 16)
    tmp_X_train = utils.remove_feature(tmp_X_train, 15)
    tmp_X_train = utils.remove_feature(tmp_X_train, 14)
    tmp_X_train = utils.remove_feature(tmp_X_train, 13)
    tmp_X_train = utils.remove_feature(tmp_X_train, 12)
    tmp_X_train = utils.remove_feature(tmp_X_train, 11)
    tmp_X_train = utils.remove_feature(tmp_X_train, 10)
    tmp_X_train = utils.remove_feature(tmp_X_train, 9)
    tmp_X_train = utils.remove_feature(tmp_X_train, 8)
    tmp_X_train = utils.remove_feature(tmp_X_train, 7)
    tmp_X_train = utils.remove_feature(tmp_X_train, 6)
    # tmp_X_train = utils.remove_feature(tmp_X_train, 5)
    # tmp_X_train = utils.remove_feature(tmp_X_train, 4)
    # tmp_X_train = utils.remove_feature(tmp_X_train, 3)
    # tmp_X_train = utils.remove_feature(tmp_X_train, 2)
    # tmp_X_train = utils.remove_feature(tmp_X_train, 1)
    # tmp_X_train = utils.remove_feature(tmp_X_train, 0)
    tmp_X_test = utils.remove_feature(tmp_X_test, 17)
    tmp_X_test = utils.remove_feature(tmp_X_test, 16)
    tmp_X_test = utils.remove_feature(tmp_X_test, 15)
    tmp_X_test = utils.remove_feature(tmp_X_test, 14)
    tmp_X_test = utils.remove_feature(tmp_X_test, 13)
    tmp_X_test = utils.remove_feature(tmp_X_test, 12)
    tmp_X_test = utils.remove_feature(tmp_X_test, 11)
    tmp_X_test = utils.remove_feature(tmp_X_test, 10)
    tmp_X_test = utils.remove_feature(tmp_X_test, 9)
    tmp_X_test = utils.remove_feature(tmp_X_test, 8)
    tmp_X_test = utils.remove_feature(tmp_X_test, 7)
    tmp_X_test = utils.remove_feature(tmp_X_test, 6)
    # tmp_X_test = utils.remove_feature(tmp_X_test, 5)
    # tmp_X_test = utils.remove_feature(tmp_X_test, 4)
    # tmp_X_test = utils.remove_feature(tmp_X_test, 3)
    # tmp_X_test = utils.remove_feature(tmp_X_test, 2)
    # tmp_X_test = utils.remove_feature(tmp_X_test, 1)
    # tmp_X_test = utils.remove_feature(tmp_X_test, 0)
    return tmp_X_train, tmp_X_test


X, Y = utils.get_dataset('dataset_threshold_100.txt')
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# X_train, Y_train = utils.get_dataset('train_threshold_100.txt')
# X_test, Y_test = utils.get_dataset('test_threshold_100.txt')

Y_train = utils.convert_Y_to_class_numbers(Y_train)
Y_test = utils.convert_Y_to_class_numbers(Y_test)

# X_train, X_test = prepare_dataset(X_train, X_test)

# plot_accuracy(X_train, Y_train, X_test, Y_test, KNeighborsClassifier(n_neighbors=5))
# plot_accuracy(X_train, Y_train, X_test, Y_test, SVC())
# plot_accuracy(X_train, Y_train, X_test, Y_test, GaussianNB())
# plot_accuracy(X_train, Y_train, X_test, Y_test, DecisionTreeClassifier())
# plot_accuracy(X_train, Y_train, X_test, Y_test, RandomForestClassifier())

# plot_feature_importances(X_train, Y_train, X_test, Y_test, DecisionTreeClassifier())
plot_feature_importances(X_train, Y_train, X_test, Y_test, RandomForestClassifier(n_estimators=500))
