import numpy as np
import matplotlib.pyplot as plt
import pylab


# plt.rcParams.update({'font.size': 16})


def graph_accuracy(Y, model, info=''):
    X = np.arange(1, len(Y) + 1)
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(X, Y, width=0.6)
    avg = sum(Y) / len(Y)
    ax.hlines(avg, 0.5, len(X) + 0.5, label='Avg = %.2f%%' % (avg*100))
    ax.set_xlabel('Experiment number')
    ax.set_ylabel('Accuraccy')
    ax.set_title(model + ' Accuracy ' + info)
    ax.legend()
    pylab.ylim (0, 1)
    plt.show()


def graph4(Y, model, label, info=''):
    n = np.arange(1, len(Y[0]) + 1)
    x1 = n - 0.3
    x2 = n - 0.1
    x3 = n + 0.1
    x4 = n + 0.3
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(x1, Y[0], width = 0.2, label='Exchange')
    ax.bar(x2, Y[1], width = 0.2, label='Ico wallets')
    ax.bar(x3, Y[2], width = 0.2, label='Mining')
    ax.bar(x4, Y[3], width = 0.2, label='Token contract')
    avg = sum(Y[0] + Y[1] + Y[2] + Y[3]) / len(Y[0] + Y[1] + Y[2] + Y[3])
    ax.hlines(avg, 0.5, len(n) + 0.5, label='Avg = %.2f%%' % (avg*100))
    ax.set_xlabel('Experiment number')
    ax.set_ylabel(label)
    ax.set_title(model + ' ' + label + ' ' + info)
    ax.legend()
    pylab.ylim (0, 1)
    plt.show()


def get_knn_accuracy():
    return [0.67, 0.69, 0.66, 0.79, 0.65, 0.69]

def get_knn_precision():
    return [[0.65, 0.76, 0.73, 0.82, 0.64, 0.70],
            [0.62, 0.39, 0.76, 0.56, 0.52, 0.57],
            [0.43, 0.43, 0.31, 0.50, 0.43, 0.71],
            [0.74, 0.79, 0.62, 0.89, 0.77, 0.82]]

def get_knn_recall():
    return [[0.68, 0.73, 0.72, 0.77, 0.74, 0.75],
            [0.50, 0.47, 0.54, 0.64, 0.48, 0.81],
            [0.20, 0.30, 0.33, 0.57, 0.27, 0.29],
            [0.94, 0.84, 0.82, 0.89, 0.77, 0.77]]

def get_knn_f1():
    return [[0.67, 0.75, 0.73, 0.80, 0.69, 0.72], 
            [0.55, 0.42, 0.63, 0.60, 0.50, 0.67], 
            [0.27, 0.35, 0.32, 0.53, 0.33, 0.42], 
            [0.83, 0.81, 0.71, 0.89, 0.77, 0.79]]

def get_svm_accuracy():
    return [0.44, 0.53, 0.56, 0.49, 0.44, 0.44]

def get_norm_svm_accuracy():
    return [0.64, 0.70, 0.63, 0.74, 0.65, 0.60]

def get_svm_precision():
    return [[0.45, 0.53, 0.56, 0.46, 0.40, 0.42], 
            [0.57, 0.50, 0.69, 0.78, 0.62, 0.73], 
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00], 
            [0.25, 0.57, 0.25, 0.56, 0.00, 0.11]]

def get_norm_svm_precision():
    return [[0.64, 0.74, 0.70, 0.72, 0.64, 0.58], 
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00], 
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00], 
            [0.63, 0.66, 0.52, 0.76, 0.66, 0.62]]

def get_svm_recall():
    return [[0.95, 0.88, 0.94, 0.95, 0.95, 0.88],
            [0.25, 0.47, 0.46, 0.50, 0.43, 0.52],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.06, 0.13, 0.09, 0.11, 0.00, 0.03]]

def get_norm_svm_recall():
    return [[0.82, 0.87, 0.92, 0.84, 0.90, 0.88],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]]

def get_svm_f1():
    return [[0.61, 0.66, 0.70, 0.62, 0.56, 0.56],
            [0.35, 0.48, 0.55, 0.61, 0.51, 0.61],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.10, 0.21, 0.13, 0.19, 0.00, 0.05]]

def get_norm_svm_f1():
    return [[0.72, 0.80, 0.79, 0.77, 0.74, 0.70],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.78, 0.79, 0.69, 0.86, 0.80, 0.77]]

def get_tree_accuracy():
    return [0.85, 0.79, 0.87, 0.91, 0.82, 0.76]

def get_tree_precision():
    return [[0.88, 0.89, 0.88, 0.93, 0.85, 0.71],
            [0.65, 0.47, 0.79, 0.68, 0.74, 0.70],
            [0.86, 0.60, 0.92, 1.00, 0.75, 0.92],
            [0.94, 0.88, 0.91, 0.98, 0.87, 0.81]]

def get_tree_recall():
    return [[0.86, 0.81, 0.88, 0.86, 0.85, 0.88],
            [0.81, 0.60, 0.79, 0.93, 0.61, 0.67],
            [0.80, 0.60, 0.92, 1.00, 0.82, 0.65],
            [0.88, 0.90, 0.91, 0.93, 0.94, 0.73]]

def get_tree_f1():
    return [[0.87, 0.85, 0.88, 0.89, 0.85, 0.79],
            [0.72, 0.53, 0.79, 0.79, 0.67, 0.68],
            [0.83, 0.60, 0.92, 1.00, 0.78, 0.76],
            [0.91, 0.89, 0.91, 0.95, 0.90, 0.77]]

def get_forest_accuracy():
    return [0.86, 0.84, 0.93, 0.95, 0.88, 0.84]

def get_forest_precision():
    return [[0.83, 0.88, 0.94, 0.95, 0.84, 0.78],
            [0.68, 0.53, 0.92, 0.86, 0.88, 0.76],
            [0.90, 1.00, 0.85, 1.00, 1.00, 1.00],
            [1.00, 0.88, 0.95, 0.98, 0.89, 0.93]]

def get_forest_recall():
    return [[0.89, 0.88, 0.92, 0.95, 0.95, 0.90],
            [0.81, 0.53, 0.92, 0.86, 0.65, 0.76],
            [0.60, 0.70, 0.92, 0.86, 0.91, 0.76],
            [0.97, 0.97, 0.95, 1.00, 0.94, 0.87]]

def get_forest_f1():
    return [[0.86, 0.88, 0.93, 0.95, 0.89, 0.84],
            [0.74, 0.53, 0.92, 0.86, 0.75, 0.76],
            [0.72, 0.82, 0.88, 0.92, 0.95, 0.87],
            [0.98, 0.92, 0.95, 0.99, 0.92, 0.90]]

def get_boosting_accuracy():
    return [0.84, 0.82, 0.90, 0.93, 0.89, 0.79]

def get_boosting_precision():
    return [[0.83, 0.88, 0.90, 0.93, 0.90, 0.72],
            [0.68, 0.56, 0.91, 0.79, 0.89, 0.75],
            [0.91, 0.88, 0.79, 0.86, 0.85, 0.83],
            [0.94, 0.86, 0.95, 0.98, 0.89, 0.90]]

def get_boosting_recall():
    return [[0.89, 0.83, 0.90, 0.91, 0.92, 0.85],
            [0.81, 0.60, 0.88, 0.79, 0.70, 0.71],
            [0.67, 0.70, 0.92, 0.86, 1.00, 0.59],
            [0.88, 0.97, 0.91, 1.00, 0.94, 0.87]]

def get_boosting_f1():
    return [[0.86, 0.85, 0.90, 0.92, 0.91, 0.78],
            [0.74, 0.58, 0.89, 0.79, 0.78, 0.73],
            [0.77, 0.78, 0.85, 0.86, 0.92, 0.69],
            [0.91, 0.91, 0.93, 0.99, 0.92, 0.88]]

def get_nn_accuracy():
    return [0.8056, 0.8056, 0.8148, 0.8611, 0.8056, 0.8148]

def get_nn_precision():
    return [[0.90, 0.86, 0.85, 0.87, 0.81, 0.74],
            [0.50, 0.50, 0.68, 0.60, 0.71, 0.78],
            [0.86, 0.78, 0.79, 0.75, 0.90, 0.91],
            [0.78, 0.84, 0.88, 0.96, 0.81, 0.91]]

def get_nn_recall():
    return [[0.84, 0.83, 0.80, 0.79, 0.87, 0.88],
            [0.38, 0.40, 0.62, 0.64, 0.43, 0.67],
            [0.80, 0.70, 0.92, 0.86, 0.82, 0.59],
            [0.97, 1.00, 1.00, 1.00, 0.97, 0.97]]

def get_nn_f1():
    return [[0.87, 0.84, 0.82, 0.83, 0.84, 0.80],
            [0.43, 0.44, 0.65, 0.62, 0.54, 0.72],
            [0.83, 0.74, 0.85, 0.80, 0.86, 0.71],
            [0.86, 0.91, 0.94, 0.98, 0.88, 0.94]]


# graph_accuracy(get_knn_accuracy(), 'KNN')
# graph4(get_knn_precision(), 'KNN', 'Precision')
# graph4(get_knn_recall(), 'KNN', 'Recall')
# graph4(get_knn_f1(), 'KNN', 'F1-score')

# graph_accuracy(get_svm_accuracy(), 'SVM', info='(raw data)')
# graph_accuracy(get_norm_svm_accuracy(), 'SVM', info='(normalized data)')
# graph4(get_svm_precision(), 'SVM', 'Precision', info='(raw data)')
# graph4(get_norm_svm_precision(), 'SVM', 'Precision', info='(normalized data)')
# graph4(get_svm_recall(), 'SVM', 'Recall', info='(raw data)')
# graph4(get_norm_svm_recall(), 'SVM', 'Recall', info='(normalized data)')
# graph4(get_svm_f1(), 'SVM', 'F1-score', info='(raw data)')
# graph4(get_norm_svm_f1(), 'SVM', 'F1-score', info='(normalized data)')

# graph_accuracy(get_tree_accuracy(), 'Decision Tree')
# graph4(get_tree_precision(), 'Decision Tree', 'Precision')
# graph4(get_tree_recall(), 'Decision Tree', 'Recall')
# graph4(get_tree_f1(), 'Decision Tree', 'F1-score')

# graph_accuracy(get_forest_accuracy(), 'Random Forest')
# graph4(get_forest_precision(), 'Random Forest', 'Precision')
# graph4(get_forest_recall(), 'Random Forest', 'Recall')
# graph4(get_forest_f1(), 'Random Forest', 'F1-score')

# graph_accuracy(get_boosting_accuracy(), 'Gradient Boosting')
# graph4(get_boosting_precision(), 'Gradient Boosting', 'Precision')
# graph4(get_boosting_recall(), 'Gradient Boosting', 'Recall')
# graph4(get_boosting_f1(), 'Gradient Boosting', 'F1-score')

# graph_accuracy(get_nn_accuracy(), 'Neural Network')
# graph4(get_nn_precision(), 'Neural Network', 'Precision')
# graph4(get_nn_recall(), 'Neural Network', 'Recall')
graph4(get_nn_f1(), 'Neural Network', 'F1-score')