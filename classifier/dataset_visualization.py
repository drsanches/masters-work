import matplotlib.pyplot as plt
import utils


def visualize(X, x_index, y_index, Y):
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    for i in range(len(X)):
        if Y[i] == 0:
            X1.append(X[i][x_index])
        elif Y[i] == 1:
            X2.append(X[i][x_index])
        elif Y[i] == 2:
            X3.append(X[i][x_index])
        elif Y[i] == 3:
            X4.append(X[i][x_index])    
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    for i in range(len(X)):
        if Y[i] == 0:
            Y1.append(X[i][y_index])
        elif Y[i] == 1:
            Y2.append(X[i][y_index])
        elif Y[i] == 2:
            Y3.append(X[i][y_index])
        elif Y[i] == 3:
            Y4.append(X[i][y_index])

    fig, ax = plt.subplots()
    ax.plot(X1, Y1, 'ro', label = 'Exchange')
    ax.plot(X2, Y2, 'bo', label = 'ICO wallets')
    ax.plot(X3, Y3, 'go', label = 'Mining')
    ax.plot(X4, Y4, 'yo', label = 'Token contract')
    ax.legend()
    plt.show()

X, Y = utils.get_dataset('dataset_threshold_100.txt')
Y = utils.convert_Y_to_class_numbers(Y)

print(X.shape)

visualize(X, 6, 9, Y)
