from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.layers import Average, Add, Maximum, Concatenate
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import utils
import os
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 


def get_part(X, n):
    tmp = X
    if (n == 1):
        tmp = utils.remove_feature(tmp, 17)
        tmp = utils.remove_feature(tmp, 16)
        tmp = utils.remove_feature(tmp, 15)
        tmp = utils.remove_feature(tmp, 14)
        tmp = utils.remove_feature(tmp, 13)
        tmp = utils.remove_feature(tmp, 12)
        tmp = utils.remove_feature(tmp, 11)
        tmp = utils.remove_feature(tmp, 10)
        tmp = utils.remove_feature(tmp, 9)
        tmp = utils.remove_feature(tmp, 8)
        tmp = utils.remove_feature(tmp, 7)
        tmp = utils.remove_feature(tmp, 6)
    elif (n == 2):
        tmp = utils.remove_feature(tmp, 17)
        tmp = utils.remove_feature(tmp, 16)
        tmp = utils.remove_feature(tmp, 15)
        tmp = utils.remove_feature(tmp, 14)
        tmp = utils.remove_feature(tmp, 13)
        tmp = utils.remove_feature(tmp, 12)
        tmp = utils.remove_feature(tmp, 5)
        tmp = utils.remove_feature(tmp, 4)
        tmp = utils.remove_feature(tmp, 3)
        tmp = utils.remove_feature(tmp, 2)
        tmp = utils.remove_feature(tmp, 1)
        tmp = utils.remove_feature(tmp, 0)
    else:
        tmp = utils.remove_feature(tmp, 11)
        tmp = utils.remove_feature(tmp, 10)
        tmp = utils.remove_feature(tmp, 9)
        tmp = utils.remove_feature(tmp, 8)
        tmp = utils.remove_feature(tmp, 7)
        tmp = utils.remove_feature(tmp, 6)
        tmp = utils.remove_feature(tmp, 5)
        tmp = utils.remove_feature(tmp, 4)
        tmp = utils.remove_feature(tmp, 3)
        tmp = utils.remove_feature(tmp, 2)
        tmp = utils.remove_feature(tmp, 1)
        tmp = utils.remove_feature(tmp, 0)
    return tmp


def create_model(input_layer):
    x = Dense(64, activation='sigmoid')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x


# X_train, Y_train = utils.get_dataset('train_threshold_100.txt')
# X_test, Y_test = utils.get_dataset('test_threshold_100.txt')

random_state=50
# random_state=40
# random_state=30
# random_state=20
# random_state=10
# random_state=0
X, Y = utils.get_dataset('dataset_threshold_100.txt')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

print("Train: " + str(X_train.shape[0]))
print("Test: " + str(X_test.shape[0]))
input_shape = (6, )
output_num = 4

# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(utils.convert_Y_to_class_numbers(Y_train)),
#                                                  utils.convert_Y_to_class_numbers(Y_train))
# print(class_weights)

input1 = Input(shape=input_shape)
input2 = Input(shape=input_shape)
input3 = Input(shape=input_shape)
input_layers = [input1, input2, input3]
out = Concatenate()([create_model(input1), create_model(input2), create_model(input3)])
out = Dense(512, activation='sigmoid')(out)
out = BatchNormalization()(out)
out = Dropout(0.2)(out)
out = Dense(256, activation='sigmoid')(out)
out = BatchNormalization()(out)
out = Dropout(0.5)(out)
out = Dense(output_num, activation='softmax')(out)
model = Model(inputs=input_layers, outputs=out)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model, to_file='nn2.png', show_shapes=True)

# tensorboard = TensorBoard(log_dir='./logs/test' + str(random_state), write_graph=True)
tensorboard = TensorBoard(log_dir='./logs/tmp' + str(random_state), write_graph=True)
history = model.fit([get_part(X_train, 1), get_part(X_train, 2), get_part(X_train, 3)], Y_train,
                    batch_size=32,
                    epochs=100,
                    verbose=1,
                    # class_weight=class_weights,
                    validation_data=([get_part(X_test, 1), get_part(X_test, 2), get_part(X_test, 3)], Y_test),
                    callbacks=[tensorboard])

scores = model.evaluate([get_part(X_test, 1), get_part(X_test, 2), get_part(X_test, 3)], Y_test, verbose=1)
print("Accuracy on test data: %.2f%%" % (scores[1]*100))

pred = model.predict([get_part(X_test, 1), get_part(X_test, 2), get_part(X_test, 3)])
pred = utils.convert_Y_to_class_numbers(pred)
Y_test = utils.convert_Y_to_class_numbers(Y_test)
print(classification_report(Y_test, pred))
print(confusion_matrix(Y_test, pred))