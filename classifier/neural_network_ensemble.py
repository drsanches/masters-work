from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.layers import Average, Add, Maximum
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import numpy as np
import utils
import os


X, Y = utils.get_dataset('dataset_threshold_100_shift_05_2.txt')

X = utils.remove_feature(X, 17)
X = utils.remove_feature(X, 16)
X = utils.remove_feature(X, 15)
X = utils.remove_feature(X, 14)
X = utils.remove_feature(X, 13)
X = utils.remove_feature(X, 12)
X = utils.remove_feature(X, 11)
X = utils.remove_feature(X, 10)
X = utils.remove_feature(X, 9)
X = utils.remove_feature(X, 8)
X = utils.remove_feature(X, 7)
X = utils.remove_feature(X, 6)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
input_shape = X[0].shape
output_num = Y.shape[1]

input_layer = Input(shape=input_shape)

ensemble1 = Dense(64, activation='sigmoid')(input_layer)
ensemble1 = BatchNormalization()(ensemble1)
ensemble1 = Dense(128, activation='sigmoid')(ensemble1)
ensemble1 = BatchNormalization()(ensemble1)
ensemble1 = Dense(256, activation='sigmoid')(ensemble1)
ensemble1 = BatchNormalization()(ensemble1)
ensemble1 = Dense(output_num, activation='softmax')(ensemble1)

ensemble2 = Dense(128, activation='sigmoid')(input_layer)
ensemble2 = BatchNormalization()(ensemble2)
ensemble2 = Dense(256, activation='sigmoid')(ensemble2)
ensemble2 = BatchNormalization()(ensemble2)
ensemble2 = Dense(output_num, activation='softmax')(ensemble2)

out = Maximum()([ensemble1, ensemble2])
model = Model(inputs=input_layer, outputs=out)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
history = model.fit(X_train, Y_train,
                    batch_size=32,
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks=[tensorboard])

scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy on test data: %.2f%%" % (scores[1]*100))

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model, to_file=str(scores[1]) + '.png', show_shapes=True)
