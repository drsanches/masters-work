from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np
import json

DATASET_PATH = '../dataset/dataset_threshold_100_shift_05_2.txt'

dataset = []
with open(DATASET_PATH) as f:
    dataset = json.load(f)

X = np.array(dataset[0])
Y = np.array(dataset[1])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
input_shape = X.shape[1]
output_shape = Y.shape[1]

print(X_train.shape)
print(X_test.shape)


model = Sequential()
model.add(Dense(128, activation='sigmoid', input_dim=input_shape))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(output_shape, activation='softmax'))

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

# Best: 74.82%

# Without contracts: 73.59%
# Without dropout: 78.93%