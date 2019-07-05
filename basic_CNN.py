

import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


def load_data_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.reshape(X_train, (60000, 28, 28, 1))
    X_test = np.reshape(X_test, (10000, 28, 28, 1))
    return X_train, y_train, X_test, y_test

# for i in range(10):
#     img_name = "plt" + str(i)
#     two_d_img = plt.imshow(np.reshape(X_train[i], (28, 28)).astype(np.uint8), interpolation="nearest")
#     plt.savefig(img_name+".pdf")
#     print(y_train[i])


# iprint(X_train.shape, X_test.shape)



#one hot encoding


def preprocess_data():

    X_train, y_train, X_test, y_test = load_data_mnist()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    X_train = np.reshape(X_train, (60000, 28, 28, 1))
    X_test = np.reshape(X_test, (10000, 28, 28, 1))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/6, random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val

# print(y_train[0], y_test[0])

def generate_model():
    # create model
    model = Sequential()

    # add model layers
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))

    return model



def train_model(model, X_train, y_train, X_val, y_val):

    # complie model with accuracy and loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train model
    hist = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=3)
    with open('train_mnist.json', 'w') as f:
        json.dump(hist.history, f)

    return model


def predict_and_evaluate(model, X_test, y_test):

    y_test_real = model.predict(X_test)
    y_pred = np.zeros_like(y_test_real)
    y_pred[np.arange(len(y_test_real)), y_test_real.argmax(1)] = 1

    return accuracy_score(y_pred, y_test)


def main():
    X_train, y_train, X_test, y_test, X_val, y_val = preprocess_data()
    model = generate_model()
    model = train_model(model, X_train, y_train, X_val, y_val)
    print(predict_and_evaluate(model, X_test, y_test))

if __name__ == "__main__":
    main()
