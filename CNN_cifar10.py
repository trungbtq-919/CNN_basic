import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Activation, Dropout


normalize_value = 255.0
num_class = 10
X_shape = (-1, 32, 32, 3)
epoch = 10
batch_size = 128


def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, y_train, X_test, y_test


def preprocess_data():
    X_train, y_train, X_test, y_test = load_data()

    #normalize data
    X_train = X_train.astype("float32")/normalize_value
    X_test = X_test.astype("float32") / normalize_value

    #one-hot coding
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1 / 6, random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val


def generate_optimizer():
    return keras.optimizers.Adam()


def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer=generate_optimizer(),
                  metrics=['accuracy'])


def generate_model():

    # Check if model already exists
    # if Path('./models/convnet_model.json').is_file():
    #
    #     with open('./models/convnet_model.json') as file:
    #         model = keras.models.model_from_json(json.load(file))
    #         file.close()
    #
    #     # likewise for model weight, if exists load from saved state
    #     if Path('./models/convnet_weights.h5').is_file():
    #         model.load_weights('./models/convnet_weights.h5')
    #
    #     compile_model(model)
    #
    #     return model
    #
    model = Sequential()

    # conv1 32x32x3 ===> 32 filters 3x3x3 =====> 30x30x32
    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=X_shape[1:]))
    model.add(Activation('relu'))

    # conv2 30x30x32 ===> 32 filters 3x3x32 ===> 28x28x32
    model.add(Conv2D(filters=32, kernel_size=(3,3)))
    model.add(Activation('relu'))

    #pooling 28x28x32 ===> 14x14x32
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))

    # conv3 14x14x32 ===> 64 filters 3x3x32 ===> 12x12x64
    model.add(Conv2D(filters=64, kernel_size=(3,3)))
    model.add(Activation('relu'))

    #conv4 12x12x64 ===> 64 filters 3x3x64 ===> 10x10x64
    model.add(Conv2D(filters=64, kernel_size=(3,3)))
    model.add(Activation('relu'))

    #pooling 10x10x64 ===> 5x5x64
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))


    ## fully-connected layer: 5x5x64 ===> 1600
    model.add(Flatten())

    # hidden layer
    model.add(Dense(units=256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    # output layer
    model.add(Dense(units=num_class))
    model.add(Activation('softmax'))


    # with open('./models/cnn_cifar10.json', 'w') as f:
    #     json.dump(model.to_json(), f)
    #     f.close()

    return model


def train_model(model, X_train, y_train, X_val, y_val):

    # compile model
    compile_model(model)

    # train model
    epoch_count = 0
    while epoch_count < epoch:

        epoch_count += 1
        print("######################## start iteration %d ########################".format(epoch_count))
        model.fit(x=X_train, y=y_train, batch_size=batch_size,
                  validation_data=(X_val, y_val), epochs=1)

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
