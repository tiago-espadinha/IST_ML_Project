'''
Machine Learning - Image Analysis Problem 1
Authors:
Tiago Simões, 96329
Tomás Fonseca, 66325
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras


# Change path to the folder where the data is located
path = 'Class1/'



def main():

    # Load datasets
    X_train = np.load(path + 'Xtrain_Classification1.npy')
    Y_train = np.load(path + 'ytrain_Classification1.npy')
    X_test = np.load(path + 'Xtest_Classification1.npy')

    # Normalize data to [0, 1]
    X_train = X_train / 255
    X_test = X_test / 255

    # Split training data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size = 0.01, random_state = 41)

    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_val.shape)
    # print(Y_val.shape)

    # Reshape data to 28x28x3
    X_train_reshaped = X_train.reshape(-1, 28, 28, 3)
    X_val_reshaped = X_val.reshape(-1, 28, 28, 3)
    X_test_reshaped = X_test.reshape(-1, 28, 28, 3)

    cnn = keras.Sequential()

    # 1st Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        activation = 'relu',
        input_shape = (28, 28, 3)
        ))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2),
        strides = (2, 2)
        ))
    
    # 2nd Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        activation = 'relu'
        ))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2),
        strides = (2, 2)
        ))
    
    # Fully Connected Classifier
    cnn.add(keras.layers.Flatten())

    # 1st Dense Layer
    cnn.add(keras.layers.Dense(
        units = 1024,
        activation = 'relu'
        ))
    cnn.add(keras.layers.Dropout(0.5))

    # 2nd Dense Layer
    cnn.add(keras.layers.Dense(
        units = 256,
        activation = 'relu'
        ))
    cnn.add(keras.layers.Dropout(0.5))

    # 3rd Dense Layer
    cnn.add(keras.layers.Dense(
        units = 64,
        activation = 'relu'
        ))
    cnn.add(keras.layers.Dropout(0.5))

    # Output Layer
    cnn.add(keras.layers.Dense(
        units = 2,
        activation = 'softmax'
        ))
    
    # Compile CNN
    cnn.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
        )
    
    # Train CNN
    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss', 
        patience = 20, 
        restore_best_weights = True)
    history = cnn.fit(
        x = X_train_reshaped,
        y = Y_train,
        epochs = 100,
        batch_size = 32,
        validation_data = (X_val_reshaped, Y_val),
        callbacks = [callback]
        )
    
    # Save model
    cnn.save('cnn.h5')

    # Evaluate CNN
    loss, acc = cnn.evaluate(
        x = X_val_reshaped,
        y = Y_val
        )
    
    print('Validation accuracy: {:.2f}%'.format(acc))
    print('Validation loss: {:.2f}'.format(loss))
    
    # Plot training and validation accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.figure()
    plt.plot(epochs, acc, 'b', label = 'Training accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()



if __name__ == "__main__":
    main()