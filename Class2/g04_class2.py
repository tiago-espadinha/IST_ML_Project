'''
Machine Learning - Image Analysis Problem 2
Authors:
Tiago Simões, 96329
Tomás Fonseca, 66325
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

seed = 21

class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="true_positives", initializer="zeros")
        self.false_positives = self.add_weight(name="false_positives", initializer="zeros")
        self.true_negatives = self.add_weight(name="true_negatives", initializer="zeros")
        self.false_negatives = self.add_weight(name="false_negatives", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > 0.5, tf.bool)

        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), tf.float32))
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.math.logical_not(y_true), y_pred), tf.float32))
        true_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.math.logical_not(y_true), tf.math.logical_not(y_pred)), tf.float32))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, tf.math.logical_not(y_pred)), tf.float32))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.true_negatives.assign_add(true_negatives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        sensitivity = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        specificity = self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())
        balanced_accuracy = (sensitivity + specificity) / 2
        return balanced_accuracy

def class_analysis(Y):
    n_nevu_train = np.sum(Y == 0)
    n_melanoma_train = np.sum(Y == 1)
    n_vascularlesions_train = np.sum(Y == 2)
    n_granulocytes_train = np.sum(Y == 3)
    n_basophils_train = np.sum(Y == 4)
    n_lymphocytes_train = np.sum(Y == 5)

    class_labels = ['Nevu', 'melanoma', 'vascular lesions', 'granulocytes', 'basophils', 'lymphocytes']
    class_counts = [n_nevu_train, n_melanoma_train, n_vascularlesions_train, n_granulocytes_train, n_basophils_train, n_lymphocytes_train]

    print('% of nevu in training set:\t\t', n_nevu_train / len(Y))
    print('% of melanoma in training set:\t\t', n_melanoma_train / len(Y))
    print('% of vascularlesions in training set:\t', n_vascularlesions_train / len(Y))
    print('% of granulocytes in training set:\t', n_granulocytes_train / len(Y))
    print('% of basophils in training set:\t\t', n_basophils_train / len(Y))
    print('% of lymphocytes in training set:\t', n_lymphocytes_train / len(Y), '\n')

    # Visualize class distribution
    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, class_counts)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.show()

def cnn(X_train, Y_train, dropout, bal_mode):
    # Split training data into training and validation sets for the CNN
    X_train_cnn, X_val, Y_train_cnn, Y_val = train_test_split(
        X_train, Y_train, test_size = 0.4, random_state = seed)
    
    print('Balancing mode:\t\t\t', bal_mode)
    X_train_bal, Y_train_bal, bal_mode = balance_data(X_train_cnn, Y_train_cnn, bal_mode)
    print('X_train_bal shape:\t\t', X_train_bal.shape)
    print('Y_train_bal shape:\t\t', Y_train_bal.shape, '\n')

    # Convert labels to one-hot encoding
    Y_train = keras.utils.to_categorical(Y_train_bal, 6)
    Y_val = keras.utils.to_categorical(Y_val, 6)

    # Reshape data
    X_train = X_train_bal.reshape(-1, 28, 28, 3)
    X_val = X_val.reshape(-1, 28, 28, 3)

    print('CNN data:')
    print('X_train shape:\t\t\t', X_train.shape)
    print('Y_train shape:\t\t\t', Y_train.shape)
    print('X_val shape:\t\t\t', X_val.shape)
    print('Y_val shape:\t\t\t', Y_val.shape, '\n')
    # Initialize CNN
    cnn = keras.Sequential()

    cnn.add(keras.layers.RandomFlip("horizontal_and_vertical", seed = seed))
    cnn.add(keras.layers.RandomRotation(0.1, seed = seed))
    cnn.add(keras.layers.RandomTranslation(0.1, 0.1, seed = seed))
    # cnn.add(keras.layers.RandomContrast(0.1, seed = seed))
    # cnn.add(keras.layers.RandomZoom(0.1, seed = seed))


    # 1st Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 8,
        kernel_size = (3, 3),
        activation = 'relu',
        input_shape = (28, 28, 3)))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2)))
    # 2nd Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 16,
        kernel_size = (3, 3),
        activation = 'relu'))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2)))
    # 3rd Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        activation = 'relu'))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2)))
    cnn.add(keras.layers.Dropout(dropout))
    
    # Fully Connected Classifier
    cnn.add(keras.layers.Flatten())
    # 1st Dense Layer
    cnn.add(keras.layers.Dense(
        units = 64,
        activation = 'relu'))
    cnn.add(keras.layers.Dropout(dropout))
    # 2nd Dense Layer
    cnn.add(keras.layers.Dense(
        units = 16,
        activation = 'relu'))
    cnn.add(keras.layers.Dropout(dropout))
    # Output Layer
    cnn.add(keras.layers.Dense(
        units = 6,
        activation = 'softmax'))
    # Compile CNN
    optimizer = keras.optimizers.AdamW(learning_rate = 0.001, weight_decay = 0.001)
    cnn.compile(
        optimizer = optimizer,
        loss = 'categorical_crossentropy',
        metrics = [BalancedAccuracy()])
    
    # cnn_aux = cnn
    # cnn_aux.build(input_shape = X_train.shape)
    # cnn_aux.summary()

    # Train CNN
    callback = keras.callbacks.EarlyStopping(
        monitor = 'val_loss', 
        patience = 20, 
        restore_best_weights = True)
    history = cnn.fit(
        x = X_train, 
        y = Y_train, 
        epochs = 200, 
        batch_size = 32, 
        validation_data = (X_val, Y_val), 
        callbacks = [callback], 
        use_multiprocessing = True)
    
    plt.figure()
    plt.plot(history.epoch, history.history['loss'],
               label = 'Train Loss')
    plt.plot(history.epoch, history.history['val_loss'],
                label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(history.epoch, history.history['balanced_accuracy'],
             label='Train Balanced Accuracy')
    plt.plot(history.epoch, history.history['val_balanced_accuracy'],
             label='Validation Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    
    return cnn

def reshape_data(X):
    X_reshaped = X.reshape(-1, 28, 28, 3)
    return X_reshaped

def balance_data(X_train, Y_train, bal_mode):
    # Number of samples in each class
    n_0 = X_train[Y_train == 0].shape[0]
    n_1 = X_train[Y_train == 1].shape[0]
    n_2 = X_train[Y_train == 2].shape[0]
    n_3 = X_train[Y_train == 3].shape[0]
    n_4 = X_train[Y_train == 4].shape[0]
    n_5 = X_train[Y_train == 5].shape[0]
    print('Number of elements in each class before', bal_mode, 'balancing:')
    print('Class 0:', n_0)
    print('Class 1:', n_1)
    print('Class 2:', n_2)
    print('Class 3:', n_3)
    print('Class 4:', n_4)
    print('Class 5:', n_5, '\n')
    
    if bal_mode == 'smote':
        smote = SMOTE(
            sampling_strategy = 'not majority',
            random_state = seed, 
            n_jobs = -1)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)
    elif bal_mode == 'oversample':
        ros = RandomOverSampler(
            sampling_strategy = 'not majority',
            random_state = seed)
        X_train, Y_train = ros.fit_resample(X_train, Y_train)
    elif bal_mode == 'undersample':
        rus = RandomUnderSampler(
            sampling_strategy = 'not minority',
            random_state = seed)
        X_train, Y_train = rus.fit_resample(X_train, Y_train)
    elif bal_mode == 'class_weight':
        pass
    elif bal_mode == 'None':
        pass
    else:
        print('Invalid balance mode!')
        exit()

    # Number of samples in each class
    n_0 = X_train[Y_train == 0].shape[0]
    n_1 = X_train[Y_train == 1].shape[0]
    n_2 = X_train[Y_train == 2].shape[0]
    n_3 = X_train[Y_train == 3].shape[0]
    n_4 = X_train[Y_train == 4].shape[0]
    n_5 = X_train[Y_train == 5].shape[0]
    print('Number of elements in each class after', bal_mode, 'balancing:')
    print('Class 0:', n_0)
    print('Class 1:', n_1)
    print('Class 2:', n_2)
    print('Class 3:', n_3)
    print('Class 4:', n_4)
    print('Class 5:', n_5, '\n')

    return X_train, Y_train, bal_mode

def conf_mat(Y_val, y_pred):
    cm = confusion_matrix(Y_val, y_pred)
    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(tick_marks, ['Nevu', 'Melanoma', 'Vascular Lesions', 'Granulocytes', 'Basophils', 'Lymphocytes'], rotation=45)
    plt.yticks(tick_marks, ['Nevu', 'Melanoma', 'Vascular Lesions', 'Granulocytes', 'Basophils', 'Lymphocytes'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    # FLAGS:
    # Change path to the folder where the data is located
    path = 'Class2/'
    # Choose between 'smote', 'oversample', 'undersample', 'aug_plus_os' 'double_minority' or 'None'
    bal_mode = 'oversample'
    # Choose between 'log_reg', 'nb', 'svm', 'cnn'
    chosen_model = 'cnn'
    
    print('Chosen model:\t\t\t', chosen_model, '\n')

    # Load datasets
    X_train_og = np.load(path + 'Xtrain_Classification2.npy')
    Y_train_og = np.load(path + 'ytrain_Classification2.npy')
    X_test_og = np.load(path + 'Xtest_Classification2.npy')

    print('OG X_train shape:\t\t', X_train_og.shape)
    print('OG Y_train shape:\t\t', Y_train_og.shape)
    print('OG X_test shape:\t\t', X_test_og.shape, '\n')

    # class_analysis(Y_train_og)

    # Normalize data to [0, 1]
    X_train = X_train_og / 255
    X_test = X_test_og / 255

    # Split training data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train_og, test_size = 0.2, random_state = seed)
    
    print('X_train shape:\t\t\t', X_train.shape)
    print('Y_train shape:\t\t\t', Y_train.shape)
    print('X_val shape:\t\t\t', X_val.shape)
    print('Y_val shape:\t\t\t', Y_val.shape, '\n')

    if chosen_model == 'cnn':
        print('#################################################################')
        print('CNN:\n')

        X_val_reshaped = reshape_data(X_val)
        X_test_reshaped = reshape_data(X_test)

        print('X_val_reshaped shape:\t\t', X_val_reshaped.shape)
        print('X_test_reshaped shape:\t\t', X_test_reshaped.shape, '\n')

        cnn_model = cnn(X_train, Y_train, dropout = 0.2, bal_mode = bal_mode)
        cnn_model.summary()
        y_pred = np.argmax(cnn_model.predict(X_val_reshaped), axis = 1)
        conf_mat(Y_val, y_pred)
        print('Balanced accuracy score:', balanced_accuracy_score(Y_val, y_pred))

        # np.save(path + 'y_pred.npy', y_pred)
        # np.save(path + 'Y_val.npy', Y_val)
        y_test = np.argmax(cnn_model.predict(X_test_reshaped), axis = 1)
        np.save(path + 'ytest_Classification2.npy', y_test)
        plt.show()
    else:
        print('#################################################################')
        print('Chose a valid model!')
    
    print('#################################################################')


if __name__ == "__main__":
    main()