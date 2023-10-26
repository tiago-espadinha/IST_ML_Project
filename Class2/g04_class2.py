'''
Machine Learning - Image Analysis Problem 1
Authors:
Tiago Simões, 96329
Tomás Fonseca, 66325
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow import keras

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

seed = None

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

def logistic_regression(X_train, Y_train, grid_search, bal_mode):
    if bal_mode == 'class_weight':
        class_weight = 'balanced'
    else:
        class_weight = None

    if grid_search:
        # Define hyperparameter grid
        param_grid = {
            'C': [0.01, 0.1, 1], 
            "penalty": ["elasticnet"], 
            "l1_ratio": [0.1, 0.2, 0.3],
            "max_iter": [10000], 
            "solver": ["saga"]}
    else:
        # Best hyperparameters found
        param_grid = {
            'C': [0.1], 
            "penalty": ["elasticnet"], 
            "l1_ratio": [0.1],
            "max_iter": [1000], 
            "solver": ["saga"]}
        
    log_reg = GridSearchCV(
        estimator = LogisticRegression(class_weight = class_weight), 
        param_grid = param_grid, 
        cv = 5, 
        scoring = 'balanced_accuracy', 
        n_jobs = -1)
    
    log_reg.fit(X_train, Y_train)
    
    return log_reg

def naive_bayes(X_train, Y_train, nb_mode, bal_mode):
    if bal_mode == 'class_weight':
        print('Class weight not supported for Naive Bayes!')
        exit()

    if nb_mode == 'gaussian':
        nb = GaussianNB().fit(X_train, Y_train)
    elif nb_mode == 'multinomial':
        nb = MultinomialNB().fit(X_train, Y_train)
    elif nb_mode == 'complement':
        nb = ComplementNB().fit(X_train, Y_train)
    elif nb_mode == 'bernoulli':
        nb = BernoulliNB().fit(X_train, Y_train)
    
    return nb, nb_mode

def svm(X_train, Y_train, grid_search, bal_mode):
    if bal_mode == 'class_weight':
        class_weight = 'balanced'
    else:
        class_weight = None

    if grid_search:
        param_grid = {
            'C': [0.1, 1, 10], 
            "kernel": ["linear", "poly", "rbf"], 
            "degree": [2, 3, 4], 
            "gamma": ["scale", "auto"]
            }
    else:
        param_grid = {
            'C': [0.1], 
            "kernel": ["linear"], 
            "degree": [3], 
            "gamma": ["scale"]
            }
        
    svm = GridSearchCV(
        estimator = SVC(class_weight = class_weight), 
        param_grid = param_grid, 
        cv = 5, 
        scoring = 'balanced_accuracy', 
        n_jobs = -1)
    
    svm.fit(X_train, Y_train)

    return svm

def cnn(X_train, Y_train, dropout, bal_mode):
    if bal_mode == 'class_weight':
        print('Class weight not supported for CNN!')
        exit()

    # Split training data into training and validation sets for the CNN
    X_train_cnn, X_val, Y_train_cnn, Y_val = train_test_split(
        X_train, Y_train, test_size = 0.3, random_state = seed)
    
    print('Balancing mode:\t\t\t', bal_mode)
    X_train_bal, Y_train_bal, bal_mode = balance_data(X_train_cnn, Y_train_cnn, bal_mode)
    print('X_train_bal shape:\t\t', X_train_bal.shape)
    print('Y_train_bal shape:\t\t', Y_train_bal.shape, '\n')

    # Convert labels to one-hot encoding
    Y_train = keras.utils.to_categorical(Y_train_bal, 2)
    Y_val = keras.utils.to_categorical(Y_val, 2)

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
    cnn.add(keras.layers.RandomZoom(0.1, seed = seed))

    # 1st Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 16,
        kernel_size = (3, 3),
        activation = 'relu',
        input_shape = (28, 28, 3)))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2)))
    # 2nd Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        activation = 'relu'))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2)))
    # 3rd Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        activation = 'relu'))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2)))
    cnn.add(keras.layers.Dropout(dropout))
    
    # Fully Connected Classifier
    cnn.add(keras.layers.Flatten())
    # 1st Dense Layer
    cnn.add(keras.layers.Dense(
        units = 128,
        activation = 'relu'))
    # # 2nd Dense Layer
    # cnn.add(keras.layers.Dense(
    #     units = 16,
    #     activation = 'relu'))
    # # 3rd Dense Layer
    # cnn.add(keras.layers.Dense(
    #     units = 16,
    #     activation = 'relu'))
    # Output Layer
    cnn.add(keras.layers.Dense(
        units = 6,
        activation = 'softmax'))
    # Compile CNN
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, clipnorm = 1)
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
    print('Number of elements in each before', bal_mode, 'balancing:')
    print('Class 0:', n_0)
    print('Class 1:', n_1, '\n')
    
    if bal_mode == 'smote':
        smote = SMOTE(
            sampling_strategy = 'minority',
            random_state = seed, 
            n_jobs = -1)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)
        # Number of samples in each class
        n_0 = X_train[Y_train == 0].shape[0]
        n_1 = X_train[Y_train == 1].shape[0]
        print('Number of elements in each class after SMOTE:')
        print('Class 0:', n_0)
        print('Class 1:', n_1, '\n')
    elif bal_mode == 'oversample':
        ros = RandomOverSampler(
            sampling_strategy = 'minority',
            random_state = seed)
        X_train, Y_train = ros.fit_resample(X_train, Y_train)
        print('Number of elements in each class after oversampling:')
        print('Class 0:', X_train[Y_train == 0].shape[0])
        print('Class 1:', X_train[Y_train == 1].shape[0], '\n')
    elif bal_mode == 'undersample':
        rus = RandomUnderSampler(
            sampling_strategy = 'majority',
            random_state = seed)
        X_train, Y_train = rus.fit_resample(X_train, Y_train)
        # Number of samples in each class
        n_0 = X_train[Y_train == 0].shape[0]
        n_1 = X_train[Y_train == 1].shape[0]
        print('Number of elements in each class after undersampling:')
        print('Class 0:', n_0)
        print('Class 1:', n_1, '\n')
    elif bal_mode == 'class_weight':
        pass
    elif bal_mode == 'aug_plus_os':
        # Augments and oversamples the minority class

        # Reshape data
        X_train = X_train.reshape(-1, 28, 28, 3)

        # Divide data into classes
        X_train_0 = X_train[Y_train == 0]
        X_train_1 = X_train[Y_train == 1]
        Y_train_0 = Y_train[Y_train == 0]
        Y_train_1 = Y_train[Y_train == 1]

        # Number of samples in each class
        n_0 = X_train_0.shape[0]
        n_1 = X_train_1.shape[0]

        # Number of batches to generate
        batch_size = 32
        n_batches = int((n_0 - n_1) * 0.8 / batch_size)

        datagen = keras.preprocessing.image.ImageDataGenerator(
            # rotation_range = 10,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            # shear_range = 0.1,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest')
        
        datagen.fit(X_train_1, seed = seed)

        data_aug = datagen.flow(
            x = X_train_1,
            y = Y_train_1,
            batch_size = batch_size,
            seed = seed)

        # Generate n_batches of augmented data
        X_train_1_aug = []
        Y_train_1_aug = []
        for i in range(n_batches):
            X_batch, Y_batch = next(data_aug)
            X_train_1_aug.append(X_batch)
            Y_train_1_aug.append(Y_batch)

        # Reshape augmented data
        X_train_1_aug = np.concatenate(X_train_1_aug)
        Y_train_1_aug = np.concatenate(Y_train_1_aug)

        # Normalize augmented data to [0, 1]
        X_train_1_aug = X_train_1_aug / 255

        # Add augmented data to original data
        X_train = np.concatenate((X_train, X_train_1_aug))
        Y_train = np.concatenate((Y_train, Y_train_1_aug))

        shuffler = np.random.permutation(len(Y_train))
        X_train = X_train[shuffler]
        Y_train = Y_train[shuffler]

        # # Plot augmented images
        # plt.figure(figsize = (10, 10))
        # for i in range(100):
        #     plt.subplot(10, 10, i + 1)
        #     plt.imshow(X_train_1_aug[i])
        #     plt.title('Class: 1')
        #     plt.axis('off')
        # plt.show()

        # Reshape data
        X_train = X_train.reshape(-1, 28 * 28 * 3)

        # Number of samples in each class
        n_0 = X_train[Y_train == 0].shape[0]
        n_1 = X_train[Y_train == 1].shape[0]
        print('Number of elements in each class after oversampling with augmentation:')
        print('Class 0:', n_0)
        print('Class 1:', n_1, '\n')
    elif bal_mode == 'double_minority':
        # Doubles minority class

        # Divide data into classes
        X_train_0 = X_train[Y_train == 0]
        X_train_1 = X_train[Y_train == 1]
        Y_train_0 = Y_train[Y_train == 0]
        Y_train_1 = Y_train[Y_train == 1]

        # Doubles minority class
        X_train_1 = np.concatenate((X_train_1, X_train_1))
        Y_train_1 = np.concatenate((Y_train_1, Y_train_1))


        # Add oversampled data to original data
        X_train = np.concatenate((X_train_0, X_train_1))
        Y_train = np.concatenate((Y_train_0, Y_train_1))

        shuffler = np.random.permutation(len(Y_train))
        X_train = X_train[shuffler]
        Y_train = Y_train[shuffler]

        # Number of samples in each class
        n_0 = X_train[Y_train == 0].shape[0]
        n_1 = X_train[Y_train == 1].shape[0]
        print('Number of elements in each class after doubling minority:')
        print('Class 0:', n_0)
        print('Class 1:', n_1, '\n')
    elif bal_mode == 'None':
        pass
    else:
        print('Invalid balance mode!')
        exit()

    return X_train, Y_train, bal_mode

def main():
    # FLAGS:
    # Change path to the folder where the data is located
    path = 'Class2/'
    # Set to True to perform grid search for hyperparameter tuning
    grid_search = False
    # Choose between 'smote', 'oversample', 'undersample', 'class_weight', 'aug_plus_os' 'double_minority' or 'None'
    bal_mode = 'oversample'
    # Choose between 'log_reg', 'nb', 'svm', 'cnn'
    chosen_model = 'cnn'
    
    print('Grid search:\t\t\t', grid_search)
    print('Chosen model:\t\t\t', chosen_model, '\n')

    # Load datasets
    X_train_og = np.load(path + 'Xtrain_Classification2.npy')
    Y_train_og = np.load(path + 'ytrain_Classification2.npy')
    X_test_og = np.load(path + 'Xtest_Classification2.npy')

    print('OG X_train shape:\t\t', X_train_og.shape)
    print('OG Y_train shape:\t\t', Y_train_og.shape)
    print('OG X_test shape:\t\t', X_test_og.shape, '\n')

    n_nevu_train = np.sum(Y_train_og == 0)
    n_melanoma_train = np.sum(Y_train_og == 1)

    print('% of nevu in training set:\t', n_nevu_train / len(Y_train_og))
    print('% of melanoma in training set:\t', n_melanoma_train / len(Y_train_og), '\n')

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

    if chosen_model == 'log_reg':
        print('#################################################################')
        print('Logistic Regression:\n')
        
        print('Balancing mode:\t\t\t', bal_mode)
        X_train_bal, Y_train_bal, bal_mode = balance_data(X_train, Y_train, bal_mode)
        print('X_train_bal shape:\t\t', X_train_bal.shape)
        print('Y_train_bal shape:\t\t', Y_train_bal.shape, '\n')
        log_reg_model = logistic_regression(X_train_bal, Y_train_bal, grid_search, bal_mode)

        if grid_search:
            print('Best parameters:', log_reg_model.best_params_)
        
        y_pred = log_reg_model.predict(X_val)
        print('Balanced accuracy score:', balanced_accuracy_score(Y_val, y_pred))

        # y_test = log_reg_model.predict(X_test)
        # np.save('ytest_Classification2.npy', y_test)
    
    elif chosen_model == 'nb':
        print('#################################################################')
        print('Naive Bayes:\n')
        
        print('Balancing mode:\t\t\t', bal_mode)
        X_train_bal, Y_train_bal, bal_mode = balance_data(X_train, Y_train, bal_mode)
        print('X_train_bal shape:\t\t', X_train_bal.shape)
        print('Y_train_bal shape:\t\t', Y_train_bal.shape, '\n')
        # NB Modes: 'gaussian', 'multinomial', 'complement', 'bernoulli'
        nb_model, nb_mode = naive_bayes(X_train_bal, Y_train_bal, nb_mode = 'gaussian', bal_mode = bal_mode)
        
        print('NB Mode:\t\t\t', nb_mode)
        y_pred = nb_model.predict(X_val)
        print('Balanced accuracy score:\t', balanced_accuracy_score(Y_val, y_pred))

        # y_test = nb_model.predict(X_test)
        # np.save('ytest_Classification2.npy', y_test)
    
    elif chosen_model == 'svm':
        print('#################################################################')
        print('SVM:\n')
        
        print('Balancing mode:\t\t\t', bal_mode)
        X_train_bal, Y_train_bal, bal_mode = balance_data(X_train, Y_train, bal_mode)
        print('X_train_bal shape:\t\t', X_train_bal.shape)
        print('Y_train_bal shape:\t\t', Y_train_bal.shape, '\n')
        svm_model = svm(X_train_bal, Y_train_bal, grid_search)
        
        if grid_search:
            print('Best parameters:', svm_model.best_params_)
        
        y_pred = svm_model.predict(X_val)

        print('Balanced accuracy score:', balanced_accuracy_score(Y_val, y_pred))

        # y_test = svm_model.predict(X_test)
        # np.save('ytest_Classification1.npy', y_test)
    
    elif chosen_model == 'cnn':
        print('#################################################################')
        print('CNN:\n')

        X_val_reshaped = reshape_data(X_val)
        X_test_reshaped = reshape_data(X_test)

        print('X_val_reshaped shape:\t\t', X_val_reshaped.shape)
        print('X_test_reshaped shape:\t\t', X_test_reshaped.shape, '\n')

        cnn_model = cnn(X_train, Y_train, dropout = 0.2, bal_mode = bal_mode)
        cnn_model.summary()
        y_pred = np.argmax(cnn_model.predict(X_val_reshaped), axis = 1)
        print('Balanced accuracy score:', balanced_accuracy_score(Y_val, y_pred))

        #np.save(path + 'y_pred.npy', y_pred)
        #np.save(path + 'Y_val.npy', Y_val)
        y_test = np.argmax(cnn_model.predict(X_test_reshaped), axis = 1)
        np.save(path + 'ytest_Classification2.npy', y_test)
        plt.show()
    else:
        print('#################################################################')
        print('Chose a valid model!')
    
    print('#################################################################')

    

if __name__ == "__main__":
    main()