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
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scikeras.wrappers import KerasClassifier

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

def cnn(X_train, Y_train, dropout):
    # Split training data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size = 0.1, random_state = 21)
    
    
    # Initialize CNN
    cnn = keras.Sequential()

    # Add data augmentation
    cnn.add(keras.layers.RandomFlip('horizontal'))
    cnn.add(keras.layers.RandomRotation(0.2))
    cnn.add(keras.layers.RandomContrast(0.2))
    cnn.add(keras.layers.RandomTranslation(0.2, 0.2))
    cnn.add(keras.layers.RandomZoom(0.2))

    # 1st Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        activation = 'relu',
        input_shape = (28, 28, 3)))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2),
        strides = (2, 2)))
    # 2nd Convolutional Layer
    cnn.add(keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        activation = 'relu'))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size = (2, 2),
        strides = (2, 2)))
    
    # Fully Connected Classifier
    cnn.add(keras.layers.Flatten())
    # 1st Dense Layer
    cnn.add(keras.layers.Dense(
        units = 1024,
        activation = 'relu'))
    cnn.add(keras.layers.Dropout(dropout))
    # 2nd Dense Layer
    cnn.add(keras.layers.Dense(
        units = 256,
        activation = 'relu'))
    cnn.add(keras.layers.Dropout(dropout))
    # 3rd Dense Layer
    cnn.add(keras.layers.Dense(
        units = 64,
        activation = 'relu'))
    cnn.add(keras.layers.Dropout(dropout))
    # Output Layer
    cnn.add(keras.layers.Dense(
        units = 2,
        activation = 'softmax'))
    # Compile CNN
    cnn.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])
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
    
    return cnn, history

def create_cnn(filters, kernel_size, pool_size, strides):
    cnn = keras.Sequential()

    # Add data augmentation
    cnn.add(keras.layers.RandomFlip('horizontal'))
    cnn.add(keras.layers.RandomRotation(0.2))
    cnn.add(keras.layers.RandomContrast(0.2))
    cnn.add(keras.layers.RandomTranslation(0.2, 0.2))
    cnn.add(keras.layers.RandomZoom(0.2))

    # 1st Convolutional Layer
    cnn.add(keras.layers.Conv2D(filters, kernel_size, activation = 'relu', input_shape = (28, 28, 3)))
    cnn.add(keras.layers.MaxPooling2D(pool_size, strides))

    # 2nd Convolutional Layer
    cnn.add(keras.layers.Conv2D(filters, kernel_size, activation = 'relu'))
    cnn.add(keras.layers.MaxPooling2D(pool_size, strides))
    
    # Fully Connected Classifier
    cnn.add(keras.layers.Flatten())

    # 1st Dense Layer
    cnn.add(keras.layers.Dense(units = 1024, activation = 'relu'))
    cnn.add(keras.layers.Dropout(0.2))

    # 2nd Dense Layer
    cnn.add(keras.layers.Dense(units = 256, activation = 'relu'))
    cnn.add(keras.layers.Dropout(0.2))

    # 3rd Dense Layer
    cnn.add(keras.layers.Dense(units = 64, activation = 'relu'))
    cnn.add(keras.layers.Dropout(0.2))

    # Output Layer
    cnn.add(keras.layers.Dense(units = 2, activation = 'softmax'))

    # Compile Model
    cnn.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])

    return cnn

def cnn_grid(X_train, Y_train, dropout, grid_search):
    # Split training data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size = 0.1, random_state = 21)
    
    callback = keras.callbacks.EarlyStopping(
            monitor = 'val_loss', 
            patience = 20, 
            restore_best_weights = True)
    
    #dictionary of model.fit
    fit_params = {
        # 'x': X_train, 
        # 'y': Y_train,
        'epochs': 200,
        'batch_size': 32,
        'validation_data': (X_val, Y_val),
        'callbacks': [callback],
        'use_multiprocessing': True
    }

    if grid_search:
        # Define hyperparameter grid
        param_grid = {
            'filters': [16, 32, 64],
            'kernel_size': [(3, 3), (5, 5)],
            'pool_size': [(2, 2), (3, 3)],
            'strides': [(2, 2), (3, 3)],
            #'dropout': [0.2, 0.3]
        }

        cnn_model = KerasClassifier(build_fn = create_cnn)

        cnn_grid = GridSearchCV(
            estimator = cnn_model, 
            param_grid = param_grid, 
            cv = 5, 
            #scoring = 'balanced_accuracy', 
            #n_jobs = -1
            )
        
        cnn_grid_result = cnn_grid.fit(X_train, Y_train, **fit_params)

        print(f"Best Parameters: {cnn_grid_result.best_params_}")
        print(f"Best Accuracy: {cnn_grid_result.best_score_}")

        best_model = cnn_grid_result.best_estimator_
        test_accuracy = best_model.score(X_val, Y_val)

    else:
        param_grid = {
            'filters': 32,
            'kernel_size': (3, 3),
            'pool_size': (2, 2),
            'strides': (2, 2),
            #'dropout': dropout
        }
        cnn_model = create_cnn(**param_grid)
    
        # Train CNN
        cnn_grid_result = cnn_model.fit(X_train, Y_train, **fit_params)
            
    
    return cnn_model, cnn_grid_result

def reshape_data(X_train, X_val, X_test):
    X_train_reshaped = X_train.reshape(-1, 28, 28, 3)
    X_val_reshaped = X_val.reshape(-1, 28, 28, 3)
    X_test_reshaped = X_test.reshape(-1, 28, 28, 3)

    return X_train_reshaped, X_val_reshaped, X_test_reshaped

def balance_data(X_train, Y_train, bal_mode):
    if bal_mode == 'smote':
        smote = SMOTE(
            sampling_strategy = 'minority',
            random_state = 21, 
            n_jobs = -1)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)
    elif bal_mode == 'oversample':
        ros = RandomOverSampler(
            sampling_strategy = 'minority',
            random_state = 21)
        X_train, Y_train = ros.fit_resample(X_train, Y_train)
    elif bal_mode == 'undersample':
        rus = RandomUnderSampler(
            sampling_strategy = 'majority',
            random_state = 21)
        X_train, Y_train = rus.fit_resample(X_train, Y_train)
    elif bal_mode == 'class_weight':
        pass
    elif bal_mode == 'alt_oversample':
        print('alt_oversample not implemented yet!')
    elif bal_mode == 'None':
        pass
    else:
        print('Invalid balance mode!')
        exit()
        
    return X_train, Y_train, bal_mode

def balanced_accuracy(Y_true, Y_pred):
    TP = tf.keras.metrics.TruePositives()
    TN = tf.keras.metrics.TrueNegatives()
    FP = tf.keras.metrics.FalsePositives()
    FN = tf.keras.metrics.FalseNegatives()

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    bal_accuracy = (sensitivity + specificity) / 2

    return bal_accuracy

def plot_loss(history):
  # Use a log scale on y-axis to show the wide range of values.
  plt.semilogy(history.epoch, history.history['loss'],
               label = 'Train Loss')
  plt.semilogy(history.epoch, history.history['val_loss'],
               label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

def main():
    # FLAGS:
    path = 'Class1/'#'lab3/' # Change path to the folder where the data is located
    grid_search = True # Set to True to perform grid search for hyperparameter tuning
    bal_mode = 'oversample' # Choose between 'smote', 'oversample', 'undersample', 'class_weight' or 'None'
    chosen_model = 'cnn' # Choose between 'log_reg', 'nb', 'svm', 'cnn'
    
    print('Grid search:\t\t\t', grid_search)
    print('Chosen model:\t\t\t', chosen_model, '\n')

    # Load datasets
    X_train_og = np.load(path + 'Xtrain_Classification1.npy')
    Y_train_og = np.load(path + 'ytrain_Classification1.npy')
    X_test_og = np.load(path + 'Xtest_Classification1.npy')

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
        X_train_og, Y_train_og, test_size = 0.2, random_state = 21)
    
    print('X_train shape:\t\t\t', X_train.shape)
    print('Y_train shape:\t\t\t', Y_train.shape)
    print('X_val shape:\t\t\t', X_val.shape)
    print('Y_val shape:\t\t\t', Y_val.shape)

    if chosen_model == 'log_reg':
        print('#################################################################')
        print('Logistic Regression:\n')
        
        X_train_bal, Y_train_bal, bal_mode = balance_data(X_train, Y_train, bal_mode)
        print('Balancing mode:\t\t\t', bal_mode)
        print('X_train_bal shape:\t\t', X_train_bal.shape)
        print('Y_train_bal shape:\t\t', Y_train_bal.shape, '\n')
        log_reg_model = logistic_regression(X_train_bal, Y_train_bal, grid_search, bal_mode)

        if grid_search:
            print('Best parameters:', log_reg_model.best_params_)
        
        y_pred = log_reg_model.predict(X_val)
        print('Balanced accuracy score:', balanced_accuracy_score(Y_val, y_pred))

        # y_test = log_reg_model.predict(X_test)
        # np.save('ytest_Classification1.npy', y_test)
    
    elif chosen_model == 'nb':
        print('#################################################################')
        print('Naive Bayes:\n')

        X_train_bal, Y_train_bal, bal_mode = balance_data(X_train, Y_train, bal_mode)
        print('Balancing mode:\t\t\t', bal_mode)
        print('X_train_bal shape:\t\t', X_train_bal.shape)
        print('Y_train_bal shape:\t\t', Y_train_bal.shape, '\n')
        # NB Modes: 'gaussian', 'multinomial', 'complement', 'bernoulli'
        nb_model, nb_mode = naive_bayes(X_train_bal, Y_train_bal, nb_mode = 'gaussian', bal_mode = bal_mode)
        
        print('NB Mode:\t\t\t', nb_mode)
        y_pred = nb_model.predict(X_val)
        print('Balanced accuracy score:\t', balanced_accuracy_score(Y_val, y_pred))

        # y_test = nb_model.predict(X_test)
        # np.save('ytest_Classification1.npy', y_test)
    
    elif chosen_model == 'svm':
        print('#################################################################')
        print('SVM:\n')
        
        X_train_bal, Y_train_bal, bal_mode = balance_data(X_train, Y_train, bal_mode)
        print('Balancing mode:\t\t\t', bal_mode)
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
        X_train_bal, Y_train_bal, bal_mode = balance_data(X_train, Y_train, bal_mode)
        print('Balancing mode:\t\t\t', bal_mode)
        print('X_train_bal shape:\t\t', X_train_bal.shape)
        print('Y_train_bal shape:\t\t', Y_train_bal.shape, '\n')

        X_train_reshaped, X_val_reshaped, X_test_reshaped = reshape_data(X_train_bal, X_val, X_test)
        print('X_train_reshaped shape:\t\t', X_train_reshaped.shape)
        print('X_val_reshaped shape:\t\t', X_val_reshaped.shape)
        print('X_test_reshaped shape:\t\t', X_test_reshaped.shape, '\n')
        cnn_model, history = cnn_grid(X_train_reshaped, Y_train_bal, 0.3, grid_search)
        #cnn_model, history = cnn(X_train_reshaped, Y_train_bal, dropout = 0.3)
        plot_loss(history)

        y_pred = np.argmax(cnn_model.predict(X_val_reshaped), axis = -1)
        print('Balanced accuracy score:', balanced_accuracy_score(Y_val, y_pred))
        
        # y_test = cnn_model.predict(X_test_reshaped)
        # np.save('ytest_Classification1.npy', y_test)
        plt.show()
    else:
        print('#################################################################')
        print('Chose a valid model!')
    
    print('#################################################################')

    

if __name__ == "__main__":
    main()