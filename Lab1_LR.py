import numpy as np
#from sklearn.linear_model

path = 'C:/Users/Tiago/Desktop/Lab_ML/'

def y_calc(beta, input_set):
    n_features = input_set.shape[0]
    X0 = np.ones((n_features, 1))
    X = np.hstack((X0, input_set))
    print(beta.shape[0])
    output_aprox = X @ beta
    return output_aprox

def SSE_calc(output_set, output_aprox):
    sum = 0
    for i in range(output_set.shape[0]):
        delta=output_aprox[i][0]-output_set[i][0]
        sum += delta * delta
    return sum
        

def beta_calc(input_set):
    n_features = input_set.shape[0]
    X0 = np.ones((n_features, 1))
    X = np.hstack((X0, input_set))
    X_T = np.transpose(X)
    beta = np.linalg.inv(X_T @ X) @ X_T @ y_train

    return beta

# Load training end testing sets
x_test = np.load(path + 'X_test_regression1.npy')
x_train = np.load(path + 'X_train_regression1.npy')
y_train = np.load(path + 'y_train_regression1.npy')

# Print set shapes
print(x_test.shape)
print(x_train.shape)
print(y_train.shape)
print(f'y_train: {y_train}')

beta = beta_calc(x_train)
y_approx = y_calc(beta, x_train)
SSE = SSE_calc(y_train, y_approx)
y_test = y_calc(beta, x_train)
print(f'y_approx: {y_approx}')
print(f'SSE = {SSE}')
print(f'y_test = {y_test}')

