import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate

path = 'Reg1/'

def metrics(y_real, y_pred):
    if y_real.shape != y_pred.shape:
        print(f'y_real.shape = {y_real.shape}, y_pred.shape = {y_pred.shape}')
        print('Confirm that both of your inputs have the same shape!')
    else:
        SSE = np.sum((y_real - y_pred) ** 2)
        r2 = r2_score(y_real, y_pred)
        return SSE, r2

def main():
    
    X_train = np.load(path + 'X_train_regression1.npy')
    y_train = np.load(path + 'y_train_regression1.npy')
    X_test = np.load(path + 'X_test_regression1.npy')
    y_test = np.load(path + 'y_test_regression1.npy')
    
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit(X_train).transform(X_train)
    y_train_scaled = y_scaler.fit(y_train).transform(y_train)
    X_test_scaled = X_scaler.fit(X_test).transform(X_test)
    
    N_test = X_test.shape[0]

    # Ridge ----------------------------------------------------------------------------------
    # alphas_R = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    # alphas_R = np.arange(1.1, 2.1, 0.001) # optimized interval

    ridge = Ridge(alpha = 1.462).fit(X_train_scaled, y_train_scaled)

    y_test_pred_scaled_R = ridge.predict(X_test_scaled).reshape(N_test, 1)
    y_test_pred_R = y_scaler.inverse_transform(y_test_pred_scaled_R)
    #np.save(path + 'y_test_regression1_.npy', y_test_pred_R)
    # print('Ridge:\ty_test_pred\t=\n', y_test_pred_R)

    test_pred = y_test_pred_R - y_test

    # print('test_pred:\t= ', test_pred)

    SSE, _ = metrics(y_test, y_test_pred_R)
    print('SSE:\t= ', SSE)

if __name__ == "__main__":
    main()