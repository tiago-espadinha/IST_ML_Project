import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import cross_validate

path = 'lab1/data/'

def calc_SSE(y_real, y_pred):
    if y_real.shape != y_pred.shape:
        print(f'y_real.shape = {y_real.shape}, y_pred.shape = {y_pred.shape}')
        print('Confirm that both of your inputs have the same shape!')
    else:
        SSE = np.sum((y_real - y_pred) ** 2)
        return SSE
    
def cv_metric(model, x, y, k, N):
    cv_results = cross_validate(model, x, y, cv = k, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = True)
    cv_SSE = abs(cv_results['test_neg_mean_squared_error'].mean()) * N
    cv_r2 = cv_results['train_r2'].mean()
    return cv_SSE, cv_r2

def main():
    
    x_train = np.load(path + 'X_train_regression1.npy')
    y_train = np.load(path + 'y_train_regression1.npy')
    x_test = np.load(path + 'X_test_regression1.npy')

    alpha = 0.1
    k = 3 # k folds in cv
    N = x_train.shape[0]

    # print('x_train shape\t= ', x_train.shape)
    # print('y_train shape\t= ', y_train.shape)
    # print('x_test shape\t= ', x_test.shape)

    # Simple Linear Regression --------------------------------------------------
    regr = LinearRegression().fit(x_train, y_train)

    # # print('intercept\t=\n', regr.intercept_)
    # # print('coef\t=\n', regr.coef_)

    # # betas = np.append(regr.intercept_, regr.coef_).reshape((x_train.shape[1] + 1, 1))
    # # print('betas\t=\n', betas)
    # # X0 = np.ones((x_train.shape[0], 1))
    # # print('X0\t=\n', X0)
    # # X = np.hstack((X0, x_train))
    # # print('X\t=\n', X)
    # # y_pred = X @ betas
    # # print('y_pred\t=\n', y_pred)

    y_pred = regr.predict(x_train)
    # print('SLR:\ty_pred\t=\n', y_pred)

    SSE = calc_SSE(y_train, y_pred)
    print('SLR:\tSSE\t= ', SSE)

    cv_SSE, cv_r2 = cv_metric(regr, x_train, y_train, k, N)
    print('SLR:\tcv_SSE\t= ', cv_SSE)
    print('SLR:\tcv_r2\t= ', cv_r2)

    # Ridge ---------------------------------------------------------------------
    ridge = Ridge(alpha = 0.49901).fit(x_train, y_train)

    y_pred = ridge.predict(x_train).reshape(N, 1)
    # print('R:\ty_pred\t=\n', y_pred)
    SSE = calc_SSE(y_train, y_pred)
    print('R:\tSSE\t= ', SSE)

    cv_SSE, cv_r2 = cv_metric(ridge, x_train, y_train, k, N)
    print('R:\tcv_SSE\t= ', cv_SSE)
    print('R:\tcv_r2\t= ', cv_r2)

    # RidgeCV -------------------------------------------------------------------
    alphas_RCV = np.arange(0.00001, 0.5, 0.001)
    ridgeCV = RidgeCV(alphas = alphas_RCV, cv = k).fit(x_train, y_train)

    y_pred = ridgeCV.predict(x_train).reshape(N, 1)
    # print('RCV:\ty_pred\t=\n', y_pred)
    print('Best alpha RCV\t= ', ridgeCV.alpha_)
    SSE = calc_SSE(y_train, y_pred)
    print('RCV:\tSSE\t= ', SSE)

    cv_SSE, cv_r2 = cv_metric(ridgeCV, x_train, y_train, k, N)
    print('RCV:\tcv_SSE\t= ', cv_SSE)
    print('RCV:\tcv_r2\t= ', cv_r2)

    # Lasso ---------------------------------------------------------------------
    lasso = Lasso(alpha = 0.02801).fit(x_train, y_train)

    y_pred = lasso.predict(x_train).reshape(N, 1)
    # print('L:\ty_pred\t=\n', y_pred)
    SSE = calc_SSE(y_train, y_pred)
    print('L:\tSSE\t= ', SSE)

    cv_SSE, cv_r2 = cv_metric(lasso, x_train, y_train, k, N)
    print('L:\tcv_SSE\t= ', cv_SSE)
    print('L:\tcv_r2\t= ', cv_r2)

    # LassoCV -------------------------------------------------------------------
    alphas_LCV = np.arange(0.00001, 0.5, 0.001)
    lassoCV = LassoCV(alphas = alphas_LCV, cv = k).fit(x_train, y_train.ravel())

    y_pred = lassoCV.predict(x_train).reshape(N, 1)
    # print('LCV:\ty_pred\t=\n', y_pred)
    print('Best alpha LCV\t= ', lassoCV.alpha_)
    SSE = calc_SSE(y_train, y_pred)
    print('LCV:\tSSE\t= ', SSE)

    cv_SSE, cv_r2 = cv_metric(lassoCV, x_train, y_train.ravel(), k, N)
    print('LCV:\tcv_SSE\t= ', cv_SSE)
    print('LCV:\tcv_r2\t= ', cv_r2)


if __name__ == "__main__":
    main()