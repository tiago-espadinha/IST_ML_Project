import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import cross_validate

path = 'lab1/data/'

# np.save(path + 'y_test_regression1.npy', y_pred)

def metrics(y_real, y_pred):
    if y_real.shape != y_pred.shape:
        print(f'y_real.shape = {y_real.shape}, y_pred.shape = {y_pred.shape}')
        print('Confirm that both of your inputs have the same shape!')
    else:
        SSE = np.sum((y_real - y_pred) ** 2)
        r2 = r2_score(y_real, y_pred)
        return SSE, r2
    
def cv_metrics(model, x, y, k, N):
    scoring = {'mse': make_scorer(mean_squared_error)}
    cv_results = cross_validate(model, x, y, cv = k, scoring = scoring)
    
    # print('cv_results\t= ', cv_results)
    
    avg_mse = np.mean(cv_results['test_mse'])
    cv_SSE = avg_mse * N
    # cv_r2 = cv_results['train_r2'].mean()
    return cv_SSE

def main():
    
    x_train = np.load(path + 'X_train_regression1.npy')
    y_train = np.load(path + 'y_train_regression1.npy')
    x_test = np.load(path + 'X_test_regression1.npy')

    step = 0.01
    k = 3 # k folds in cv
    N = x_train.shape[0]

    # Simple Linear Regression --------------------------------------------------
    regr = LinearRegression().fit(x_train, y_train)

    y_pred = regr.predict(x_train)

    cv_SSE = cv_metrics(regr, x_train, y_train, k, N)
    print('SLR:\tcv_SSE\t= ', cv_SSE)
    # print('SLR:\tcv_r2\t= ', cv_r2)
    
    print('----------')

    # RidgeCV -------------------------------------------------------------------
    alphas_RCV = np.arange(0.00001, 0.8, step)

    ridgeCV = RidgeCV(alphas = alphas_RCV, cv = k).fit(x_train, y_train)

    y_pred = ridgeCV.predict(x_train).reshape(N, 1)

    # print('RCV:\ty_pred\t=\n', y_pred)
    # print('Best alpha RCV\t= ', ridgeCV.alpha_)

    cv_SSE = cv_metrics(ridgeCV, x_train, y_train, k, N)
    print('RCV:\tcv_SSE\t= ', cv_SSE)
    # print('RCV:\tcv_r2\t= ', cv_r2)

    print('----------')

    # LassoCV -------------------------------------------------------------------
    alphas_LCV = np.arange(0.00001, 0.8, step)

    lassoCV = LassoCV(alphas = alphas_LCV, cv = k).fit(x_train, y_train.ravel())

    y_pred = lassoCV.predict(x_train).reshape(N, 1)

    # print('LCV:\ty_pred\t=\n', y_pred)
    # print('Best alpha LCV\t= ', lassoCV.alpha_)

    cv_SSE = cv_metrics(lassoCV, x_train, y_train.ravel(), k, N)
    print('LCV:\tcv_SSE\t= ', cv_SSE)
    # print('LCV:\tcv_r2\t= ', cv_r2)

    print('----------')

    # ElasticNetCV --------------------------------------------------------------
    alphas_ENCV = np.arange(0.00001, 0.8, step)
    l1_ratios_ENCV = np.arange(0.1, 1, 0.1)
    cv_SSE = 1e100

    for l1_ratio in l1_ratios_ENCV:
        elasticNetCV = ElasticNetCV(alphas = alphas_ENCV, cv = k, l1_ratio = l1_ratio).fit(x_train, y_train.ravel())
    
        y_pred = elasticNetCV.predict(x_train).reshape(N, 1)

        # print('ENCV:\ty_pred\t=\n', y_pred)
        # print('Best alpha ENCV\t= ', elasticNetCV.alpha_)
        
        cv_SSE_aux = cv_metrics(elasticNetCV, x_train, y_train.ravel(), k, N)
        
        if cv_SSE_aux < cv_SSE:
            cv_SSE = cv_SSE_aux
            best_l1_ratio = l1_ratio

    print('ENCV:\tbest_l1_ratio\t= ', best_l1_ratio)
    elasticNetCV = ElasticNetCV(alphas = alphas_ENCV, cv = k, l1_ratio = best_l1_ratio).fit(x_train, y_train.ravel())
    
    y_pred = elasticNetCV.predict(x_train).reshape(N, 1)

    # print('ENCV:\ty_pred\t=\n', y_pred)
    # print('Best alpha ENCV\t= ', elasticNetCV.alpha_)
    
    cv_SSE = cv_metrics(elasticNetCV, x_train, y_train.ravel(), k, N)
    print('ENCV:\tcv_SSE\t= ', cv_SSE)
    # print('ENCV:\tcv_r2\t= ', cv_r2)

    print('----------')

if __name__ == "__main__":
    main()