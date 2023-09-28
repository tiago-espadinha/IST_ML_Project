import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns

path = 'lab1/data/'
k_save = 3 # k folds to save
model_save = 'R' # model to save

def metrics(y_real, y_pred):
    if y_real.shape != y_pred.shape:
        print(f'y_real.shape = {y_real.shape}, y_pred.shape = {y_pred.shape}')
        print('Confirm that both of your inputs have the same shape!')
    else:
        SSE = np.sum((y_real - y_pred) ** 2)
        r2 = r2_score(y_real, y_pred)
        return SSE, r2

def cv_metrics(model, x, y, k, N):
    scoring = ('r2', 'neg_mean_squared_error')
    cv_results = cross_validate(model, x, y, cv = k, scoring = scoring, return_train_score = True)
    
    # print('cv_results\t= ', cv_results)
    
    avg_mse = np.mean(abs(cv_results['test_neg_mean_squared_error']))
    cv_SSE = avg_mse * N
    cv_r2 = cv_results['train_r2'].mean()
    return cv_SSE, cv_r2

def plot_features(x):
    # Create a grid of subplots for scatter plots
    num_features = x.shape[1]
    fig, axes = plt.subplots(nrows=2, ncols=num_features // 2, figsize=(15, 6))

    # Create scatter plots for each feature
    for feature_index, ax in enumerate(axes.flat):
        feature_values = x[:, feature_index]
        ax.scatter(range(len(feature_values)), feature_values, alpha=0.5)
        ax.set_title(f"Feature {feature_index + 1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel(f"Feature {feature_index + 1}")

    plt.tight_layout()
    plt.show()

def main():
    
    X_train = np.load(path + 'X_train_regression1.npy')
    y_train = np.load(path + 'y_train_regression1.npy')
    X_test = np.load(path + 'X_test_regression1.npy')

    plot_features(X_train) # NO OUTLIERS!
    
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit(X_train).transform(X_train)
    y_train_scaled = y_scaler.fit(y_train).transform(y_train)
    X_test_scaled = X_scaler.fit(X_test).transform(X_test)

    k_tests = [3, 5] # k folds to test
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    for k in k_tests:
        print('-------------------------------------------------------')
        print('k = ', k)
        print()

        # Linear Regression ----------------------------------------------------------------------
        regr = LinearRegression().fit(X_train_scaled, y_train_scaled)
        
        cv_SSE_LR, cv_r2_LR = cv_metrics(regr, X_train_scaled, y_train_scaled, k, N_train)
        print('LR:\t\tcv_SSE\t\t= ', cv_SSE_LR)
        # print('LR:\t\tcv_r2\t\t= ', cv_r2_LR)

        y_train_pred_scaled_LR = regr.predict(X_train_scaled).reshape(N_train, 1)
        y_train_pred_LR = y_scaler.inverse_transform(y_train_pred_scaled_LR)

        SSE_LR, r2_LR = metrics(y_train, y_train_pred_LR)

        print('LR:\t\tSSE\t\t= ', SSE_LR)
        print('LR:\t\tr2\t\t= ', r2_LR)
        print()

        if k == k_save and model_save == 'LR':
            y_test_pred_scaled_LR = regr.predict(X_test_scaled).reshape(N_test, 1)
            y_test_pred_LR = y_scaler.inverse_transform(y_test_pred_scaled_LR)
            np.save(path + 'y_test_regression1.npy', y_test_pred_LR)
            # print('SLR:\ty_test_pred\t=\n', y_test_pred_LR)
        
        # Ridge ----------------------------------------------------------------------------------
        # alphas_R = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        alphas_R = np.arange(1.1, 2.1, 0.001)
        best_cv_SSE_R = 1e100

        for alpha_R in alphas_R:
            ridge = Ridge(alpha = alpha_R).fit(X_train_scaled, y_train_scaled)
            cv_SSE_R, cv_r2_R = cv_metrics(ridge, X_train_scaled, y_train_scaled, k, N_train)
            
            if cv_SSE_R < best_cv_SSE_R:
                best_cv_SSE_R = cv_SSE_R
                best_cv_r2_R = cv_r2_R
                best_alpha_R = alpha_R
                best_ridge = ridge
        
        print('Ridge:\t\tBest alpha\t= ', best_alpha_R)
        print('Ridge:\t\tcv_SSE\t\t= ', best_cv_SSE_R)
        # print('Ridge:\t\tcv_r2\t\t= ', best_cv_r2_R)

        y_train_pred_scaled_R = best_ridge.predict(X_train_scaled).reshape(N_train, 1)
        y_train_pred_R = y_scaler.inverse_transform(y_train_pred_scaled_R)

        SSE_R, r2_R = metrics(y_train, y_train_pred_R)

        print('Ridge:\t\tSSE\t\t= ', SSE_R)
        print('Ridge:\t\tr2\t\t= ', r2_R)

        print()

        if k == k_save and model_save == 'R':
            y_test_pred_scaled_R = best_ridge.predict(X_test_scaled).reshape(N_test, 1)
            y_test_pred_R = y_scaler.inverse_transform(y_test_pred_scaled_R)
            np.save(path + 'y_test_regression1.npy', y_test_pred_R)
            # print('Ridge:\ty_test_pred\t=\n', y_test_pred_R)

        # Lasso ----------------------------------------------------------------------------------
        # alphas_L = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        alphas_L = np.arange(0.01, 0.06, 0.001)
        best_cv_SSE_L = 1e100

        for alpha_L in alphas_L:
            lasso = Lasso(alpha = alpha_L, max_iter = 10000).fit(X_train_scaled, y_train_scaled)
            cv_SSE_L, cv_r2_L = cv_metrics(lasso, X_train_scaled, y_train_scaled, k, N_train)
            
            if cv_SSE_L < best_cv_SSE_L:
                best_cv_SSE_L = cv_SSE_L
                best_cv_r2_L = cv_r2_L
                best_alpha_L = alpha_L
                best_lasso = lasso
        
        print('Lasso:\t\tBest alpha\t= ', best_alpha_L)
        print('Lasso:\t\tcv_SSE\t\t= ', best_cv_SSE_L)
        # print('Lasso:\t\tcv_r2\t\t= ', best_cv_r2_L)

        y_train_pred_scaled_L = best_lasso.predict(X_train_scaled).reshape(N_train, 1)
        y_train_pred_L = y_scaler.inverse_transform(y_train_pred_scaled_L)

        SSE_L, r2_L = metrics(y_train, y_train_pred_L)

        print('Lasso:\t\tSSE\t\t= ', SSE_L)
        print('Lasso:\t\tr2\t\t= ', r2_L)

        print()

        if k == k_save and model_save == 'L':
            y_test_pred_scaled_L = best_lasso.predict(X_test_scaled).reshape(N_test, 1)
            y_test_pred_L = y_scaler.inverse_transform(y_test_pred_scaled_L)
            np.save(path + 'y_test_regression1.npy', y_test_pred_L)
            # print('RCV:\ty_pred\t=\n', y_test_pred_L)

        # ElasticNet -----------------------------------------------------------------------------
        # alphas_EN = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        alphas_EN = np.arange(0.08, 0.18, 0.001)
        l1_ratios_EN = np.arange(0.00001, 0.04, 0.001)
        best_cv_SSE_EN = 1e100
        
        for l1_ratio_EN in l1_ratios_EN:
            for alpha_EN in alphas_EN: 
                elasticNet = ElasticNet(alpha = alpha_EN, l1_ratio = l1_ratio_EN, max_iter = 100000).fit(X_train_scaled, y_train_scaled.ravel())
                cv_SSE_EN, cv_r2_EN = cv_metrics(elasticNet, X_train_scaled, y_train_scaled.ravel(), k, N_train)
                
                if cv_SSE_EN < best_cv_SSE_EN:
                    best_cv_SSE_EN = cv_SSE_EN
                    best_cv_r2_EN = cv_r2_EN
                    best_alpha_EN = alpha_EN
                    best_l1_ratio_EN = l1_ratio_EN
                    best_elasticNet = elasticNet

        print('ElasticNet:\tBest alpha\t= ', best_alpha_EN)
        print('ElasticNet:\tBest l1_ratio\t= ', best_l1_ratio_EN)
        print('ElasticNet:\tcv_SSE\t\t= ', best_cv_SSE_EN)
        # print('ElasticNet:\tcv_r2\t\t= ', best_cv_r2_EN)

        y_train_pred_scaled_EN = best_elasticNet.predict(X_train_scaled).reshape(N_train, 1)
        y_train_pred_EN = y_scaler.inverse_transform(y_train_pred_scaled_EN)

        SSE_EN, r2_EN = metrics(y_train, y_train_pred_EN)

        print('ElasticNet:\tSSE\t\t= ', SSE_EN)
        print('ElasticNet:\tr2\t\t= ', r2_EN)

        if k == k_save and model_save == 'EN':
            y_test_pred_scaled_EN = best_elasticNet.predict(X_test_scaled).reshape(N_test, 1)
            y_test_pred_EN = y_scaler.inverse_transform(y_test_pred_scaled_EN)
            np.save(path + 'y_test_regression1.npy', y_test_pred_EN)
            # print('ElasticNet:\ty_test_pred\t=\n', y_test_pred_EN)

    print('-------------------------------------------------------')

if __name__ == "__main__":
    main()