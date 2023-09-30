import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate

# Change path to the folder where the data is located
path = ''
k_save = 5 # k folds to save
model_save = 'R' # model to save (R - Ridge, L - Lasso, EN - ElasticNet)
single_model = False # True to run only the model specified in model_save

def metrics(y_real, y_pred):
    if y_real.shape != y_pred.shape:
        print(f'y_real.shape = {y_real.shape}, y_pred.shape = {y_pred.shape}')
        print('Confirm that both of your inputs have the same shape!')
    else:
        SSE = mean_squared_error(y_real, y_pred) * y_real.shape[0]
        r2 = r2_score(y_real, y_pred)
        return SSE, r2

def cv_metrics(model, x, y, k, N):
    scoring = ('r2', 'neg_mean_squared_error')
    cv_results = cross_validate(model, x, y, cv = k, scoring = scoring, return_train_score = True)
    
    # print('cv_results\t= ', cv_results)
    
    avg_mse = np.mean(abs(cv_results['test_neg_mean_squared_error']))
    cv_SSE = avg_mse * (N / k)
    folds_avg_mse = abs(cv_results['test_neg_mean_squared_error'])
    folds_cv_SSE = folds_avg_mse * (N / k)
    return cv_SSE, folds_cv_SSE

def plot_features(x_train, y_train):
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

    plt.figure(figsize=(15, 6))
    for i in range(x_train.shape[1]):
        plt.subplot(2, 6, i + 1)
        for j in range(len(y_train)):
            plt.scatter(j, x_train[j, i], color=colors[j % len(colors)])
        plt.xlabel(f'Feature {i + 1}')
        plt.ylabel('Value')

    plt.subplot(2, 6, 11)
    for j in range(len(y_train)):
        plt.scatter(j, y_train[j], color=colors[j % len(colors)])
    plt.xlabel('Sample')
    plt.ylabel('Output')

    plt.tight_layout()
    plt.show()

def plot_SSE(x_arr, y_arr, best_x, best_y, title, k):

    if title == 'Ridge':
        fig = plt.figure(1)
        plt.xlim(0, 3)
    elif title == 'Lasso':
        fig = plt.figure(2)
        plt.xlim(0, 1)
    else:
        fig = plt.figure()
    
    plt.plot(x_arr, y_arr, label = 'k = ' + str(k))
    plt.legend()
    plt.scatter(best_x, best_y, c='r', marker='x')
    plt.xlabel('alpha')
    plt.ylabel('CV SSE')
    plt.title(title)

def plot_SSE_3D(data_arr, best_x, best_y, best_z, title, k):
    
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(data_arr[:,0], data_arr[:,1], data_arr[:,2], cmap='viridis', label = 'k = ' + str(k))
        ax.scatter(best_x, best_y, best_z, c='r', marker='x', s=50)
        ax.set_xlabel('l1_ratio')
        ax.set_ylabel('alpha')
        ax.set_zlabel('CV SSE')
        ax.set_title(title + ', k=' + str(k))

def main():
    
    X_train = np.load(path + 'X_train_regression1.npy')
    y_train = np.load(path + 'y_train_regression1.npy')
    X_test = np.load(path + 'X_test_regression1.npy')

    #plot_features(X_train, y_train) # one possible outilier
    
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit(X_train).transform(X_train)
    y_train_scaled = y_scaler.fit(y_train).transform(y_train)
    X_test_scaled = X_scaler.fit(X_test).transform(X_test)

    #plot_features(X_train_scaled, y_train_scaled)

    k_tests = [3, 5] # k folds to test
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    for k in k_tests:
        print('-------------------------------------------------------')
        print('k = ', k)
        print()

        # # Linear Regression ----------------------------------------------------------------------
        regr = LinearRegression().fit(X_train_scaled, y_train_scaled)
        
        cv_SSE_LR, folds_cv_SSE_LR = cv_metrics(regr, X_train_scaled, y_train_scaled, k, N_train)
        print('LR:\t\tcv_SSE\t\t= ', cv_SSE_LR)
        print('LR:\t\tfolds_cv_SSE\t= ', folds_cv_SSE_LR)

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
        if (single_model and model_save == 'R') or not single_model:
            # alphas_R = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
            # alphas_R = np.arange(1.1, 2.1, 0.001) # optimized interval
            alphas_R = np.arange(0.001, 3, 0.001) # for the plot
            best_cv_SSE_R = 1e100
            cv_SSE_R_array = np.array([])

            for alpha_R in alphas_R:
                ridge = Ridge(alpha = alpha_R).fit(X_train_scaled, y_train_scaled)
                cv_SSE_R, folds_cv_SSE_R = cv_metrics(ridge, X_train_scaled, y_train_scaled, k, N_train)
                cv_SSE_R_array = np.append(cv_SSE_R_array, cv_SSE_R)

                if cv_SSE_R < best_cv_SSE_R:
                    best_cv_SSE_R = cv_SSE_R
                    best_folds_cv_SSE_R = folds_cv_SSE_R
                    best_alpha_R = alpha_R
                    best_ridge = ridge
            
            if k==k_tests[0]:
                cv_SSE_R_array_3 = cv_SSE_R_array
            elif k==k_tests[1]:
                cv_SSE_R_array_5 = cv_SSE_R_array
                cv_SSE_R_array_M = (cv_SSE_R_array_3 + cv_SSE_R_array_5)/2
                argmin = np.argmin(cv_SSE_R_array_M)
                best_alpha_RM = alphas_R[argmin]



            plot_SSE(alphas_R, cv_SSE_R_array, best_alpha_R, best_cv_SSE_R, 'Ridge', k)
            
            print('Ridge:\t\tBest alpha\t= ', best_alpha_R)
            print('Ridge:\t\tcv_SSE\t\t= ', best_cv_SSE_R)
            print('Ridge:\t\tfolds_cv_SSE\t= ', best_folds_cv_SSE_R)

            y_train_pred_scaled_R = best_ridge.predict(X_train_scaled).reshape(N_train, 1)
            y_train_pred_R = y_scaler.inverse_transform(y_train_pred_scaled_R)

            SSE_R, r2_R = metrics(y_train, y_train_pred_R)

            print('Ridge:\t\tSSE\t\t= ', SSE_R)
            print('Ridge:\t\tr2\t\t= ', r2_R)

            if k==k_tests[1]:
                ridge_M = Ridge(alpha = best_alpha_RM).fit(X_train_scaled, y_train_scaled)
                cv_SSE_R1, _ = cv_metrics(ridge, X_train_scaled, y_train_scaled, k_tests[0], N_train)
                cv_SSE_R2, _ = cv_metrics(ridge, X_train_scaled, y_train_scaled, k_tests[1], N_train)
                print('Ridge_M:\tBest alpha RM\t= ', best_alpha_RM)
                print('Ridge_M:\tcv_SSE k=' + str(k_tests[0]) + '\t= ', cv_SSE_R1)
                print('Ridge_M:\tcv_SSE k=' + str(k_tests[1]) + '\t= ', cv_SSE_R2)

            print()

            if model_save == 'R' and k==k_tests[1]:
                y_test_pred_scaled_R = ridge_M.predict(X_test_scaled).reshape(N_test, 1)
                y_test_pred_R = y_scaler.inverse_transform(y_test_pred_scaled_R)
                np.save(path + 'y_test_regression1.npy', y_test_pred_R)
                # print('Ridge:\ty_test_pred\t=\n', y_test_pred_R)

        # Lasso ----------------------------------------------------------------------------------
        if (single_model and model_save == 'L') or not single_model:
            # alphas_L = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
            # alphas_L = np.arange(0.01, 0.06, 0.001) # optimized interval
            alphas_L = np.arange(0.001, 1, 0.001) # for the plot
            best_cv_SSE_L = 1e100
            cv_SSE_L_array = np.array([])

            for alpha_L in alphas_L:
                lasso = Lasso(alpha = alpha_L, max_iter = 10000).fit(X_train_scaled, y_train_scaled)
                cv_SSE_L, folds_cv_SSE_L = cv_metrics(lasso, X_train_scaled, y_train_scaled, k, N_train)
                cv_SSE_L_array = np.append(cv_SSE_L_array, cv_SSE_L)

                if cv_SSE_L < best_cv_SSE_L:
                    best_cv_SSE_L = cv_SSE_L
                    best_folds_cv_SSE_L = folds_cv_SSE_L
                    best_alpha_L = alpha_L
                    best_lasso = lasso

            plot_SSE(alphas_L, cv_SSE_L_array, best_alpha_L, best_cv_SSE_L, 'Lasso', k)
            
            print('Lasso:\t\tBest alpha\t= ', best_alpha_L)
            print('Lasso:\t\tcv_SSE\t\t= ', best_cv_SSE_L)
            print('Lasso:\t\tfolds_cv_SSE\t= ', best_folds_cv_SSE_L)

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
        if (single_model and model_save == 'EN') or not single_model:
            # alphas_EN = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
            alphas_EN = np.arange(0.08, 0.18, 0.001)
            l1_ratios_EN = np.arange(0.00001, 0.04, 0.001)
            best_cv_SSE_EN = 1e100
            cv_SSE_EN_array = np.array([])

            for l1_ratio_EN in l1_ratios_EN:
                for alpha_EN in alphas_EN: 
                    elasticNet = ElasticNet(alpha = alpha_EN, l1_ratio = l1_ratio_EN, max_iter = 100000).fit(X_train_scaled, y_train_scaled.ravel())
                    cv_SSE_EN, folds_cv_SSE_EN = cv_metrics(elasticNet, X_train_scaled, y_train_scaled.ravel(), k, N_train)
                    cv_SSE_EN_array = np.append(cv_SSE_EN_array, [[l1_ratio_EN, alpha_EN, cv_SSE_EN]])
                    
                    if cv_SSE_EN < best_cv_SSE_EN:
                        best_cv_SSE_EN = cv_SSE_EN
                        best_folds_cv_SSE_EN = folds_cv_SSE_EN
                        best_alpha_EN = alpha_EN
                        best_l1_ratio_EN = l1_ratio_EN
                        best_elasticNet = elasticNet

            cv_SSE_EN_array = np.reshape(cv_SSE_EN_array, (4000, 3))
            plot_SSE_3D(cv_SSE_EN_array, best_l1_ratio_EN, best_alpha_EN, best_cv_SSE_EN, 'ElasticNet', k)

            print('ElasticNet:\tBest alpha\t= ', best_alpha_EN)
            print('ElasticNet:\tBest l1_ratio\t= ', best_l1_ratio_EN)
            print('ElasticNet:\tcv_SSE\t\t= ', best_cv_SSE_EN)
            print('ElasticNet:\tfolds_cv_SSE\t= ', best_folds_cv_SSE_EN)

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

    plt.show()

if __name__ == "__main__":
    main()