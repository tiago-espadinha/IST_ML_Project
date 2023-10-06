'''
Machine Learning - Linear Regression Problem 2
Authors:
Tiago Simões, 96329
Tomás Fonseca, 66325
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import cross_val_score

# Change path to the folder where the data is located
path = '' 

# Use StandardScaler to scale the data
scale = True 

# Test different thresholds for RANSAC, 
# if False alpha_c1 = 0.6 and alpha_c2 = 0.7 are used
test_alpha = False 


# Calculates the MSE for each cluster and the whole dataset
def error_metrics(y_real, y_pred_c1, y_pred_c2, y_pred, inlier):
    
    MSE_c1 = mean_squared_error(y_real[inlier == True], y_pred_c1[inlier == True])
    MSE_c2 = mean_squared_error(y_real[inlier == False], y_pred_c2[inlier == False])
    MSE_all = mean_squared_error(y_real, y_pred)

    print('MSE\t| ', MSE_c1, '\t|', MSE_c2, '\t|', MSE_all)

# Uses cross validation to calculate the MSE    
def cv_metrics(model, x, y, n_folds):
    
    cv_results = cross_val_score(model, x, y, cv = n_folds, scoring = 'neg_mean_squared_error')
    
    avg_mse = np.mean(abs(cv_results))
    folds_avg_mse = abs(cv_results)
    return avg_mse, folds_avg_mse

# Performs RANSAC clustering
def ransac_classifier(x, y, threshold):
    
    ransac = RANSACRegressor(random_state = 1, residual_threshold = threshold).fit(x, y)
    inlier_mask = ransac.inlier_mask_

    x_cluster1 = x[inlier_mask == True]
    x_cluster2 = x[inlier_mask == False]
    y_cluster1 = y[inlier_mask == True]
    y_cluster2 = y[inlier_mask == False]

    return ransac, x_cluster1, x_cluster2, y_cluster1, y_cluster2, inlier_mask

def main():
    
    # Load datasets
    X_train = np.load(path + 'X_train_regression2.npy')
    Y_train = np.load(path + 'y_train_regression2.npy')
    X_test = np.load(path + 'X_test_regression2.npy')

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler() 

    if scale:
        X_train = X_scaler.fit(X_train).transform(X_train)
        Y_train = Y_scaler.fit(Y_train).transform(Y_train)
        X_test = X_scaler.transform(X_test)

    print('-------------------------------------------------------')
    print('RANSAC Regression\n')

    best_alpha_c1 = 0.6
    best_alpha_c2 = 0.7
    if test_alpha:
        alphas = np.arange(0.4, 1.2, 0.1)
        best_MSE = 1e100

        for alpha_c1 in alphas:
            
            # identify clusters and fit regressor to first cluster
            ransac1, x_cluster1, x_cluster2, y_cluster1, y_cluster2, inlier_mask = ransac_classifier(X_train, Y_train, alpha_c1)
            regr_c1 = ransac1.fit(x_cluster1, y_cluster1)
            CV_MSE_c1, fold_MSE_c1 = cv_metrics(regr_c1, x_cluster1, y_cluster1, x_cluster1.shape[0])
            
            for alpha_c2 in alphas:

                # fit regressor to second cluster
                ransac2, x_cluster2_in, _, y_cluster2_in, _, _ = ransac_classifier(x_cluster2, y_cluster2, alpha_c2)
                regr_c2 = ransac2.fit(x_cluster2_in, y_cluster2_in)
                CV_MSE_c2, fold_MSE_c2 = cv_metrics(regr_c2, x_cluster2, y_cluster2, x_cluster2.shape[0])

                # find best threshold combination
                if CV_MSE_c1 + CV_MSE_c2 < best_MSE:
                    best_MSE = CV_MSE_c1 + CV_MSE_c2
                    best_alpha_c1 = alpha_c1
                    best_alpha_c2 = alpha_c2

    # fit regressor to first cluster with best thresholds
    ransac1, x_cluster1, x_cluster2, y_cluster1, y_cluster2, inlier_mask = ransac_classifier(X_train, Y_train, best_alpha_c1)
    regr_c1 = ransac1.fit(x_cluster1, y_cluster1)
    CV_MSE_c1, fold_MSE_c1 = cv_metrics(regr_c1, x_cluster1, y_cluster1, x_cluster1.shape[0])

    # fit regressor to second cluster with best thresholds
    ransac2, x_cluster2_in, _, y_cluster2_in, _, _ = ransac_classifier(x_cluster2, y_cluster2, best_alpha_c2)
    regr_c2 = ransac2.fit(x_cluster2_in, y_cluster2_in)
    CV_MSE_c2, fold_MSE_c2 = cv_metrics(regr_c2, x_cluster2, y_cluster2, x_cluster2.shape[0])
    
    # fit regressor to all data
    regr_all = LinearRegression().fit(X_train, Y_train)
    CV_MSE_all, fold_MSE_all = cv_metrics(regr_all, X_train, Y_train, 10)

    print('Cluster 1 size\t: ', y_cluster1.shape[0])
    print('Cluster 2 size\t: ', y_cluster2_in.shape[0])

    print('best_alpha_c1\t= ', best_alpha_c1)
    print('best_alpha_c2\t= ', best_alpha_c2)

    # predict training data
    y_pred_c1 = regr_c1.predict(X_train).reshape(Y_train.shape[0], 1)
    y_pred_c2 = regr_c2.predict(X_train).reshape(Y_train.shape[0], 1)
    y_pred_all = regr_all.predict(X_train).reshape(Y_train.shape[0], 1)
    
    if scale:
        y_pred_c1 = Y_scaler.inverse_transform(y_pred_c1)
        y_pred_c2 = Y_scaler.inverse_transform(y_pred_c2)
        y_pred_all = Y_scaler.inverse_transform(y_pred_all)
        
    # Analyse results
    print('\n\t|\tCluster 1\t|\tCluster 2\t|\tAll data')
    print('CV_MSE\t| ', CV_MSE_c1, '\t|', CV_MSE_c2, '\t|', CV_MSE_all)
    print('Max_MSE\t| ', np.max(fold_MSE_c1), '\t|', np.max(fold_MSE_c2), '\t|', np.max(fold_MSE_all))
    error_metrics(Y_train, y_pred_c1, y_pred_c2, y_pred_all, inlier_mask)

    # predict test data
    y_test_pred1 = regr_c1.predict(X_test).reshape(X_test.shape[0], 1)
    y_test_pred2 = regr_c2.predict(X_test).reshape(X_test.shape[0], 1)

    if scale:
        y_test_pred1 = Y_scaler.inverse_transform(y_test_pred1)
        y_test_pred2 = Y_scaler.inverse_transform(y_test_pred2)

    y_test_pred = np.hstack((y_test_pred1, y_test_pred2))
    np.save(path + 'y_test_regression2.npy', y_test_pred)

    print('-------------------------------------------------------')

if __name__ == "__main__":
    main()