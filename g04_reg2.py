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
from sklearn.linear_model import LinearRegression, RANSACRegressor, LassoCV
from sklearn.model_selection import cross_val_score
import seaborn as sns

path = 'lab2/' # Change path to the folder where the data is located
scale = True

# Calculates the MSE for each cluster and the whole dataset
def error_metrics(y_real, y_pred_c1, y_pred_c2, y_pred, inlier_mask):

    err_sq_1 = mean_squared_error(y_real[inlier_mask == True], y_pred_c1[inlier_mask == True])
    err_sq_2 = mean_squared_error(y_real[inlier_mask == False], y_pred_c2[inlier_mask == False])
    err_sq_3 = mean_squared_error(y_real, y_pred)

    # err_sq_1 = np.sum(np.square(np.subtract(y_real[inlier_mask == True], y_pred_c1[inlier_mask == True])))
    # err_sq_2 = np.sum(np.square(np.subtract(y_real[inlier_mask == False], y_pred_c2[inlier_mask == False])))
    # err_sq_3 = np.sum(np.square(np.subtract(y_real, y_pred)))

    # error_SSE = np.sum(np.minimum(err_sq_1, err_sq_2))
    print('MSE\t= ', err_sq_1, '\t;\t', err_sq_2, '\t;\t', err_sq_3)

# Uses cross validation to calculate the MSE    
def cv_metrics(model, x, y, k):
    scoring = 'neg_mean_squared_error'
    cv_results = cross_val_score(model, x, y, cv = k, scoring = scoring)
    
    avg_mse = np.mean(abs(cv_results))
    folds_avg_mse = abs(cv_results)
    max_avg_mse = np.max(folds_avg_mse)
    return avg_mse, folds_avg_mse, max_avg_mse

# Performs RANSAC clustering
def ransac_classifier(x_train, y_train, res_threshold):
    ransac = RANSACRegressor(random_state = 1, residual_threshold = res_threshold).fit(x_train, y_train)
    inlier_mask = ransac.inlier_mask_
    # inliers = x_train[inlier_mask]
    # print('inliers.shape = ', inliers.shape)
    # print(inliers)
    ransac_pred = ransac.predict(x_train)

    x_cluster1 = x_train[ransac.inlier_mask_ == True]
    x_cluster2 = x_train[ransac.inlier_mask_ == False]
    y_cluster1 = y_train[ransac.inlier_mask_ == True]
    y_cluster2 = y_train[ransac.inlier_mask_ == False]

    # plt.figure()
    # categories = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Output']
    # flattened_data = x_train.flatten()
    # flattened_data = np.append(flattened_data, y_train)
    # category_labels = np.repeat(categories, x_train.shape[0])
    # hue = ['Cluster 2' if i == True else 'Cluster 1' for i in ransac.inlier_mask_]
    # hue = np.repeat(hue, x_train.shape[1])
    # for i in range (x_train.shape[1]):
    #     plt.subplot(2, 2, i + 1)
    #     plt.scatter(x_train[:, i], y_train, c = ransac.inlier_mask_, s = 50, cmap = 'viridis')
    #     plt.title('RANSAC Clustering - Feature ' + str(i + 1))
    # plt.title('RANSAC Model')

    return ransac, x_cluster1, x_cluster2, y_cluster1, y_cluster2, inlier_mask

def main():
    
    # Load datasets
    X_train = np.load(path + 'X_train_regression2.npy')
    y_train = np.load(path + 'y_train_regression2.npy')
    X_test = np.load(path + 'X_test_regression2.npy')

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    if scale:
        X_train_scaled = X_scaler.fit(X_train).transform(X_train)
        y_train_scaled = y_scaler.fit(y_train).transform(y_train)
        X_test_scaled = X_scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        y_train_scaled = y_train
        X_test_scaled = X_test

    print('-------------------------------------------------------')
    print('RANSAC Regression')
    
    alphas = np.arange(0.4, 1.2, 0.1)
    best_MSE = 1e100

    for alpha_c1 in alphas:
        
        # identify clusters and fit regressor to first cluster
        ransac1, x_cluster1_scaled1, x_cluster2_scaled1, y_cluster1_scaled1, y_cluster2_scaled1, inlier_mask = ransac_classifier(X_train_scaled, y_train_scaled.ravel(), alpha_c1)

        regr1 = ransac1.fit(x_cluster1_scaled1, y_cluster1_scaled1)

        MSE1, _, max_MSE1 = cv_metrics(regr1, x_cluster1_scaled1, y_cluster1_scaled1.ravel(), x_cluster1_scaled1.shape[0])
        
        for alpha_c2 in alphas:

            # fit regressor to second cluster
            ransac2, x_cluster1_scaled2, _, y_cluster1_scaled2, _, _ = ransac_classifier(x_cluster2_scaled1, y_cluster2_scaled1.ravel(), alpha_c2)

            regr2 = ransac2.fit(x_cluster1_scaled2, y_cluster1_scaled2)

            MSE2, _, max_MSE2 = cv_metrics(regr2, x_cluster2_scaled1, y_cluster2_scaled1.ravel(), x_cluster2_scaled1.shape[0])

            # find best threshold combination
            if MSE1 + MSE2 < best_MSE:
                best_MSE = MSE1 + MSE2
                best_alpha_c1 = alpha_c1
                best_alpha_c2 = alpha_c2

    # fit regressor to first cluster with best thresholds
    ransac1, x_cluster1_scaled1, x_cluster2_scaled1, y_cluster1_scaled1, y_cluster2_scaled1, inlier_mask = ransac_classifier(X_train_scaled, y_train_scaled.ravel(), best_alpha_c1)
    regr1 = ransac1.fit(x_cluster1_scaled1, y_cluster1_scaled1)
    MSE1, _, max_MSE1 = cv_metrics(regr1, x_cluster1_scaled1, y_cluster1_scaled1.ravel(), x_cluster1_scaled1.shape[0])

    # fit regressor to second cluster with best thresholds
    ransac2, x_cluster1_scaled2, _, y_cluster1_scaled2, _, _ = ransac_classifier(x_cluster2_scaled1, y_cluster2_scaled1.ravel(), best_alpha_c2)
    regr2 = ransac2.fit(x_cluster1_scaled2, y_cluster1_scaled2)
    MSE2, _, max_MSE2 = cv_metrics(regr2, x_cluster2_scaled1, y_cluster2_scaled1.ravel(), x_cluster2_scaled1.shape[0])
    
    print('x_cluster1.shape\t= ', x_cluster1_scaled1.shape)
    print('y_cluster1.shape\t= ', y_cluster1_scaled1.shape)
    print('x_cluster2.shape\t= ', x_cluster1_scaled2.shape)
    print('y_cluster2.shape\t= ', y_cluster1_scaled2.shape)

    # fit regressor to all data
    regr3 = LinearRegression().fit(X_train_scaled, y_train_scaled.ravel())
    MSE3, _, max_MSE3 = cv_metrics(regr3, X_train_scaled, y_train_scaled.ravel(), 10)

    print('best_alpha_c1\t= ', best_alpha_c1)
    print('best_alpha_c2\t= ', best_alpha_c2)
    
    # predict training data
    y_pred1_scaled = regr1.predict(X_train_scaled).reshape(y_train.shape[0], 1)
    y_pred2_scaled = regr2.predict(X_train_scaled).reshape(y_train.shape[0], 1)
    y_pred3_scaled = regr3.predict(X_train_scaled).reshape(y_train.shape[0], 1)
    
    if scale:
        y_pred1 = y_scaler.inverse_transform(y_pred1_scaled)
        y_pred2 = y_scaler.inverse_transform(y_pred2_scaled)
        y_pred3 = y_scaler.inverse_transform(y_pred3_scaled)
    else:
        y_pred1 = y_pred1_scaled
        y_pred2 = y_pred2_scaled
        y_pred3 = y_pred3_scaled

    print('cv_MSE\t= ', MSE1, '\t\t;\t', MSE2, '\t;\t', MSE3)
    print('max_MSE\t= ', max_MSE1, '\t\t;\t', max_MSE2, '\t;\t', max_MSE3)
    
    # Analyse results
    error_metrics(y_train.ravel(), y_pred1.ravel(), y_pred2.ravel(), y_pred3.ravel(), inlier_mask)

    # predict test data
    y_test_pred1_scaled = regr1.predict(X_test_scaled).reshape(X_test_scaled.shape[0], 1)
    y_test_pred2_scaled = regr2.predict(X_test_scaled).reshape(X_test_scaled.shape[0], 1)

    if scale:
        y_test_pred1 = y_scaler.inverse_transform(y_test_pred1_scaled)
        y_test_pred2 = y_scaler.inverse_transform(y_test_pred2_scaled)
    else:
        y_test_pred1 = y_test_pred1_scaled
        y_test_pred2 = y_test_pred2_scaled

    y_test_pred = np.hstack((y_test_pred1, y_test_pred2))
    np.save(path + 'y_test_regression2.npy', y_test_pred)

    # plt.show()
    print('-------------------------------------------------------')

if __name__ == "__main__":
    main()