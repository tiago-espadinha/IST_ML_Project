'''
Machine Learning - Linear Regression Problem 2
Authors:
Tiago Simões, 96329
Tomás Fonseca, 66325
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.model_selection import cross_validate

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans

path = '' # Change path to the folder where the data is located
save_ratios = 'Huber' # 'Huber', 'Thiel', 'Ransac' or False
show_plots = False

def error_metrics(y_real, x_real, model, save = False):
    y_pred_c1 = model[0].predict(x_real)
    y_pred_c2 = model[1].predict(x_real)

    err_sq_1 = np.square(y_real - y_pred_c1)
    err_sq_2 = np.square(y_real - y_pred_c2)
    y_sq = np.square(y_real)

    ratio_c1 = err_sq_1 / y_sq
    ratio_c2 = err_sq_2 / y_sq
    best_ratio = np.minimum(ratio_c1, ratio_c2)

    n_10 = (best_ratio>0.20).sum()
    print(n_10)

    if save:
        np.save(path + 'ratio1.npy', ratio_c1)
        np.save(path + 'ratio2.npy', ratio_c2)
        np.save(path + 'ratio.npy', best_ratio)

def gaussian_mixture(x_train, y_train):
    gmm = GaussianMixture(n_components = 2, covariance_type = 'full', random_state = 1)
    gmm.fit(x_train)
    clusters = gmm.predict(x_train)

    x_cluster1 = x_train[clusters == 0]
    x_cluster2 = x_train[clusters == 1]
    y_cluster1 = y_train[clusters == 0]
    y_cluster2 = y_train[clusters == 1]
    print('y_cluster1.shape = ', y_cluster1.shape)
    print('y_cluster2.shape = ', y_cluster2.shape)

    plt.figure()
    categories = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Output']
    flattened_data = x_train.flatten()
    flattened_data = np.append(flattened_data, y_train)
    category_labels = np.repeat(categories, x_train.shape[0])
    hue = ['Cluster 2' if i == 1 else 'Cluster 1' for i in clusters]
    hue = np.repeat(hue, x_train.shape[1]+ y_train.shape[1])
    #sns.stripplot(x=category_labels, y=flattened_data, jitter=True, hue=hue)
    for i in range (x_train.shape[1]):
        plt.subplot(2, 2, i + 1)
        plt.scatter(x_train[:,i], y_train, c=clusters, s=50, cmap='viridis')
        plt.title('Gaussian Mixture - Feature ' + str(i+1))
    
    return x_cluster1, x_cluster2, y_cluster1, y_cluster2

def bayesian_gaussian_mixture(x_train, y_train):
    gmm = BayesianGaussianMixture(n_components = 2, covariance_type = 'full', random_state = 1)
    gmm.fit(x_train)
    clusters = gmm.predict(x_train)

    x_cluster1 = x_train[clusters == 0]
    x_cluster2 = x_train[clusters == 1]
    y_cluster1 = y_train[clusters == 0]
    y_cluster2 = y_train[clusters == 1]

    plt.figure()
    categories = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Output']
    flattened_data = x_train.flatten()
    flattened_data = np.append(flattened_data, y_train)
    category_labels = np.repeat(categories, x_train.shape[0])
    hue = ['Cluster 2' if i == 1 else 'Cluster 1' for i in clusters]
    hue = np.repeat(hue, x_train.shape[1]+ y_train.shape[1])
    for i in range (x_train.shape[1]):
        plt.subplot(2, 2, i + 1)
        plt.scatter(x_train[:,i], y_train, c=clusters, s=50, cmap='viridis')
        plt.title('Bayesian Gaussian - Feature ' + str(i+1))
    
    return x_cluster1, x_cluster2, y_cluster1, y_cluster2

def kmeans(x_train, y_train):

    kmeans = KMeans(n_clusters = 2, random_state = 1)
    kmeans.fit(x_train)
    clusters = kmeans.predict(x_train)

    x_cluster1 = x_train[clusters == 0]
    x_cluster2 = x_train[clusters == 1]
    y_cluster1 = y_train[clusters == 0]
    y_cluster2 = y_train[clusters == 1]
    print('y_cluster1.shape = ', y_cluster1.shape)
    print('y_cluster2.shape = ', y_cluster2.shape)

    plt.figure()
    categories = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Output']
    flattened_data = x_train.flatten()
    flattened_data = np.append(flattened_data, y_train)
    category_labels = np.repeat(categories, x_train.shape[0])
    hue = ['Cluster 2' if i == 1 else 'Cluster 1' for i in clusters]
    hue = np.repeat(hue, x_train.shape[1]+ y_train.shape[1])
    #sns.stripplot(x=category_labels, y=flattened_data, jitter=True, hue=hue)

    for i in range (x_train.shape[1]):
        plt.subplot(2, 2, i + 1)
        plt.scatter(x_train[:,i], y_train, c=clusters, s=50, cmap='viridis')
        plt.title('K-Means Clustering - Feature ' + str(i+1))
    
    return x_cluster1, x_cluster2, y_cluster1, y_cluster2

def metrics(y_real, y_pred):
    if y_real.shape != y_pred.shape:
        print(f'y_real.shape = {y_real.shape}, y_pred.shape = {y_pred.shape}')
        print('Confirm that both of your inputs have the same shape!')
    else:
        SSE = mean_squared_error(y_real, y_pred) * y_real.shape[0]
        r2 = r2_score(y_real, y_pred)
        return SSE, r2

def cv_metrics(model, x, y, n_folds):
    cv_results = cross_validate(model, x, y, cv = n_folds, scoring = 'neg_mean_squared_error', return_train_score = True)
    
    avg_mse = np.mean(abs(cv_results['test_score']))
    folds_avg_mse = abs(cv_results['test_score'])
    print('cv_MSE = ', avg_mse)
    return avg_mse, folds_avg_mse

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

def huber_regression(x_train, y_train, alpha):
    huber = HuberRegressor(alpha=alpha).fit(x_train, y_train)
    huber_pred = huber.predict(x_train)
    cv_metrics(huber, x_train, y_train.ravel(), x_train.shape[0])
    return huber_pred, huber

def theil_sen_regression(x_train, y_train):
    theil_sen = TheilSenRegressor(random_state=1).fit(x_train, y_train)
    theil_sen_pred = theil_sen.predict(x_train)
    cv_metrics(theil_sen, x_train, y_train.ravel(), x_train.shape[0])
    return theil_sen_pred, theil_sen

def ransac_regression(x_train, y_train):
    ransac = RANSACRegressor(random_state=1).fit(x_train, y_train)
    ransac_pred = ransac.predict(x_train)
    cv_metrics(ransac, x_train, y_train.ravel(), x_train.shape[0])
    return ransac_pred, ransac

def main():
    
    x_train = np.load(path + 'X_train_regression2.npy')
    y_train = np.load(path + 'y_train_regression2.npy')
    x_test = np.load(path + 'X_test_regression2.npy')

    c1_x_train, c2_x_train, c1_y_train, c2_y_train = gaussian_mixture(x_train, y_train)
    c1_x_train, c2_x_train, c1_y_train, c2_y_train = bayesian_gaussian_mixture(x_train, y_train)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    c1_x_train_scaled = x_scaler.fit(c1_x_train).transform(c1_x_train)
    c1_y_train_scaled = y_scaler.fit(c1_y_train).transform(c1_y_train)
    c2_x_train_scaled = x_scaler.fit(c2_x_train).transform(c2_x_train)
    c2_y_train_scaled = y_scaler.fit(c2_y_train).transform(c2_y_train)

    x_train_scaled = x_scaler.fit(x_train).transform(x_train)
    y_train_scaled = y_scaler.fit(y_train).transform(y_train)
    # X_test_scaled = x_scaler.fit(x_test).transform(x_test)

    # Huber Regression -----------------------------------------------------------------------
    print('\n-------------------------------------------------------')
    print('Huber Regression')

    print('Cluster 1:\t ', end='')
    c1_y_pred, huber1=huber_regression(c1_x_train_scaled, c1_y_train_scaled.ravel(), 0.1)
    print('Cluster 2:\t ', end='')
    c2_y_pred, huber2=huber_regression(c2_x_train_scaled, c2_y_train_scaled.ravel(), 0.1)
    print('All data:\t ', end='')
    c3_y_pred, huber3=huber_regression(x_train_scaled, y_train.ravel(), 0.1)

    huber = [huber1, huber2, huber3]
    
    huber_save = False
    if save_ratios == 'Huber':
        huber_save = True

    error_metrics(y_train_scaled.ravel(), x_train_scaled, huber, huber_save)

    # Theil-Sen Regression -----------------------------------------------------------------------
    print('\n-------------------------------------------------------')
    print('Theil-Sen Regression')

    print('Cluster 1:\t ', end='')
    c1_y_pred, thiel1=theil_sen_regression(c1_x_train_scaled, c1_y_train_scaled.ravel())
    print('Cluster 2:\t ', end='')
    c2_y_pred, thiel2=theil_sen_regression(c2_x_train_scaled, c2_y_train_scaled.ravel())
    print('All data:\t ', end='')
    c3_y_pred, thiel3=theil_sen_regression(x_train_scaled, y_train_scaled.ravel())

    thiel = [thiel1, thiel2, thiel3]

    thiel_save = False
    if save_ratios == 'Thiel':
        thiel_save = True

    error_metrics(y_train_scaled.ravel(), x_train_scaled, thiel, thiel_save)

    # RANSAC Regression -----------------------------------------------------------------------
    print('\n-------------------------------------------------------')
    print('RANSAC Regression')

    print('Cluster 1:\t ', end='')
    c1_y_pred, ransac1=ransac_regression(c1_x_train_scaled, c1_y_train_scaled.ravel())
    print('Cluster 2:\t ', end='')
    c2_y_pred, ransac2=ransac_regression(c2_x_train_scaled, c2_y_train_scaled.ravel())
    print('All data:\t ', end='')
    c3_y_pred, ransac3=ransac_regression(x_train_scaled, y_train_scaled.ravel())
    ransac=[ransac1, ransac2, ransac3]
    
    ransac_save = False
    if save_ratios == 'Ransac':
        ransac_save = True

    error_metrics(y_train_scaled.ravel(), x_train_scaled, ransac, ransac_save)

    if show_plots:
        plt.show()
    return

    y_test_pred_LR = y_scaler.inverse_transform(y_test_pred_scaled_LR)
    np.save(path + 'y_test_regression1.npy', y_test_pred_LR)
    # print('SLR:\ty_test_pred\t=\n', y_test_pred_LR)

if __name__ == "__main__":
    main()