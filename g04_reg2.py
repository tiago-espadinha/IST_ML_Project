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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.model_selection import cross_validate

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
import seaborn as sns


path = '' # Change path to the folder where the data is located
k_save = 5 # k folds to save
model_save = 'R' # model to save (R - Ridge, L - Lasso, EN - ElasticNet)
single_model = False # True to run only the model specified in model_save

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
    #sns.stripplot(x=category_labels, y=flattened_data, jitter=True, hue=hue)
    for i in range (x_train.shape[1]):
        plt.subplot(2, 2, i + 1)
        plt.scatter(x_train[:,i], y_train, c=clusters, s=50, cmap='viridis')
        plt.title('K-Means Clustering - Feature ' + str(i+1))
    plt.title('Gaussian Mixture Model')
    
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

def cv_metrics(model, x, y, k):
    scoring = 'neg_mean_squared_error'
    cv_results = cross_validate(model, x, y, cv = k, scoring = scoring, return_train_score = True)
    
    # print('cv_results\t= ', cv_results)
    
    avg_mse = np.mean(abs(cv_results['test_score']))
    folds_avg_mse = abs(cv_results['test_score'])
    return avg_mse, folds_avg_mse

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

def huber_regression(x_train, y_train, alpha):
    huber = HuberRegressor(alpha=alpha).fit(x_train, y_train)
    huber_pred = huber.predict(x_train)
    return huber_pred, huber

def theil_sen_regression(x_train, y_train):
    theil_sen = TheilSenRegressor(random_state=1).fit(x_train, y_train)
    theil_sen_pred = theil_sen.predict(x_train)
    return theil_sen_pred, theil_sen

def ransac_regression(x_train, y_train):
    ransac = RANSACRegressor(random_state=1).fit(x_train, y_train)
    ransac_pred = ransac.predict(x_train)
    return ransac_pred, ransac

def main():
    
    X_train = np.load(path + 'X_train_regression2.npy')
    y_train = np.load(path + 'y_train_regression2.npy')
    X_test = np.load(path + 'X_test_regression2.npy')

    #plot_features(X_train, y_train) # one possible outilier
    c1_x_train, c2_x_train, c1_y_train, c2_y_train = gaussian_mixture(X_train, y_train)
    #kmeans(X_train, y_train)
    plt.show()
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit(X_train).transform(X_train)
    y_train_scaled = y_scaler.fit(y_train).transform(y_train)
    X_test_scaled = X_scaler.fit(X_test).transform(X_test)

    #plot_features(X_train_scaled, y_train_scaled)

    k_tests = [3, 5] # k folds to test
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]


    # Huber Regression -----------------------------------------------------------------------
    print('-------------------------------------------------------')
    print('Huber Regression')
    c1_y_pred, huber1=huber_regression(c1_x_train, c1_y_train.ravel(), 0.1)
    c2_y_pred, huber2=huber_regression(c2_x_train, c2_y_train.ravel(), 0.1)
    c3_y_pred, huber3=huber_regression(X_train, y_train.ravel(), 0.1)

    SSE1, folds_SSE1 = cv_metrics(huber1, c1_x_train, c1_y_train.ravel(), 10)
    SSE2, folds_SSE2 = cv_metrics(huber2, c2_x_train, c2_y_train.ravel(), 10)
    SSE3, folds_SSE3 = cv_metrics(huber3, X_train, y_train.ravel(), 10)

    print('cv_SSE\t\t= ', SSE1, ' + ', SSE2, ' + ', SSE3)
    print('folds_cv_SSE\t= ', folds_SSE1, ' + ', folds_SSE2, ' + ', folds_SSE3)


    # Theil-Sen Regression -----------------------------------------------------------------------
    print('-------------------------------------------------------')
    print('Theil-Sen Regression')
    c1_y_pred, thiel1=theil_sen_regression(c1_x_train, c1_y_train.ravel())
    c2_y_pred, thiel2=theil_sen_regression(c2_x_train, c2_y_train.ravel())
    c3_y_pred, thiel3=theil_sen_regression(X_train, y_train.ravel())

    SSE1, folds_SSE1 = cv_metrics(thiel1, c1_x_train, c1_y_train.ravel(), 10)
    SSE2, folds_SSE2 = cv_metrics(thiel2, c2_x_train, c2_y_train.ravel(), 10)
    SSE3, folds_SSE3 = cv_metrics(thiel3, X_train, y_train.ravel(), 10)
    
    print('cv_SSE\t\t= ', SSE1, ' + ', SSE2, ' + ', SSE3)
    print('folds_cv_SSE\t= ', folds_SSE1, ' + ', folds_SSE2, ' + ', folds_SSE3)

    # RANSAC Regression -----------------------------------------------------------------------
    print('-------------------------------------------------------')
    print('RANSAC Regression')
    c1_y_pred, ransac1=ransac_regression(c1_x_train, c1_y_train.ravel())
    c2_y_pred, ransac2=ransac_regression(c2_x_train, c2_y_train.ravel())
    c3_y_pred, ransac3=ransac_regression(X_train, y_train.ravel())

    SSE1, folds_SSE1 = cv_metrics(ransac1, c1_x_train, c1_y_train.ravel(), 10)
    SSE2, folds_SSE2 = cv_metrics(ransac2, c2_x_train, c2_y_train.ravel(), 10)
    SSE3, folds_SSE3 = cv_metrics(ransac3, X_train, y_train.ravel(), 10)
    
    print('cv_SSE\t\t= ', SSE1, ' + ', SSE2, ' + ', SSE3)
    print('folds_cv_SSE\t= ', folds_SSE1, ' + ', folds_SSE2, ' + ', folds_SSE3)
    return

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

if __name__ == "__main__":
    main()