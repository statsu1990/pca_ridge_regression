import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

import data
import pca_ridge

# ############
# data
# ############
sample_num=30
multico_x_nums_inSet=[3, 2, 1]
error_stds_inSet=[0.1] * 3
coefs_inSet=[0.6, -0.6, 1.0]
bias=0.3
error_std_y = 0.2

datamultico = data.DataMultico(sample_num, multico_x_nums_inSet, error_stds_inSet, 
                               coefs_inSet, bias, error_std_y)
datamultico.make_data()

print('x')
print(datamultico.x)
print('y')
print(datamultico.y)

print('corrcoef')
print(np.corrcoef(datamultico.x.transpose()))

print('--------------------------')
print('\nref coef')
print(datamultico.REF_COEF)
print('bias')
print(bias)
print('--------------------------')


# #############
# regression
# #############
def rmse(_y, _pre_y):
    return np.sqrt(np.average(np.square(_pre_y - _y)))

# linear regression
print('--------------------------')
linear_reg = LinearRegression()
linear_reg.fit(datamultico.x, datamultico.y)
print('\nlinear_reg coef')
print(linear_reg.coef_)
print('bias')
print(linear_reg.intercept_)
print('rmse')
print(rmse(datamultico.y, linear_reg.predict(datamultico.x)))
print('--------------------------')

# ridge regression
print('--------------------------')
ridge_alphas = [0.0001, 0.001, 0.1, 10, 1000]
for ridge_alpha in ridge_alphas:
    ridge_reg = Ridge(alpha=ridge_alpha)
    ridge_reg.fit(datamultico.x, datamultico.y)
    print('\nridge_reg {0} coef'.format(ridge_alpha))
    print(ridge_reg.coef_)
    print('bias')
    print(ridge_reg.intercept_)
    print('rmse')
    print(rmse(datamultico.y, ridge_reg.predict(datamultico.x)))
print('--------------------------')

# pca ridge regression
print('--------------------------')
pca_n_componets = 0.99
pca_ridge_alphas = [0.0001, 0.001, 0.1, 10, 1000]
for ridge_alpha in ridge_alphas:
    pca_ridge_reg = pca_ridge.PCA_RIDGE_ver1(n_components=pca_n_componets, alpha=ridge_alpha)
    pca_ridge_reg.fit(datamultico.x, datamultico.y)
    print('\npca ridge_reg ver1 {0} coef'.format(ridge_alpha))
    print(pca_ridge_reg.coef_)
    print('bias')
    print(pca_ridge_reg.intercept_)
    print('rmse')
    print(rmse(datamultico.y, pca_ridge_reg.predict(datamultico.x)))
print('--------------------------')

# pca ridge regression
print('--------------------------')
pca_n_componets = 0.99
pca_ridge_alphas = [0.0001, 0.001, 0.1, 10, 1000]
for ridge_alpha in ridge_alphas:
    pca_ridge_reg = pca_ridge.PCA_RIDGE_ver2(n_components=pca_n_componets, alpha=ridge_alpha)
    pca_ridge_reg.fit(datamultico.x, datamultico.y)
    print('\npca ridge_reg ver2 {0} coef'.format(ridge_alpha))
    print(pca_ridge_reg.coef_)
    print('bias')
    print(pca_ridge_reg.intercept_)
    print('rmse')
    print(rmse(datamultico.y, pca_ridge_reg.predict(datamultico.x)))

print('--------------------------')


