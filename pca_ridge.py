import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

class PCA_RIDGE_ver1():
    """
    ridge after pca
    """
    def __init__(self, n_components, alpha):
        self.N_COMPONENTS = n_components
        self.ALPHA = alpha
        return

    def fit(self, x, y):
        # pca
        self.pca = PCA(n_components=self.N_COMPONENTS)
        self.pca.fit(x)
        # x done pca
        pca_x = self.pca.transform(x)
        # ridge
        self.ridge_reg = Ridge(alpha=self.ALPHA)
        self.ridge_reg.fit(pca_x, y)

        # coef, intercept_
        self.coef_ = np.dot(self.ridge_reg.coef_, self.pca.components_)
        self.intercept_ = self.ridge_reg.intercept_ - np.dot(self.coef_, self.pca.mean_)

        return

    def predict(self, x):
        pca_x = self.pca.transform(x)
        y = self.ridge_reg.predict(pca_x)
        return y

class PCA_RIDGE_ver2():
    """
    ridge after pca.
    ridge coef and intercept including pca coef and mean
    """
    def __init__(self, n_components, alpha):
        self.N_COMPONENTS = n_components
        self.ALPHA = alpha
        return

    def fit(self, x, y):
        # pca
        pca = PCA(n_components=self.N_COMPONENTS)
        pca.fit(x)
        # x done pca
        pca_x = pca.transform(x)
        # temp ridge
        temp_ridge_reg = Ridge(alpha=self.ALPHA)
        temp_ridge_reg.fit(pca_x, y)

        # ridge
        self.ridge_reg = Ridge()
        self.ridge_reg.coef_ = np.dot(temp_ridge_reg.coef_, pca.components_)
        self.ridge_reg.intercept_ = temp_ridge_reg.intercept_ - np.dot(self.ridge_reg.coef_, pca.mean_)

        # coef, intercept_
        self.coef_ = self.ridge_reg.coef_
        self.intercept_ = self.ridge_reg.intercept_

        return

    def predict(self, x):
        y = self.ridge_reg.predict(x)
        return y


