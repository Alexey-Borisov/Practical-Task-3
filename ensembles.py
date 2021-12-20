import numpy as np
import time
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


class RandomForestMSE:
    def __init__(
            self, n_estimators, max_depth=None, feature_subsample_size=None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        if self.feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3
        self.trees_parameters = trees_parameters
        self.ensemble = []
        self.history = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        self.history = {'time': [], 'pred_loss': []}
        start_time = time.time()
        for i in range(self.n_estimators):
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.trees_parameters)
            feature_subsample = np.random.choice(X.shape[1], size=int(X.shape[1] * self.feature_subsample_size),
                                                 replace=False)
            object_subsample = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            model.fit(X[object_subsample][:, feature_subsample], y[object_subsample])
            self.ensemble.append((model, feature_subsample))

            # time logging
            cur_time = time.time()
            self.history['time'].append(cur_time - start_time) 

        # validation
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val, None, y_val)
            val_score = rmse(y_val, y_val_pred)
            return val_score

    def predict(self, X, n_estimators=None, y_true=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        if n_estimators is None or n_estimators > self.n_estimators:
            n_estimators = self.n_estimators
        y_pred = np.zeros(X.shape[0])
        for cur_idx, (model, feature_subsample) in enumerate(self.ensemble[:n_estimators]):
            y_pred += model.predict(X[:, feature_subsample])

            if y_true is not None:
                loss = rmse(y_true, y_pred / (cur_idx + 1))
                self.history['pred_loss'].append(loss)
        return y_pred / n_estimators
    
    
class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_subsample_size = feature_subsample_size
        if self.feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3
        self.max_depth = max_depth
        self.trees_parameters = trees_parameters
        self.ensemble = []
        self.history = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        
        # loss function for one-dimensional optimization
        def loss(alpha):
            return ((f + alpha * model_pred - y)**2).mean()

        f = np.zeros_like(y, dtype=float)
        self.history = {'time': [], 'pred_loss': []}
        start_time = time.time()
        for i in range(self.n_estimators):
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.trees_parameters)
            feature_subsample = np.random.choice(X.shape[1], size=int(X.shape[1] * self.feature_subsample_size),
                                                 replace=False)
            model.fit(X[:, feature_subsample], (y - f))
            model_pred = model.predict(X[:, feature_subsample])

            optim_alpha = minimize_scalar(loss).x
            f += optim_alpha * self.learning_rate * model_pred
            self.ensemble.append((model, feature_subsample, optim_alpha * self.learning_rate))
            
            # time logging
            cur_time = time.time()
            self.history['time'].append(cur_time - start_time) 

        # validation
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val, None, y_val)
            val_score = rmse(y_val, y_val_pred)
            return val_score

    def predict(self, X, n_estimators=None, y_true=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        if n_estimators is None or n_estimators > self.n_estimators:
            n_estimators = self.n_estimators
        y_pred = np.zeros(X.shape[0])
        for model, feature_subsample, coef in self.ensemble[:n_estimators]:
            y_pred += coef * model.predict(X[:, feature_subsample])

            if y_true is not None:
                loss = rmse(y_true, y_pred)
                self.history['pred_loss'].append(loss)
        return y_pred
