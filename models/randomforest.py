import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    """
    Random forest model for baseline binary classification.
    """

    def __init__(self, input_dim, seq_len, n_estimators=500, max_depth=25):
        """
        Initialize the Random Forest model.

        :param input_dim: Number of features in the input data (e.g., number of sensors).
        :param n_estimators: Number of trees in the forest (default: 100).
        :param max_depth: Maximum depth of the trees (default: None).
        """
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced_subsample',
            min_samples_split=10,
            min_samples_leaf=4
        )

    def _summarize(self, X):
        """
        Convert the input data to a 2D array for the Random Forest model.
        :param X: Array of shape (num_samples, seq_len, input_dim).
        :return: mean, std, min, max of the input data
        """
        if X.ndim == 3:
            mean = np.mean(X, axis=1)
            std = np.std(X, axis=1)
            min_val = np.min(X, axis=1)
            max_val = np.max(X, axis=1)
            return np.concatenate((mean, std, min_val, max_val), axis=1)
        elif X.ndim == 2:
            return X # Already in 2D format
        else:
            raise ValueError("Unexpected input shape for RandomForestModel: {}".format(X.shape))

    def fit(self, X, y):
        """
        Fit the Random Forest model to the training data.

        :param X: Training data of shape (num_samples, seq_len, input_dim).
        :param y: Labels of shape (num_samples,).
        """

        X = self._summarize(X)
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the binary labels for the input data.

        :param X: Input data of shape (num_samples, seq_len, input_dim).
        :return: Predicted labels of shape (num_samples,).
        """
        X = self._summarize(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict the probabilities for the input data.

        :param X: Input data of shape (num_samples, seq_len, input_dim).
        :return: Predicted probabilities of shape (num_samples, 2).
        """
        X = self._summarize(X)
        return self.model.predict_proba(X)[:, 1] # Get probabilities for the positive class