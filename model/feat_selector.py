import logging
from sklearn.base import BaseEstimator, TransformerMixin

@Log(NAME_LOGGER)
class FeatsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.correlated_features = set()
        correlation_matrix = pd.DataFrame(data=X).corr()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > self.threshold:
                    self.correlated_features.add(i)
        return self

    def transform(self, X):
        use_columns = set(range(X.shape[1])) - self.correlated_features
        return X[:, list(use_columns)]