class algorithm:

    def __init__(self):
        pass

    def model(self):
        pass

    def hyper_params(self):
        pass

    def get_model_name(self):
        pass

    def fit(self, X, y):
        pass

    def predit(self, X):
        pass


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class random_forest_regressor(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = RandomForestRegressor(**params)
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


class random_forest_classifier(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = RandomForestClassifier()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


class linearRegression(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = LinearRegression()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


class logisticRegression(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = LogisticRegression()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


import lightgbm as lgb


class LGBMRegressor(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = lgb.LGBMRegressor()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


class LGBMClassifier(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = lgb.LGBMClassifier()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


from pyspark.ml.classification import RandomForestClassifier as RandomForestClassifier_spark
from pyspark.ml.regression import RandomForestRegressor as RandomForestRegressor_spark


class random_forest_spark_regressor(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = RandomForestRegressor_spark(**params)
        # self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.transform(X=X)


class random_forest_spark_classifier(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = RandomForestClassifier_spark()
        # self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.transform(X=X)
