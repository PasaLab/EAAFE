import copy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from libs.one_minus_rae_score import one_minus_rae
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.feature import VectorAssembler

import importlib
from conf import settings

import collections

import logging

logger = logging.getLogger(__name__)

eval_model_path = settings.EVALUATOR_MODEL_PATH
eval_metrics_path = settings.EVALUATOR_METRIC_PATH
eval_model_module = importlib.import_module(eval_model_path)

try:
    eval_metrics_module = importlib.import_module(eval_metrics_path)
except ModuleNotFoundError:
    eval_metrics_module = importlib.import_module('sklearn.metrics')


def get_eval_model_from_module(eval_model_name, params):
    return getattr(eval_model_module, eval_model_name)(params)


def get_eval_metrics_from_module(eval_metric_name):
    return getattr(eval_metrics_module, eval_metric_name)

# evaluated data cache, save recently evaluated data for speed up
cache = collections.OrderedDict()

class Evaluator:

    def __init__(self):
        pass

    def init(self, eval_model_name, eval_model_params,
             eval_metric_name, eval_method,
             spark=None, task=None):

        self.eval_hyperparams = None

        self.eval_model = get_eval_model_from_module(eval_model_name, params=eval_model_params)

        self.evaluation_method = eval_method

        self.task = task

        self.metric = eval_metric_name

        self.spark = spark

        if eval_method == 'cross_validate':
            pass
        elif eval_method == 'hold_out':
            self.metric = get_eval_metrics_from_module(eval_metric_name)

    def cross_val_score(self, X, y):

        if self.metric == '1-rae':
            return cross_val_score(self.eval_model.model, X, y, scoring=make_scorer(one_minus_rae), cv=5).mean()

        return cross_val_score(self.eval_model.model, X, y, scoring=self.metric, cv=5, n_jobs=8).mean()

    def cross_val_score_spark(self, X, y):

        if self.task == 'classification':
            evaluator = MulticlassClassificationEvaluator(metricName=self.metric)
        elif self.task == 'regression':
            evaluator = RegressionEvaluator(metricName=self.metric)

        grid = ParamGridBuilder().build()
        crossvalidator = CrossValidator(estimator=self.eval_model.model, estimatorParamMaps=grid,
                                        evaluator=evaluator, numFolds=5)
        X.columns = range(0, len(X.columns))
        columns = [str(c) for c in X.columns]

        X['label'] = y
        X = X.astype('float')
        X = X.fillna(0)
        df = self.spark.createDataFrame(X)

        vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
        df = vecAssembler.transform(df)

        cvModel = crossvalidator.fit(df)

        return cvModel.avgMetrics[0]

    def holdout_val_score(self, X, y, test_size=0.2):
        train_X, test_X, train_Y, test_Y = train_test_split(
            X, y, test_size=test_size, random_state=0)
        self.eval_model.fit(train_X, train_Y)
        return self.metric(self.eval_model.predit(test_X), test_Y)

    def holdout_val_score_spark(self, X, y, test_size):

        if self.task == 'classification':
            evaluator = MulticlassClassificationEvaluator(metricName=self.metric)
        elif self.task == 'regression':
            evaluator = RegressionEvaluator(metricName=self.metric)
        grid = ParamGridBuilder().build()

        tvs = TrainValidationSplit(estimator=self.eval_model.model, estimatorParamMaps=grid,
                                   evaluator=evaluator, trainRatio=0.8)
        X.columns = range(0, len(X.columns))
        columns = [str(c) for c in X.columns]

        X['label'] = y
        X = X.astype('float')
        X = X.fillna(0)
        df = self.spark.createDataFrame(X)

        vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
        df = vecAssembler.transform(df)

        tvsModel = tvs.fit(df)

        return tvsModel.validationMetrics[0]

    def get_eval(self, X, y, test_size=0.2):
        """获取实际评估值
        """

        cache_key = self.get_cache_key(X)

        logger.info(f'cache key {cache_key}')
        print(cache)

        if cache_key in cache:
            logger.info('Use cached evaluation.')
            return cache[cache_key]

        if len(cache) >= 1000:
            cache.popitem(last=False)

        if self.evaluation_method == 'cross_validate':
            cache[cache_key] = self.cross_val_score(X, y)
            # return self.cross_val_score(X, y)
        elif self.evaluation_method == 'hold_out':
            cache[cache_key] = self.holdout_val_score(X, y, test_size=test_size)
            # return self.holdout_val_score(X, y, test_size=test_size)
        elif self.evaluation_method == 'cross_validate_spark':
            cache[cache_key] = self.cross_val_score_spark(X, y)
            # return self.cross_val_score_spark(X, y)
        elif self.evaluation_method == 'hold_out_spark':
            cache[cache_key] = self.holdout_val_score_spark(X, y, test_size=test_size)
            # return self.holdout_val_score_spark(X, y, test_size=test_size)
        print("============================== end eval =================================")
        return cache[cache_key]

    def get_cache_key(self, X):
        """Get dict key of current dataset

        The first row of data table as the key.

        Args:
            X: dataset
        Returns:
            key:
        """
        x1 = X.iloc[0, :]
        return tuple(sorted(x1))