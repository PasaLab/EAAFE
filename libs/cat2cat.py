from numpy.lib.function_base import select
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_consistent_length, _num_samples

import numpy as np
import copy
import pandas as pd
import json
import random


class tf_cat2cat():

    def __init__(self):
        pass

    def transform(self, fe1, fe2):
        pass


class cat2cat_get_count_feature(tf_cat2cat):
    def __init__(self):
        super().__init__()

    def get_hash_feature(self, feature1, feature2):
        return feature1 + feature2 + feature1 * feature2

    def get_count_feature(self, feature):
        return (pd.DataFrame(feature).fillna(0)).groupby([0])[0].transform('count').values

    def transform(self, fe1, fe2):
        return self.get_count_feature(self.get_hash_feature(fe1, fe2))

    def __repr__(self):
        return 'cat2cat_get_count_feature'


class cat2cat_get_nunique_feature(tf_cat2cat):
    def __init__(self):
        super().__init__()

    def get_nunique_feature(self, feature1, feature2):
        feature = np.concatenate([np.reshape(feature1, [-1, 1]), np.reshape(feature2, [-1, 1])],
                                 axis=1)
        return (pd.DataFrame(feature).fillna(0)).groupby(0)[1].transform('nunique').values

    def transform(self, fe1, fe2):
        return self.get_nunique_feature(fe1, fe2)

    def __repr__(self):
        return 'cat2cat_get_nunique_feature'


from pyspark.sql.functions import countDistinct

class tf_cat2cat_spark():

    def __init__(self):
        pass

    def transform(self, fe1, fe2, spark):
        pass

class cat2cat_spark_get_count_feature(tf_cat2cat_spark):

    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2, spark):

        df = pd.concat([fe1, fe2], axis=1)
        df.reset_index(inplace=True)
        columns = list('012')

        df.columns = columns

        df = spark.createDataFrame(df)

        tmp_df = df.groupby(columns[1:]).count()

        df = df.join(tmp_df, on=columns[1:]).orderBy('0')

        df = df.toPandas()

        return df['count']


    def __repr__(self):
        return 'cat2cat_spark_get_count_feature'

class cat2cat_spark_get_nunique_feature(tf_cat2cat_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2, spark):

        df = pd.concat([fe1, fe2], axis=1)
        df.reset_index(inplace=True)
        columns = list('012')
        df.columns = columns

        df = spark.createDataFrame(df)

        tmp_df = df.groupby('1').agg(countDistinct('2').alias('nunique'))

        df = df.join(tmp_df, on='1').orderBy('0')

        df = df.toPandas()

        return df['nunique']

    def __repr__(self):
        return 'cat2cat_spark_get_nunique_feature'