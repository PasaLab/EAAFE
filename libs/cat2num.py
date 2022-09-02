import numpy as np
import pandas as pd


class tf_cat2num():

    def __init__(self):
        pass

    def transform(self, fe1, fe2):
        pass


class cat2num_get_mean_feature(tf_cat2num):
    def __init__(self):
        super().__init__()

    def get_cat2num_mean_feature(self, feature1, feature2):
        feature = np.concatenate([np.reshape(feature1, [-1, 1]), np.reshape(feature2, [-1, 1])],
                                 axis=1)
        return (pd.DataFrame(feature).fillna(0)).groupby(0)[1].transform('mean').values

    def transform(self, fe1, fe2):
        return self.get_cat2num_mean_feature(fe1, fe2)

    def __repr__(self):
        return 'cat2num_get_mean_feature'


from pyspark.sql.functions import mean

class tf_cat2num_spark():

    def __init__(self):
        pass

    def transform(self, fe1, fe2, spark):
        pass

class cat2num_spark_get_mean_feature(tf_cat2num_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2, spark):
        df = pd.concat([fe1, fe2], axis=1)
        df.reset_index(inplace=True)
        columns = list('012')
        df.columns = columns

        df = spark.createDataFrame(df)
        tmp_df = df.groupby('1').agg(mean('2').alias('mean'))

        df = df.join(tmp_df, on='1').orderBy('0')

        df = df.toPandas()

        return df['mean']

    def __repr__(self):
        return 'cat2num_spark_get_mean_feature'
