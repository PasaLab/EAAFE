import numpy as np
import pandas as pd


class tf_num2num():
    def __init__(self):
        pass

    def transform(self, fe1, fe2):
        pass


class num2num_add(tf_num2num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2):
        return np.squeeze(fe1 + fe2)

    def __repr__(self):
        return 'num2num_add'


class num2num_sub(tf_num2num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2):
        return np.squeeze(fe1 - fe2)

    def __repr__(self):
        return 'num2num_sub'


class num2num_mul(tf_num2num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2):
        return np.squeeze(fe1 * fe2)

    def __repr__(self):
        return 'num2num_mul'


class num2num_div(tf_num2num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2):
        while (np.any(fe2 == 0)):
            fe2 = fe2 + 1e-3
        return np.squeeze(fe1 / fe2)

    def __repr__(self):
        return 'num2num_div'


from pyspark.sql.functions import col


class tf_num2num_spark():
    def __init__(self):
        pass

    def transform(self, fe1, fe2, spark):
        pass


class num2num_spark_add(tf_num2num_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2, spark):
        df = pd.concat([fe1, fe2], axis=1)
        df.reset_index(inplace=True)
        columns = list('012')

        df.columns = columns

        df = spark.createDataFrame(df)

        df = df.withColumn('add', col('1') + (col('2')))

        df = df.toPandas()

        return df['add']

    def __repr__(self):
        return 'num2num_spark_add'


class num2num_spark_sub(tf_num2num_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2, spark):
        df = pd.concat([fe1, fe2], axis=1)
        df.reset_index(inplace=True)
        columns = list('012')

        df.columns = columns

        df = spark.createDataFrame(df)

        df = df.withColumn('minus', col('1') - col('2'))

        df = df.toPandas()

        return df['minus']

    def __repr__(self):
        return 'num2num_spark_sub'


class num2num_spark_mul(tf_num2num_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2, spark):
        df = pd.concat([fe1, fe2], axis=1)
        df.reset_index(inplace=True)
        columns = list('012')

        df.columns = columns

        df = spark.createDataFrame(df)

        df = df.withColumn('mul', col('1') * col('2'))

        df = df.toPandas()

        return df['mul']

    def __repr__(self):
        return 'num2num_spark_mul'


class num2num_spark_div(tf_num2num_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2, spark):
        df = pd.concat([fe1, fe2], axis=1)
        df.reset_index(inplace=True)
        columns = list('012')

        df.columns = columns

        df = spark.createDataFrame(df)

        df = df.withColumn('div', col('1') / (col('2') + 0.001))

        df = df.toPandas()

        return df['div']

    def __repr__(self):
        return 'num2num_spark_div'
