from sklearn.preprocessing import MinMaxScaler
import numpy as np


class tf_num():

    def __init__(self):
        pass

    def transfrom(self, fe1):
        pass


class num_sqrt(tf_num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1):
        return np.sqrt(np.abs(fe1))

    def __repr__(self):
        return 'num_sqrt'


class num_minmaxscaler(tf_num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1):
        scaler = MinMaxScaler()
        return np.squeeze(scaler.fit_transform(np.reshape(fe1, [-1, 1])))

    def __repr__(self):
        return 'num_minmaxscaler'


class num_log(tf_num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1):
        while (np.any(fe1 == 0)):
            fe1 = fe1 + 1e-3
        return np.log(np.abs(fe1))

    def __repr__(self):
        return 'num_log'


class num_reciprocal(tf_num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1):
        while (np.any(fe1 == 0)):
            fe1 = fe1 + 1e-3
        return np.reciprocal(fe1)

    def __repr__(self):
        return 'num_reciprocal'


from pyspark.sql.functions import sqrt, log
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler as spark_MinMaxScalerM


class tf_num_spark():

    def __init__(self):
        pass

    def transfrom(self, fe1):
        pass


class num_spark_sqrt(tf_num_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, spark):
        df = fe1.to_frame()
        df.reset_index(inplace=True)
        columns = list('01')
        df.columns = columns

        df = spark.createDataFrame(df)

        df = df.withColumn('sqrt', sqrt('1'))

        df = df.toPandas()

        return df['sqrt']

    def __repr__(self):
        return 'num_spark_sqrt'


class num_spark_minmaxscaler(tf_num_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, spark):
        df = fe1.to_frame()
        df.reset_index(inplace=True)
        columns = list('01')
        df.columns = columns

        df = spark.createDataFrame(df)

        vecAssembler = VectorAssembler(inputCols=['1'], outputCol='assembler_1')
        df = vecAssembler.transform(df)

        mmScaler = spark_MinMaxScalerM(inputCol="assembler_1", outputCol="minmaxscale")

        model = mmScaler.fit(df)
        df = model.transform(df)

        Dissembler = udf(lambda v: float(v[0]))

        df = df.select(Dissembler('minmaxscale').alias('scaled'))
        df = df.toPandas()

        return df['scaled']

    def __repr__(self):
        return 'num_spark_minmaxscaler'


class num_spark_log(tf_num_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, spark):
        df = fe1.to_frame()
        df.reset_index(inplace=True)
        columns = list('01')

        df.columns = columns

        df = spark.createDataFrame(df)

        df = df.withColumn('log', log('1'))

        df = df.toPandas()

        return df['log']

    def __repr__(self):
        return 'num_spark_log'


class num_spark_reciprocal(tf_num_spark):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, spark):
        df = fe1.to_frame()
        df.reset_index(inplace=True)
        columns = list('01')
        df.columns = columns

        df = spark.createDataFrame(df)

        reciprocal = udf(lambda v: float(1 / (v + 0.001)))

        df = df.withColumn('reciprocal', reciprocal('1'))

        df = df.toPandas()
        return df['reciprocal']

    def __repr__(self):
        return 'num_spark_reciprocal'
