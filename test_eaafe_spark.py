from util import load_data
from util import Config

from AutoFE import AutoFE

import logging

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")


setup_logging()

data_path = 'data/pima.csv'
schema_path = 'data/pima.json'
X, y, Schema = load_data(data_path=data_path, schema_path=schema_path)

spark_transfrom_func_dict = {'cat2cat': ['cat2cat_spark_get_count_feature',
                                         'cat2cat_spark_get_nunique_feature'
                                         ],
                             'cat2num': ['cat2num_spark_get_mean_feature'
                                         ],
                             'num': ['num_spark_sqrt',
                                     'num_spark_minmaxscaler',
                                     'num_spark_log',
                                     'num_spark_reciprocal'
                                     ],
                             'num2num': ['num2num_spark_add',
                                         'num2num_spark_sub',
                                         'num2num_spark_mul',
                                         'num2num_spark_div'
                                         ]
                             }

config = Config()
config.set_transform_func(spark_transfrom_func_dict)

config.set_task('classification')

config.set_eval_model('random_forest_spark_classifier')

params_dict = {}
config.set_model_params(params_dict)

eval_metric = 'f1'

config.set_eval_metric(eval_metric)

eval_method_dict = 'cross_validate_spark'

config.set_eval_method(eval_method_dict)

# run EAAFE by SPARK
search_method = 'EAAFE_SPARK'

# start spark
config.start_spark()

config.set_search_method(search_method)

EAAFE_search_params_dict = {
    'c_orders': 1,
    'n_orders': 2,
    'pop_size': 4,
    'cross_rate': 0.4,
    'mutate_rate': 0.1,
    'n_generations': 2,
}
config.set_search_hyper_params(EAAFE_search_params_dict)

config.set_save_sequence_path('pima_seq.pickle')

auto_fe = AutoFE(config=config, X=X, y=y, schema=Schema)

auto_fe.run()

auto_fe.save()
