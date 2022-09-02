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

"""
Get transformed data.
"""

data_path = 'data/pima.csv'
schema_path = 'data/pima.json'
X, y, Schema = load_data(data_path=data_path, schema_path=schema_path)

transfrom_func_dict = {'cat2cat': ['cat2cat_get_count_feature',
                                   'cat2cat_get_nunique_feature'
                                   ],
                       'cat2num': ['cat2num_get_mean_feature'
                                   ],
                       'num': ['num_sqrt',
                               'num_minmaxscaler',
                               'num_log',
                               'num_reciprocal'
                               ],
                       'num2num': ['num2num_add',
                                   'num2num_sub',
                                   'num2num_mul',
                                   'num2num_div'
                                   ]
                       }

config = Config()
config.set_transform_func(transfrom_func_dict)
# config.set_transform_func(spark_transfrom_func_dict)

config.set_task('classification')
config.set_eval_model('random_forest_classifier')

params_dict = {}
config.set_model_params(params_dict)

eval_metric = 'f1_micro'

config.set_eval_metric(eval_metric)

eval_method_dict = 'cross_validate'

config.set_eval_method(eval_method_dict)

search_method = 'EAAFE'

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

config.set_load_sequence_path('pima_seq.pickle')

auto_fe = AutoFE(config=config, X=X, y=y, schema=Schema)

# get transformed data
data = auto_fe.transform_sequence()

print(data)
