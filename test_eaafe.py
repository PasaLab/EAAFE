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

# load dataset and the feature information
data_path = 'data/pima.csv'
schema_path = 'data/pima.json'
X, y, Schema = load_data(data_path=data_path, schema_path=schema_path)

# set feature transformation functions, reference to `libs`
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
# set trainsformation function search space 
config.set_transform_func(transfrom_func_dict)
# set task type, classification or regression
config.set_task('classification')
# set evaluation algorithm, reference to `libs`
config.set_eval_model('random_forest_classifier')
# set evaluation algorithm parameters, empty donets default parameters
params_dict = {}
config.set_model_params(params_dict)
# set evaluation metrics
eval_metric = 'f1_micro'
config.set_eval_metric(eval_metric)
# set evalution types, cross_validate, hold_out or hold_out_spark
eval_method_dict = 'cross_validate'
config.set_eval_method(eval_method_dict)
# set search algorithm to EAAFE
search_method = 'EAAFE'
config.set_search_method(search_method)
# set search hyperparameters
EAAFE_search_params_dict = {
    'c_orders': 1,
    'n_orders': 2,
    'pop_size': 4,
    'n_generations': 2,
    'cross_rate': 0.4,
    'mutate_rate': 0.1,
}
config.set_search_hyper_params(EAAFE_search_params_dict)
logger.info('run')
config.set_save_sequence_path('pima_seq.pickle')
# instantiation
auto_fe = AutoFE(config=config, X=X, y=y, schema=Schema)
# start search
auto_fe.run()
