import pandas as pd
import json
from pyspark.sql import SparkSession


class Config():
    def __init__(self):
        self.config = {}

    def set_transform_func(self, transform_func_dict):
        self.config['transform'] = {}
        self.config['transform']['cat2cat'] = transform_func_dict['cat2cat']
        self.config['transform']['cat2num'] = transform_func_dict['cat2num']
        self.config['transform']['num'] = transform_func_dict['num']
        self.config['transform']['num2num'] = transform_func_dict['num2num']

    def set_task(self, task):
        self.config['task'] = task

    def set_eval_model(self, eval_method_dict=None):
        self.config['eval_model'] = eval_method_dict

    def set_model_params(self, model_params):
        self.config['model_params'] = model_params

    def set_eval_metric(self, eval_metric):
        self.config['eval_metric'] = eval_metric

    def set_eval_method(self, eval_method):
        self.config['eval_method'] = eval_method

    def set_search_method(self, search_method):
        self.config['search_method'] = search_method

    def set_search_hyper_params(self, search_params_dict):
        self.config['search_params_dict'] = search_params_dict

    def set_save_sequence_path(self, path):
        self.config['save_seq_path'] = path

    def set_load_sequence_path(self, path):
        self.config['load_seq_path'] = path

    def set_save_model_path(self, path):
        self.config['save_model_path'] = path

    def set_load_model_path(self, path):
        self.config['load_model_path'] = path

    def set_pretrain_params_tasks(self, pretrain_params_tasks_dict):
        self.config['pretrain_params_tasks_dict'] = pretrain_params_tasks_dict

    def set_save_pretrain_model_path(self, save_pretrain_model_path):
        self.config['save_pretrain_model_path'] = save_pretrain_model_path

    def set_pretrain_params_hyperparams(self, pretrain_params_hyperparams):
        self.config['pretrain_params_hyperparams'] = pretrain_params_hyperparams

    def start_spark(self):
        import os
        os.environ["PYSPARK_PYTHON"] = "**"  # Python interpreter path
        os.environ["PYSPARK_DRIVER_PYTHON"] = "**"
        self.config['spark'] = SparkSession.builder.getOrCreate()


def load_data(data_path=None, schema_path=None, target_index=None):
    """ Load data from data path, and determine the feature type

    Args:
        data_path: data path
        schema_path: the path of file which storage feature types
        target_index: target column index
    
    Returns:
        Data, Target, Schema
    """
    df = pd.read_csv(data_path, header=None)
    
    if target_index is not None: 
        columns = df.columns.values
        df[[columns[target_index], columns[-1]]] = df[[columns[-1], columns[target_index]]]
        # target_col = df.columns[target_index]
    # else:
    target_col = df.columns[-1]

    def read_tables_info_json(datainfo):

        numFE = []
        catFE = []
        with open(datainfo, 'r') as datainfo_fp:
            d_info = json.load(datainfo_fp)

        for column, ctype in d_info.items():
            if ctype == "cat":
                catFE.append(int(column))
            else:
                numFE.append(int(column))

        return (d_info, catFE, numFE)

    Schema = read_tables_info_json(schema_path)

    Target = df.loc[:, target_col]

    Data = df.drop(columns=[target_col])

    return Data, Target, Schema

import importlib
from conf import settings

# import evaluator
evaluator_path = settings.EVALUATOR_PATH
evaluator_module = importlib.import_module(evaluator_path)


def get_evaluator_class_from_module():
    return getattr(evaluator_module, 'Evaluator')()


# import feature transformation functions

cat2cat_module_path = settings.CAT2CAT_MODULE_PATH
cat2num_module_path = settings.CAT2NUM_MODULE_PATH
num_module_path = settings.NUM_MODULE_PATH
num2num_module_path = settings.NUM2NUM_MODULE_PATH

cat2cat_module = importlib.import_module(cat2cat_module_path)
cat2num_module = importlib.import_module(cat2num_module_path)
num_module = importlib.import_module(num_module_path)
num2num_module = importlib.import_module(num2num_module_path)


def get_cat2cat_class_from_module(op_name):
    return getattr(cat2cat_module, op_name)()


def get_cat2num_class_from_module(op_name):
    return getattr(cat2num_module, op_name)()


def get_num_class_from_module(op_name):
    return getattr(num_module, op_name)()


def get_num2num_class_from_module(op_name):
    return getattr(num2num_module, op_name)()


def convert_to_transform_dict(cat2cat, cat2num, num, num2num):
    cat2cat_transform_dict = {}
    cat2num_transform_dict = {}
    num_transform_dict = {}
    num2num_transform_dict = {}

    for index, op_name in zip(range(1, len(cat2cat) + 1), cat2cat):
        cat2cat_transform_dict[index] = get_cat2cat_class_from_module(op_name)

    for index, op_name in zip(range(1, len(cat2num) + 1), cat2num):
        cat2num_transform_dict[index] = get_cat2num_class_from_module(op_name)

    for index, op_name in zip(range(-len(num), 0), num):
        num_transform_dict[index] = get_num_class_from_module(op_name)

    for index, op_name in zip(range(1, len(num2num) + 1), num2num):
        num2num_transform_dict[index] = get_num2num_class_from_module(op_name)
    return {'cat2cat': cat2cat_transform_dict, 'cat2num': cat2num_transform_dict, 'num': num_transform_dict,
            'num2num': num2num_transform_dict}
