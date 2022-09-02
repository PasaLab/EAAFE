import argparse
import os
import logging
import copy
import json

from joblib import Parallel, delayed, parallel_backend

import numpy as np
import geatpy as ea
import pickle

import time

from util import convert_to_transform_dict
from util import get_evaluator_class_from_module
import logging

import ray

ray.init(ignore_reinit_error=True)

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")


class search_pipeline(ea.Problem):
    def __init__(self, config):

        name = 'AUTOFE'

        self.config = config

        self.datainfo, self.catFE, self.numFE = config['schema']

        self.columns = list(config['schema'][0].keys())

        self.columns = [int(fe) for fe in self.columns]

        self.cat_columns = config['schema'][1]
        self.num_columns = config['schema'][2]

        logger.info('columns %s', self.columns)

        self.transform_dict = convert_to_transform_dict(config['transform']['cat2cat'],
                                                        config['transform']['cat2num'],
                                                        config['transform']['num'],
                                                        config['transform']['num2num']
                                                        )
        # set evaluator
        self.evaluator = get_evaluator_class_from_module()
        self.evaluator.init(eval_model_name=config['eval_model'],
                            eval_model_params=config['model_params'],
                            eval_metric_name=config['eval_metric'],
                            eval_method=config['eval_method'])

        self.flag_seqs = []

        for i in range(len(self.columns)):
            if self.columns[i] in self.cat_columns:
                for j in range(self.config['search_params_dict']['c_orders']):
                    self.flag_seqs.append('cat')
            else:
                for j in range(self.config['search_params_dict']['n_orders']):
                    self.flag_seqs.append('num')

        logger.info('flag_seqs %s', self.flag_seqs)

        self.FES_NUM = len(self.datainfo)

        self.NUMS_NUM = len(self.numFE)

        self.CATS_NUM = len(self.catFE)

        self.len = self.CATS_NUM * config['search_params_dict']['c_orders'] + self.NUMS_NUM * \
                   config['search_params_dict'][
                       'n_orders']

        self.cat2cat = config['transform']['cat2cat']
        self.cat2num = config['transform']['cat2num']
        self.num = config['transform']['num']
        self.num2num = config['transform']['num2num']

        self.cat2cat_op_num = len(self.cat2cat)
        self.cat2num_op_num = len(self.cat2num)
        self.num_op_num = len(self.num)
        self.num2num_op_num = len(self.num2num)

        self.cat_availables_op = []
        self.num_availables_op = []

        if self.CATS_NUM != 0:
            self.cat_availables_op = list(
                range(0, self.FES_NUM * max(self.cat2cat_op_num + 1, self.cat2num_op_num + 1)))
        if self.NUMS_NUM != 0:
            self.num_availables_op = list(range(-self.NUMS_NUM * (self.num2num_op_num + 1) - self.num_op_num, 0))

        self.availables = self.num_availables_op + self.cat_availables_op

        self.availables_len = len(self.availables)

        self.num_availables_op = self.num_availables_op + [0]

        logger.info('cat_availables: %s', self.cat_availables_op)
        logger.info('num_availables: %s', self.num_availables_op)
        logger.info('availables: %s', self.availables)

        self.X = config['X']
        self.y = config['y']

        self.task = config['task']

        self.POP_SIZE = config['search_params_dict']['pop_size']
        self.CROSS_RATE = config['search_params_dict']['cross_rate']
        self.MUTATE_RATE = config['search_params_dict']['mutate_rate']
        self.N_GENERATIONS = config['search_params_dict']['n_generations']

        self.CV = 5
        self.SCORING = 'f1_micro'
        self.N_ESTIMATORS = 5

        self.max_fitness = 0
        self.count = 1

        M = 1  # dimension of target
        maxormins = [-1]  # maximize or minize the target

        Dim = self.len

        varTypes = [1] * Dim

        # constrain lowwer bound
        lb = []
        for i in range(self.FES_NUM):
            if i in self.catFE:
                lb = lb + [min(self.cat_availables_op)] * config['search_params_dict']['c_orders']
            else:
                lb = lb + [min(self.num_availables_op)] * config['search_params_dict']['n_orders']

        ub = []
        numFE_index = []
        catFE_index = []
        index = 0
        fe_index_dict = {}
        fe_split_index = []

        # constrain upper bound
        for i in range(self.FES_NUM):
            if i in self.catFE:
                ub = ub + [max(self.cat_availables_op)] * config['search_params_dict']['c_orders']
                catFE_index = catFE_index + list(range(index, index + config['search_params_dict']['c_orders']))
                fe_index_dict[i] = list(range(index, index + config['search_params_dict']['c_orders']))
                index = index + config['search_params_dict']['c_orders']
            else:
                ub = ub + [max(self.num_availables_op)] * config['search_params_dict']['n_orders']
                numFE_index = numFE_index + list(range(index, index + config['search_params_dict']['n_orders']))
                fe_index_dict[i] = list(range(index, index + config['search_params_dict']['n_orders']))
                index = index + config['search_params_dict']['n_orders']
            fe_split_index.append(index)

        self.numFE_index = numFE_index
        self.catFE_index = catFE_index
        self.fe_index_dict = fe_index_dict
        self.fe_split_index = fe_split_index[:-1]

        lbin = [1] * Dim
        ubin = [1] * Dim

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    @ray.remote
    def evaluate(self, Var):
        X = copy.deepcopy(self.X)
        y = copy.deepcopy(self.y)
        DNA = np.hsplit(Var, self.fe_split_index)

        ops = []

        for feName, actions in zip(range(self.FES_NUM), DNA):

            actions2ops = []
            fe = X[feName].values

            if feName in self.cat_columns:
                for action in actions:

                    target_index = int(np.mod(action, self.FES_NUM))

                    fe2Name = self.columns[target_index]

                    if fe2Name in self.cat_columns:
                        fe2 = X[fe2Name].values

                        if self.cat2cat_op_num > self.cat2num_op_num:
                            action_binary = action // self.FES_NUM
                        else:
                            action_binary = action % (self.cat2cat_op_num + 1)

                        if action_binary == 0:
                            break
                        elif action_binary < 0:
                            break
                        else:
                            actions2ops.append(
                                self.transform_dict['cat2cat'][action_binary].__repr__() + '(' + str(fe2Name) + ')')
                            fe = self.transform_dict['cat2cat'][action_binary].transform(fe, fe2)

                    elif fe2Name in self.num_columns:

                        fe2 = X[fe2Name].values

                        if self.cat2cat_op_num < self.cat2num_op_num:
                            action_binary = action // self.FES_NUM
                        else:
                            action_binary = action % (self.cat2num_op_num + 1)

                        if action_binary == 0:
                            break
                        elif action_binary < 0:
                            break
                        else:
                            actions2ops.append(
                                self.transform_dict['cat2num'][action_binary].__repr__() + '(' + str(fe2Name) + ')')
                            fe = self.transform_dict['cat2num'][action_binary].transform(fe, fe2)

            elif feName in self.num_columns:

                for action in actions:

                    if action in range(-self.num_op_num, 1):
                        action = int(action)
                        if action == 0:
                            break
                        # elif action < 0:
                        #     break
                        else:
                            actions2ops.append(self.transform_dict['num'][action].__repr__())
                            fe = self.transform_dict['num'][action].transform(fe)
                    else:
                        action = -(action + self.num_op_num + 1)
                        action_binary = action // (self.NUMS_NUM)
                        target_index = int(np.mod(action, self.NUMS_NUM))
                        fe2Name = self.num_columns[target_index]
                        if fe2Name == feName:
                            continue
                        fe2 = X[fe2Name].values

                        if action_binary == 0:
                            break
                        elif action_binary < 0:
                            break
                        else:
                            actions2ops.append(
                                self.transform_dict['num2num'][action_binary].__repr__() + '(' + str(fe2Name) + ')')
                            fe = self.transform_dict['num2num'][action_binary].transform(fe, fe2)

            if len(actions2ops) == 0:
                new_name = str(feName)
            else:
                new_name = str(feName) + '_' + '-'.join(actions2ops)

            X[new_name] = fe
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(-1, inplace=True)

        res = self.evaluator.get_eval(X=X, y=y)
        logger.info('res %s ,trans sequence %s', res, list(Var))
        return res

    def transform_data(self, Var):
        X = copy.deepcopy(self.X)
        y = copy.deepcopy(self.y)
        Var = np.array(Var)
        DNA = np.hsplit(Var, self.fe_split_index)

        ops = []

        for feName, actions in zip(range(self.FES_NUM), DNA):

            actions2ops = []
            fe = X[feName].values

            if feName in self.cat_columns:
                for action in actions:

                    target_index = int(np.mod(action, self.FES_NUM))

                    fe2Name = self.columns[target_index]

                    if fe2Name in self.cat_columns:
                        fe2 = X[fe2Name].values

                        if self.cat2cat_op_num > self.cat2num_op_num:
                            action_binary = action // self.FES_NUM
                        else:
                            action_binary = action % (self.cat2cat_op_num + 1)

                        if action_binary == 0:
                            break
                        elif action_binary < 0:
                            break
                        else:
                            actions2ops.append(
                                self.transform_dict['cat2cat'][action_binary].__repr__() + '(' + str(fe2Name) + ')')
                            fe = self.transform_dict['cat2cat'][action_binary].transform(fe, fe2)

                    elif fe2Name in self.num_columns:

                        fe2 = X[fe2Name].values

                        if self.cat2cat_op_num < self.cat2num_op_num:
                            action_binary = action // self.FES_NUM
                        else:
                            action_binary = action % (self.cat2num_op_num + 1)

                        if action_binary == 0:
                            break
                        elif action_binary < 0:
                            break
                        else:
                            actions2ops.append(
                                self.transform_dict['cat2num'][action_binary].__repr__() + '(' + str(fe2Name) + ')')
                            fe = self.transform_dict['cat2num'][action_binary].transform(fe, fe2)

            elif feName in self.num_columns:

                for action in actions:

                    if action in range(-self.num_op_num, 1):
                        action = int(action)
                        if action == 0:
                            break
                        else:
                            actions2ops.append(self.transform_dict['num'][action].__repr__())
                            fe = self.transform_dict['num'][action].transform(fe)
                    else:
                        action = -(action + self.num_op_num + 1)
                        action_binary = action // (self.NUMS_NUM)
                        target_index = int(np.mod(action, self.NUMS_NUM))
                        fe2Name = self.num_columns[target_index]
                        if fe2Name == feName:
                            continue
                        fe2 = X[fe2Name].values

                        if action_binary == 0:
                            break
                        elif action_binary < 0:
                            break
                        else:
                            actions2ops.append(
                                self.transform_dict['num2num'][action_binary].__repr__() + '(' + str(fe2Name) + ')')
                            fe = self.transform_dict['num2num'][action_binary].transform(fe, fe2)

            if len(actions2ops) == 0:
                new_name = str(feName)
            else:
                new_name = str(feName) + '_' + '-'.join(actions2ops)

            X[new_name] = fe
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(-1, inplace=True)

        X['y'] = y
        return X

    def aimFunc(self, pop):

        Vars = pop.Phen
        fitness = []
        # ray
        fitness = ray.get([self.evaluate.remote(self, Var) for Var in Vars])

        cur = max(fitness)
        logger.info('iter: %s, curr_best: %s, his_best: %s', self.count, cur, self.max_fitness)

        self.count = self.count + 1
        self.max_fitness = max(cur, self.max_fitness)

        fitness = (-np.log(1 - np.array(fitness))) ** 10
        pop.ObjV = np.array([fitness]).T


class SEARCH_FE_PIPELINE(object):

    def __init__(self, config):
        super().__init__()

        self.problem = search_pipeline(config=config)

    def transform_sequence(self):

        if self.problem.config['load_seq_path'] != None:
            with open(self.problem.config['load_seq_path'], 'rb') as fr:
                Var = pickle.load(fr)
                transformed_data = self.problem.transform_data(Var)
        return transformed_data

    def run(self):
        problem = self.problem

        NIND = self.problem.config['search_params_dict']['pop_size']

        # Different search space for different featrue type.

        Encodings = ['RI'] * problem.FES_NUM

        Fields = []

        for i in range(problem.FES_NUM):
            curr_index = problem.fe_index_dict[i]
            curr_field = ea.crtfld(Encodings[i], problem.varTypes[curr_index], problem.ranges[:, curr_index],
                                   problem.borders[:, curr_index])
            Fields.append(curr_field)

        population = ea.PsyPopulation(Encodings, Fields, NIND)

        eaAlgorithm = ea.soea_psy_SGA_templet(problem, population)

        eaAlgorithm.verbose = False

        eaAlgorithm.drawing = 0

        eaAlgorithm.MAXGEN = self.problem.config['search_params_dict']['n_generations']

        [BestIndi, population] = eaAlgorithm.run()

        best_seq = list(BestIndi.Phen[0])

        logger.info('The best feature transformation function sequence is : %s', best_seq)

        with open(self.problem.config['save_seq_path'], 'wb') as fw:
            pickle.dump(best_seq, fw)
