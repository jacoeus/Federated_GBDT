#!/usr/bin/env python    
# -*- coding: utf-8 -*-

# =============================================================================
# DecisionTree Base Class
# =============================================================================

from .algorithm_prototype import BasicAlgorithms
from tree_core.splitter import \
    SplitInfo, Splitter
from .node import Node
from .feature_histogram import \
    HistogramBag, FeatureHistogram
from typing import List


class DecisionTree(BasicAlgorithms):

    def __init__(self, criterion_method="xgboost", criterion_params=[0.1], max_depth=5,
                 min_sample_split=2, min_imputiry_split=1e-3, min_leaf_node=1,
                 max_split_nodes=2 ** 16, feature_importance_type="split",
                 n_iter_no_change=True, tol=0.001,
                 use_missing=False, zero_as_missing=False,):

        # input parameters
        self.criterion_method = criterion_method
        self.criterion_params = criterion_params
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_impurity_split = min_imputiry_split
        self.min_leaf_node = min_leaf_node
        self.max_split_nodes = max_split_nodes
        self.feature_importance_type = feature_importance_type
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.use_missing = use_missing
        self.zero_as_missing = zero_as_missing

        # transfer var
        self.transfer_inst = None

        # runtime variable
        self.feature_importance = {}
        self.tree_node = []
        self.cur_layer_nodes = []
        self.cur_to_split_nodes = []
        self.tree_node_num = 0
        self.runtime_idx = None
        self.valid_features = []
        self.sample_weights = Node
        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node)
        self.inst2node_idx = None  # record the node id an instance belongs to
        self.sample_weights = None

        # data
        self.data_bin = None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.data_with_node_assignments = None

        # g_h
        self.grad_and_hess = None

        # for data protection
        self.split_maskdict = {}
        self.missing_dir_maskdict = {}

    def get_feature_importance(self):
        print(self.feature_importance)
        return self.feature_importance
    def set_valid_features(self, valid_features=None):
        self.valid_features = valid_features
    def get_valid_features(self):
        return self.valid_features

    @staticmethod
    def get_grad_hess_sum(grad_and_hess_table):
        grad, hess = grad_and_hess_table.reduce(
            lambda value1, value2: (value1[0] + value2[0], value1[1] + value2[1]))
        return grad, hess

    def set_flowid(self, flowid=0):
        self.transfer_inst.set_flowid(flowid)
    
    def set_runtime_idx(self, runtime_idx):
        self.runtime_idx = runtime_idx
        self.sitename = ":".join([self.sitename, str(self.runtime_idx)])

    def set_valid_features(self, valid_features=None):
        self.valid_features = valid_features

    def set_grad_and_hess(self, grad_and_hess):
        self.grad_and_hess = grad_and_hess

    def set_input_data(self, data_bin, bin_split_points, bin_sparse_points):

        self.data_bin = data_bin
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points

    def get_local_histograms(self, node_map, ret='tensor'):
        acc_histograms = FeatureHistogram.calculate_histogram(
            self.data_with_node_assignments, self.grad_and_hess,
            self.bin_split_points, self.bin_sparse_points,
            self.valid_features, node_map)
        return acc_histograms

    @staticmethod
    def get_node_map(nodes: List[Node], left_node_only=False):
        node_map = {}
        idx = 0
        for node in nodes:
            if node.id != 0 and (not node.is_left_node and left_node_only):
                continue
            node_map[node.id] = idx
            idx += 1
        return node_map

    @ staticmethod
    def assign_instance_to_root_node(data_bin, root_node_id):
        return data_bin.mapValues(lambda inst: (1, root_node_id))

    def update_feature_importance(self, splitinfo, record_site_name=False):

        if self.feature_importance_type == "split":
            inc = 1
        elif self.feature_importance_type == "gain":
            inc = splitinfo.gain
        else:
            raise ValueError("feature importance type {} not support yet".format(self.feature_importance_type))

        fid = splitinfo.best_fid

        key = fid

        if key not in self.feature_importance:
            self.feature_importance[key] = 0

        self.feature_importance[key] += inc

    """
    To implement
    """

    # @abc.abstractmethod
    # def fit(self):
    #     pass
    #
    # @abc.abstractmethod
    # def predict(self, data_inst):
    #     pass
    #
    # @abc.abstractmethod
    # def initialize_root_node(self, *args):
    #     pass
    #
    # @abc.abstractmethod
    # def compute_best_splits(self, *args):
    #     pass
    #
    # @abc.abstractmethod
    # def update_instances_node_positions(self, *args):
    #     pass
    #
    # @abc.abstractmethod
    # def assign_a_instance(self, *args):
    #     pass
    #
    # @abc.abstractmethod
    # def assign_instances_to_new_node(self, *args):
    #     pass
    #
    # @abc.abstractmethod
    # def update_tree(self, *args):
    #     pass
    #
    # @abc.abstractmethod
    # def convert_bin_to_real(self, *args):
    #     pass
    #
    # @abc.abstractmethod
    # def get_model_meta(self):
    #     raise NotImplementedError("method should overload")
    #
    # @abc.abstractmethod
    # def get_model_param(self):
    #     raise NotImplementedError("method should overload")
    #
    # @abc.abstractmethod
    # def set_model_param(self, model_param):
    #     pass
    #
    # @abc.abstractmethod
    # def set_model_meta(self, model_meta):
    #     pass
    #
    # @abc.abstractmethod
    # def traverse_tree(self, *args):
    #     pass

    # def get_model(self):
    #
    #     model_meta = self.get_model_meta()
    #     model_param = self.get_model_param()
    #     return model_meta, model_param
    #
    # def load_model(self, model_meta=None, model_param=None):
    #     self.set_model_meta(model_meta)
    #     self.set_model_param(model_param)

    """
    For debug
    """

    def print_leafs(self):
        pass

    @staticmethod
    def print_split(split_infos: [SplitInfo]):
        pass

    @staticmethod
    def print_hist(hist_list: [HistogramBag]):
        pass
