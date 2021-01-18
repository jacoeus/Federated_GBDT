from tree_core.node import Node
from tree_core.quantile_summaries import quantile_summary_factory
from homo_decision_tree.homo_decision_tree_client import HomoDecisionTreeClient
import numpy as np
from typing import List
from preprocess import CompDataset
from preprocess import get_user_data
from collections import Counter
import pickle
import random


class Worker(object):
    SAVE_DIR = '/tmp/model/'

    def __init__(self, booster_dim, bin_num, feature_num, u):
        self.booster_dim = booster_dim
        self.bin_num = bin_num
        self.feature_num = feature_num
        x, y = get_user_data(user_idx=u)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        self.data_bin = x
        self.y = np.array(y)
        self.u = u

    def build_projects(self):
        projects = []
        for i in range(20):
            projects.append(Project(self.booster_dim, self.bin_num,
                                    self.feature_num, self.data_bin[i * 1000: (i + 1) * 1000, :],
                                    self.y[i * 1000: (i + 1) * 1000], self.u))
            projects.append(Project(self.booster_dim, self.bin_num, self.feature_num,
                                    self.data_bin[4950-(i + 1) * 1000: 4950 - i * 1000, :],
                                    self.y[4950 - (i + 1) * 1000: 4950 - i * 1000], self.u))
        return projects


class Project(object):
    SAVE_DIR = '/tmp/model/'

    def __init__(self, booster_dim, bin_num, feature_num, x, y, u):
        self.booster_dim = booster_dim
        self.bin_num = bin_num
        self.feature_num = feature_num
        # self.data_bin = data_bin[0]
        # self.y = np.array(data_bin[1])
        self.tree_list = []
        self.estimators = None
        self.bin_split_points = []
        self.epoch_idx = 0
        self.valid_features = []
        self.feature_importance = {}
        self.loss_method = None
        self.y_hat = None
        self.init_score = None
        self.all_g_h = None
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        # self.ori_data_bin = x
        # self.ori_y = np.array(y)
        self.data_bin = x
        self.y = np.array(y)
        self.u = u

    """
    Boosting Fit
    """

    def set_datebin_feature(self, feature_list):
        self.data_bin = self.data_bin[:, feature_list]

    def count_label(self):
        return self.u, Counter(self.y)

    def receive_quantile_info(self):
        summary_list = []
        for i in range(self.feature_num):
            summary_list.append(quantile_summary_factory(False))
        for rid in range(len(self.data_bin)):
            for fid in range(len(self.data_bin[rid])):
                summary_list[fid].insert(self.data_bin[rid][fid])
        for sum_obj in summary_list:
            sum_obj.compress()
        return summary_list

    def fit_init(self, bin_split_points):
        self.bin_split_points = bin_split_points
        from tree_core.cross_entropy import SoftmaxCrossEntropyLoss
        self.loss_method = SoftmaxCrossEntropyLoss()
        self.y_hat, self.init_score = self.loss_method.initialize(self.y, self.booster_dim)

    def fit_booster_init(self):
        self.all_g_h = [(self.loss_method.compute_grad(self.y[i], self.loss_method.predict(self.y_hat[i])),
                         self.loss_method.compute_hess(self.y[i], self.loss_method.predict(self.y_hat[i]))) for i
                        in range(self.y.size)]
        self.estimators = None

    """
    Decision Fit
    """

    def fit_tree_init(self, class_idx):
        g_h = [(self.all_g_h[i][0][class_idx], self.all_g_h[i][1][class_idx]) for i in range(self.y.size)]
        self.estimators = HomoDecisionTreeClient(self.data_bin, self.bin_split_points, g_h, self.feature_num)
        # assert len(self.estimators) == class_idx + 1

    def receive_g_h_info(self, class_idx):  # for class_idx in range(self.booster_dim)
        return self.estimators.fit_send_g_h()

    def fit_distribute_global_g_h(self, class_idx, global_g_sum, global_h_sum):
        self.estimators.fit_get_global_g_h(global_g_sum, global_h_sum)

    def fit_tree_stop(self, class_idx):
        self.estimators.fit_break()

    def receive_cur_layer_node_num_info(self, class_idx):
        return self.estimators.fit_cur_layer_node_num()

    def receive_local_h_info(self, class_idx, dep):
        return self.estimators.fit_send_local_h(dep)

    def fit_distribute_split_info(self, class_idx, dep, split_info):
        self.estimators.fit_get_split_info(dep, split_info)

    def fit_convert(self, class_idx):
        self.estimators.fit_convert()

    def fit_update_y_hat(self, class_idx, lr, epoch_idx):
        cur_sample_weight = self.estimators.get_sample_weights()
        for index, hat in enumerate(self.y_hat):
            hat[class_idx] += cur_sample_weight[index] * lr
        # self.tree_list.append(self.estimators)

        save_path = self.SAVE_DIR + "{}-{}.pkl".format(epoch_idx, class_idx, )
        file = open(save_path, 'wb')
        pickle.dump(self.estimators, file)
        file.close()

    def traverse_tree(self, data_inst, tree: List[Node]):
        nid = 0  # root node id
        while True:
            if tree[nid].is_leaf:
                return tree[nid].weight

            cur_node = tree[nid]
            fid, bid = cur_node.fid, cur_node.bid
            if data_inst[fid] <= bid + 1e-8:
                nid = tree[nid].left_nodeid
            else:
                nid = tree[nid].right_nodeid

    def predict(self, data_inst, lr, boosting_round):
        predicts = []
        tree_list = []
        for boost_idx in range(boosting_round):
            for class_idx in range(self.booster_dim):
                model_path = self.SAVE_DIR + "{}-{}.pkl".format(boost_idx, class_idx, )
                load_file = open(model_path, 'rb')
                tree = pickle.load(load_file)
                load_file.close()
                tree_list.append(tree)

        for record in data_inst:
            weight_list = [0] * self.booster_dim
            for index, tree in enumerate(tree_list):
                weight_list[index % self.booster_dim] += self.traverse_tree(record, tree.tree_node, ) * lr

            weights = np.array(weight_list).reshape((-1, self.booster_dim))
            predict = self.loss_method.predict(weights).tolist()
            predicts.append(np.argmax(predict))

        return predicts

    def get_init_score(self):
        return self.init_score

    def get_valid_features(self):
        return self.valid_features

    def set_valid_features(self, valid_features: list):
        for i in valid_features:
            if i not in self.valid_features:
                self.valid_features.append(i)

    def choose_valid_feature_data(self):
        assert len(self.bin_split_points) == len(self.data_bin[0])
        self.data_bin = self.data_bin[:, self.valid_features]
        self.bin_split_points = [self.bin_split_points[i] for i in self.valid_features]
        self.feature_num = len(self.valid_features)
        print("Now the data and split point has {} features".format(len(self.valid_features)))
        self.valid_features = []

    def update_feature_importance(self):
        tree_feature_importance = self.estimators.get_feature_importance()
        new_valid_features = []

        for fid in tree_feature_importance:
            if fid not in self.feature_importance:
                self.feature_importance[fid] = 0
            self.feature_importance[fid] += tree_feature_importance[fid]
            new_valid_features.append(fid)

        self.set_valid_features(new_valid_features)
