import functools

from tree_core.criterion import XgboostCriterion
from tree_core.feature_histogram import FeatureHistogram
from tree_core.decision_tree import DecisionTree
from tree_core.node import Node
from tree_core.feature_histogram import HistogramBag
from tree_core.splitter import SplitInfo
from typing import List


class HomoDecisionTreeClient(DecisionTree):

    def __init__(self, data_bin, bin_split_points, g_h, feature_num):

        super(HomoDecisionTreeClient, self).__init__()
        self.dep = None
        # self.tree_idx = None
        self.splitter = XgboostCriterion()
        self.data_bin = data_bin
        self.g_h = g_h
        self.bin_split_points = bin_split_points
        self.feature_num = feature_num
        # self.epoch_idx = epoch_idx
        self.split_info = []
        self.agg_histograms = []
        # self.valid_features = valid_feature
        self.tree_node = []  # start from root node
        self.tree_node_num = 0
        self.cur_layer_node = []
        #self.feature_importance = {}
        self.book = {i: i for i in range(len(self.data_bin))}
        self.sample_weights = [0] * len(self.data_bin)
        self.inst2node_idx = [(1, 0)] * len(self.data_bin)
        self.table_with_assignment = (self.data_bin, self.inst2node_idx)

    """
    Computing functions
    """

    def get_node_map(self, nodes: List[Node], left_node_only=True):
        node_map = {}
        idx = 0
        for node in nodes:
            if node.id != 0 and (not node.is_left_node and left_node_only):
                continue
            node_map[node.id] = idx
            idx += 1
        return node_map

    def get_grad_hess_sum(self, grad_and_hess_table):
        grad, hess = 0, 0
        for gh in grad_and_hess_table:
            grad += gh[0]
            hess += gh[1]
        return grad, hess

    def get_local_histogram(self, cur_to_split: List[Node], g_h, table_with_assign,
                            split_points, sparse_point, valid_feature):

        node_map = self.get_node_map(nodes=cur_to_split)
        histograms = FeatureHistogram.calculate_histogram(
            table_with_assign, g_h,
            split_points, self.feature_num,
            valid_feature, node_map,)

        hist_bags = []
        for hist_list in histograms:
            hist_bags.append(HistogramBag(hist_list))

        return hist_bags

    def get_left_node_local_histogram(self, cur_nodes: List[Node], tree: List[Node], g_h, table_with_assign,
                                      split_points, feature_num, valid_feature):
        node_map = self.get_node_map(cur_nodes, left_node_only=True)

        histograms = FeatureHistogram.calculate_histogram(
            table_with_assign, g_h, split_points,
            feature_num, valid_feature, node_map)

        hist_bags = []
        for hist_list in histograms:
            hist_bags.append(HistogramBag(hist_list))

        left_nodes = []
        for node in cur_nodes:
            if node.is_left_node or node.id == 0:
                left_nodes.append(node)

        # set histogram id and parent histogram id
        for node, hist_bag in zip(left_nodes, hist_bags):
            hist_bag.hid = node.id
            hist_bag.p_hid = node.parent_nodeid
        return hist_bags

    """
    Tree Updating
    """

    def update_tree(self, cur_to_split: List[Node], split_info: List[SplitInfo]):
        """
        update current tree structure
        ----------
        split_info
        """
        
        next_layer_node = []
        assert len(cur_to_split) == len(split_info)

        for idx in range(len(cur_to_split)):
            sum_grad = cur_to_split[idx].sum_grad
            sum_hess = cur_to_split[idx].sum_hess
            if split_info[idx].best_fid is None or split_info[idx].gain <= self.min_impurity_split + 1e-8:
                cur_to_split[idx].is_leaf = True
                self.tree_node.append(cur_to_split[idx])
                continue
            '''
            if idx != len(cur_to_split) - 1 and cur_to_split[idx].sibling_nodeid == cur_to_split[idx+1].id and split_info[idx].gain < split_info[idx+1].gain:
                cur_to_split[idx].is_leaf = True
                self.tree_node.append(cur_to_split[idx])
                continue

            if idx != 0 and cur_to_split[idx].sibling_nodeid == cur_to_split[idx-1].id and split_info[idx].gain < split_info[idx-1].gain:
                cur_to_split[idx].is_leaf = True
                self.tree_node.append(cur_to_split[idx])
                continue
            '''
            
            cur_to_split[idx].fid = split_info[idx].best_fid
            cur_to_split[idx].bid = split_info[idx].best_bid

            p_id = cur_to_split[idx].id
            l_id, r_id = self.tree_node_num + 1, self.tree_node_num + 2
            cur_to_split[idx].left_nodeid, cur_to_split[idx].right_nodeid = l_id, r_id
            self.tree_node_num += 2

            l_g, l_h = split_info[idx].sum_grad, split_info[idx].sum_hess
            # create new left node and new right node
            left_node = Node(id=l_id,
                             sum_grad=l_g,
                             sum_hess=l_h,
                             weight=self.splitter.node_weight(l_g, l_h),
                             parent_nodeid=p_id,
                             sibling_nodeid=r_id,
                             is_left_node=True)
            right_node = Node(id=r_id,
                              sum_grad=sum_grad - l_g,
                              sum_hess=sum_hess - l_h,
                              weight=self.splitter.node_weight(sum_grad - l_g, sum_hess - l_h),
                              parent_nodeid=p_id,
                              sibling_nodeid=l_id,
                              is_left_node=False)

            next_layer_node.append(left_node)
            next_layer_node.append(right_node)
            self.tree_node.append(cur_to_split[idx])
            self.update_feature_importance(split_info[idx])
            # self.update_feature_importance(split_info[idx], record_site_name=False)

        return next_layer_node

    @staticmethod
    def assign_a_instance(row, tree: List[Node], bin_split_points):
        leaf_status, nodeid = row[1]
        node = tree[nodeid]
        if node.is_leaf:
            return node.weight

        fid = node.fid
        bid = node.bid
        record_bid = 0
        for bi in range(len(bin_split_points[fid])):
            if row[0][fid] <= bin_split_points[fid][bi]:
                record_bid = bi
                break
        if record_bid <= bid:
            return 1, tree[nodeid].left_nodeid
        else:
            return 1, tree[nodeid].right_nodeid

    def assign_instances_to_new_node(self, table_with_assignment, tree_node: List[Node]):
        assign_method = functools.partial(self.assign_a_instance, tree=tree_node,
                                          bin_split_points=self.bin_split_points)
        assign_result = []
        data_bin = []
        g_h = []
        result_index = 0
        for i in range(len(table_with_assignment[0])):
            result = assign_method((table_with_assignment[0][i], table_with_assignment[1][i]))
            if isinstance(result, tuple):
                assign_result.append(result)
                data_bin.append(table_with_assignment[0][i])
                g_h.append(self.g_h[i])
                self.book[result_index] = self.book[i]
                result_index += 1
            else:
                assert self.sample_weights[self.book[i]] == 0
                self.sample_weights[self.book[i]] = result
        # leaf_val = assign_result.filter(lambda key, value: isinstance(value, tuple) is False)
        # assign_result = assign_result.subtractByKey(leaf_val)
        assert result_index == len(assign_result)
        return (data_bin, assign_result), g_h

    """
    Pre/Post process
    """

    #def get_feature_importance(self):
        #return self.feature_importance

    def convert_bin_to_real(self):
        """
        convert current bid in tree nodes to real value
        """
        for node in self.tree_node:
            if not node.is_leaf:
                node.bid = self.bin_split_points[node.fid][node.bid]

    # def assign_instance_to_root_node(self, data_bin, root_node_id):
    #     return data_bin.mapValues(lambda inst: (1, root_node_id))

    """
    Fit & Predict
    """
    def fit_send_g_h(self):  # for class_idx in range(self.booster_dim)
        return self.get_grad_hess_sum(self.g_h)

    def fit_get_global_g_h(self, global_g_sum, global_h_sum):
        root_node = Node(id=0, sum_grad=global_g_sum, sum_hess=global_h_sum, weight=self.splitter.node_weight(
            global_g_sum, global_h_sum))
        self.cur_layer_node = [root_node]

    def fit_break(self):
        for node in self.cur_layer_node:
            node.is_leaf = True
            self.tree_node.append(node)
        for i in range(len(self.table_with_assignment[0])):
            assert self.sample_weights[self.book[i]] == 0
            self.sample_weights[self.book[i]] = self.tree_node[self.table_with_assignment[1][i][1]].weight
        assert len(self.sample_weights) == len(self.data_bin)
        # for i in range(len(self.sample_weights)):
        #     assert self.sample_weights[i] != 0
        # return self.tree_node

    def fit_cur_layer_node_num(self):
        # self.split_info, self.agg_histograms = [], []
        self.split_info = []
        return len(self.cur_layer_node)

    def fit_send_local_h(self, dep):
        cur_to_split = self.cur_layer_node
        node_map = self.get_node_map(nodes=cur_to_split)
        # print('node map is {}'.format(node_map))
        local_histogram = self.get_left_node_local_histogram(
            cur_nodes=cur_to_split,
            tree=self.tree_node,
            g_h=self.g_h,
            table_with_assign=self.table_with_assignment,
            split_points=self.bin_split_points,
            feature_num=self.feature_num,
            valid_feature=self.valid_features
        )
        # self.agg_histograms += local_histogram
        return local_histogram

    def fit_get_split_info(self, dep, split_info):
        new_layer_node = self.update_tree(self.cur_layer_node, split_info)
        self.cur_layer_node = new_layer_node
        self.table_with_assignment, self.g_h = self.assign_instances_to_new_node(self.table_with_assignment,
                                                                                 self.tree_node)

    def fit_convert(self):
        self.convert_bin_to_real()

    def get_tree_node(self):
        nodes = []
        for node in self.tree_node:
            nodes.append((node.weight, node.sibling_nodeid, node.parent_nodeid, node.is_leaf))
        return nodes

    def get_sample_weights(self):
        return self.sample_weights
