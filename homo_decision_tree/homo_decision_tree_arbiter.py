from typing import List
from tree_core.decision_tree import DecisionTree
from tree_core.splitter import Splitter
from tree_core.splitter import SplitInfo


class HomoDecisionTreeArbiter(DecisionTree):

    def __init__(self):

        super(HomoDecisionTreeArbiter, self).__init__()
        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node,)

        """
        initializing here
        """
        self.valid_features = [True] * 87

        self.tree_node = []  # start from root node
        self.tree_node_num = 0
        self.cur_layer_node = []

        self.runtime_idx = 0

        # stored histogram for faster computation {node_id:histogram_bag}
        self.stored_histograms = {}
        self.federated_ps_receive = {}
        self.federated_ps_send = {}

    def set_flowid(self, flowid=0):
        self.transfer_inst.set_flowid(flowid)

    """
    Federation Functions
    """

    # def sync_node_sample_numbers(self, suffix):
    #     cur_layer_node_num = self.transfer_inst.cur_layer_node_num.get(-1, suffix=suffix)
    #     for num in cur_layer_node_num[1:]:
    #         assert num == cur_layer_node_num[0]
    #     return cur_layer_node_num[0]
    #
    # def sync_best_splits(self, split_info, suffix):
    #     self.transfer_inst.best_split_points.remote(split_info, idx=-1, suffix=suffix)
    #
    # def sync_local_histogram(self, suffix) -> List[HistogramBag]:
    #
    #     node_local_histogram = self.aggregator.aggregate_histogram(suffix=suffix)
    #     return node_local_histogram

    """
    Split finding
    """

    def federated_find_best_split(self, node_histograms, parallel_partitions=10) -> List[SplitInfo]:
        best_splits = self.splitter.find_split(node_histograms, self.valid_features)
        return best_splits

    @staticmethod
    def histogram_subtraction(left_node_histogram, stored_histograms):
        # histogram subtraction
        all_histograms = []
        for left_hist in left_node_histogram:
            all_histograms.append(left_hist)
            # LOGGER.debug('hist id is {}, pid is {}'.format(left_hist.hid, left_hist.p_hid))
            # root node hist
            if left_hist.hid == 0:
                continue
            right_hist = stored_histograms[left_hist.p_hid] - left_hist
            right_hist.hid, right_hist.p_hid = left_hist.hid + 1, right_hist.p_hid
            all_histograms.append(right_hist)

        return all_histograms

    """
    Fit
    """

    def fit(self):
        pass

    def predict(self, data_inst=None):
        """
        Do nothing
        """
        pass

    """
    These functions are not needed in homo_decision_tree-decision-tree
    """

    def initialize_root_node(self, *args):
        pass

    def compute_best_splits(self, *args):
        pass

    def assign_a_instance(self, *args):
        pass

    def assign_instances_to_new_node(self, *args):
        pass

    def update_tree(self, *args):
        pass

    def convert_bin_to_real(self, *args):
        pass

    def get_model_meta(self):
        pass

    def get_model_param(self):
        pass

    def set_model_param(self, model_param):
        pass

    def set_model_meta(self, model_meta):
        pass

    def traverse_tree(self, *args):
        pass

    def update_instances_node_positions(self, *args):
        pass
