#!/usr/bin/env python    
# -*- coding: utf-8 -*-
import warnings

from tree_core.criterion import XgboostCriterion


class SplitInfo(object):
    def __init__(self, best_fid=None, best_bid=None,
                 sum_grad=0, sum_hess=0, gain=None, missing_dir=1, left_sample_count=0):
        self.best_fid = best_fid
        self.best_bid = best_bid
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.gain = gain
        self.missing_dir = missing_dir

    def __str__(self):
        return '**fid {}, bid {}, sum_grad{}, sum_hess{}, gain{}**'.format(self.best_fid, self.best_bid,
                                                                           self.sum_grad, self.sum_hess, self.gain)


class Splitter(object):

    def __init__(self, criterion_method, criterion_params=[0, 1], min_impurity_split=1e-2, min_sample_split=2,
                 min_leaf_node=1):
        if not isinstance(criterion_method, str):
            raise TypeError("criterion_method type should be str, but %s find" % type(criterion_method).__name__)

        if criterion_method == "xgboost":
            if not criterion_params:
                self.criterion = XgboostCriterion()
            else:
                try:
                    reg_lambda = float(criterion_params[0])
                    self.criterion = XgboostCriterion(reg_lambda)
                except:
                    warnings.warn("criterion_params' first criterion_params should be numeric")
                    self.criterion = XgboostCriterion()

        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node

    def node_weight(self, grad, hess):
        return self.criterion.node_weight(grad, hess)

    def find_split_single_histogram_guest(self, histogram, valid_features):
        # default values
        best_fid = None
        best_gain = self.min_impurity_split - 1e-8
        best_bid = None
        best_sum_grad_l = None
        best_sum_hess_l = None
        for fid in range(len(histogram)):

            assert valid_features[fid] is True
            bin_num = len(histogram[fid])

            # last bin contains sum values (cumsum from left)
            sum_grad = histogram[fid][bin_num - 1][0]
            sum_hess = histogram[fid][bin_num - 1][1]
            node_cnt = histogram[fid][bin_num - 1][2]
            # print(node_cnt)

            if node_cnt < self.min_sample_split:
                break

            # last bin will not participate in split find, so bin_num - 1
            for bid in range(bin_num):
                # left gh
                sum_grad_l = histogram[fid][bid][0]
                sum_hess_l = histogram[fid][bid][1]
                node_cnt_l = histogram[fid][bid][2]
                # right gh
                sum_grad_r = sum_grad - sum_grad_l
                sum_hess_r = sum_hess - sum_hess_l
                node_cnt_r = node_cnt - node_cnt_l

                if node_cnt_l >= self.min_leaf_node and node_cnt_r >= self.min_leaf_node:
                    gain = self.criterion.split_gain([sum_grad, sum_hess],
                                                     [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r])

                    if gain > self.min_impurity_split and gain > best_gain + 1e-8:
                        best_gain = gain
                        best_fid = fid
                        best_bid = bid
                        best_sum_grad_l = sum_grad_l
                        best_sum_hess_l = sum_hess_l
                    # else:
                    #     print('ERROR:splitter.py find_split_single_histogram_guest')

        print((best_fid, best_bid, best_gain, best_sum_grad_l, best_sum_hess_l))
        splitinfo = SplitInfo(best_fid=best_fid, best_bid=best_bid,
                              gain=best_gain, sum_grad=best_sum_grad_l, sum_hess=best_sum_hess_l)

        return splitinfo

    def find_split(self, histograms, valid_features):

        splitinfo_table = [self.find_split_single_histogram_guest(histogram, valid_features)
                           for histogram in histograms]

        return splitinfo_table
