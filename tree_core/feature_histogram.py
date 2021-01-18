#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# FeatureHistogram
# =============================================================================
import copy
import functools

from operator import add, sub
from typing import List
from datetime import datetime


class NoneType(object):
    def __eq__(self, obj):
        return isinstance(obj, NoneType)


def bisect_left(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


class HistogramBag(object):
    """
    holds histograms
    """

    def __init__(self, tensor: list, hid: int = -1, p_hid: int = -1):

        """
        :param tensor: list returned by calculate_histogram
        :param hid: histogram id
        :param p_hid: parent node histogram id
        """

        self.hid = hid
        self.p_hid = p_hid
        self.bag = tensor

    def binary_op(self, other, func, inplace=False):

        assert isinstance(other, HistogramBag)
        assert len(self.bag) == len(other)

        bag = self.bag
        newbag = None
        if not inplace:
            newbag = copy.deepcopy(other)
            bag = newbag.bag

        for bag_idx in range(len(self.bag)):
            for hist_idx in range(len(self.bag[bag_idx])):
                bag[bag_idx][hist_idx][0] = func(self.bag[bag_idx][hist_idx][0], other[bag_idx][hist_idx][0])
                bag[bag_idx][hist_idx][1] = func(self.bag[bag_idx][hist_idx][1], other[bag_idx][hist_idx][1])
                bag[bag_idx][hist_idx][2] = func(self.bag[bag_idx][hist_idx][2], other[bag_idx][hist_idx][2])

        return self if inplace else newbag

    def from_hist_tensor(self):
        pass

    def __add__(self, other):
        return self.binary_op(other, add, inplace=False)

    def __sub__(self, other):
        return self.binary_op(other, sub, inplace=False)

    def __len__(self):
        return len(self.bag)

    def __getitem__(self, item):
        return self.bag[item]

    def __str__(self):
        return str(self.bag)


class FeatureHistogramWeights:

    def __init__(self, list_of_histogram_bags: List[HistogramBag]):

        self.hists = list_of_histogram_bags
        super(FeatureHistogramWeights, self).__init__(l=list_of_histogram_bags)

    def map_values(self, func, inplace):

        if inplace:
            hists = self.hists
        else:
            hists = copy.deepcopy(self.hists)

        for histbag in hists:
            bag = histbag.bag
            for component_idx in range(len(bag)):
                for hist_idx in range(len(bag[component_idx])):
                    bag[component_idx][hist_idx][0] = func(bag[component_idx][hist_idx][0])
                    bag[component_idx][hist_idx][1] = func(bag[component_idx][hist_idx][1])
                    bag[component_idx][hist_idx][2] = func(bag[component_idx][hist_idx][2])

        if inplace:
            return self
        else:
            return FeatureHistogramWeights(list_of_histogram_bags=hists)

    def binary_op(self, other: 'FeatureHistogramWeights', func, inplace: bool):

        new_weights = []
        hists, other_hists = self.hists, other.hists
        for h1, h2 in zip(hists, other_hists):
            rnt = h1.binary_op(h2, func, inplace=inplace)
            if not inplace:
                new_weights.append(rnt)

        if inplace:
            return self
        else:
            return FeatureHistogramWeights(new_weights)

    def axpy(self, a, y: 'FeatureHistogramWeights'):

        func = lambda x1, x2: x1 + a * x2
        self.binary_op(y, func, inplace=True)

        return self

    def __iter__(self):
        pass

    def __str__(self):
        return str([str(hist) for hist in self.hists])


class FeatureHistogram(object):

    def __init__(self):
        pass

    @staticmethod
    def tensor_histogram_cumsum(histograms):
        # histogram cumsum, from left to right
        for i in range(1, len(histograms)):
            for j in range(len(histograms[i])):
                histograms[i][j] += histograms[i - 1][j]
        return histograms

    @staticmethod
    def calculate_histogram(data_bin, grad_and_hess, bin_split_points, feature_num, valid_features=None,
                            node_map=None, ):
        batch_histogram_cal = functools.partial(
            FeatureHistogram.batch_calculate_histogram, bin_split_points=bin_split_points,
            feature_num=feature_num, valid_features=valid_features, node_map=node_map, )
        batch_histogram_intermediate_rs = (data_bin, grad_and_hess)
        if len(batch_histogram_intermediate_rs[0]) == 0:
            print('ERROR:feature_histogram.py calculate_histogram')
        agg_histogram = functools.partial(FeatureHistogram.aggregate_histogram, node_map=node_map,
                                          feature_num=feature_num)
        histograms_table = batch_histogram_cal(batch_histogram_intermediate_rs)
        histograms_table = agg_histogram(histograms_table)
        rs = FeatureHistogram.recombine_histograms(histograms_table, node_map, feature_num)
        return rs

    @staticmethod
    def aggregate_histogram(fid_histograms, node_map, feature_num):
        # add histograms with same key((node id, feature id)) together
        aggregated_res = {}
        for i in range(len(fid_histograms)):
            if fid_histograms[i][0] in aggregated_res:
                aggregated_res[fid_histograms[i][0]] += fid_histograms[i][1][1]
            else:
                aggregated_res[fid_histograms[i][0]] = fid_histograms[i][1][1]
        return aggregated_res

    @staticmethod
    def generate_histogram_template(node_map: dict, bin_split_points: list, feature_num: int, valid_features: dict):

        # for every feature, generate histograms containers (initialized val are 0s)
        node_num = len(node_map)
        node_histograms = []
        for k in range(node_num):
            feature_histogram_template = []
            for fid in range(feature_num):
                # if is not valid features, skip generating
                # if valid_features is not None and valid_features[fid] is False:
                #     feature_histogram_template.append([])
                #     continue
                # else:
                # 0, 0, 0 -> grad, hess, sample count
                feature_histogram_template.append([[0, 0, 0] for j in range(len(bin_split_points[fid]))])

            node_histograms.append(feature_histogram_template)
            # check feature num
            assert len(feature_histogram_template) == feature_num

        return node_histograms

    @staticmethod
    def generate_histogram_key_value_list(node_histograms, node_map, feature_num):
        # generate key_value hist list for DTable parallelization
        ret = []
        for nid in range(len(node_map)):
            for fid in range(feature_num):
                ret.append(((nid, fid), (fid, node_histograms[nid][fid])))
        return ret

    @staticmethod
    def batch_calculate_histogram(kv_iterator, bin_split_points=None, feature_num=None, valid_features=None,
                                  node_map=None, ):
        data_bins = []
        node_ids = []
        grad = []
        hess = []
        data_record = 0  # total instance number of this partition
        # go through iterator to collect g/h feature instances/ node positions
        for i in range(len(kv_iterator[0][0])):
            data_bin = kv_iterator[0][0][i]
            nodeid_state = kv_iterator[0][1][i]
            unleaf_state, nodeid = nodeid_state
            if unleaf_state == 0 or nodeid not in node_map:
                continue
            g = kv_iterator[1][i][0]  # encrypted text in host, plaintext in guest
            h = kv_iterator[1][i][1]
            data_bins.append(data_bin)  # features
            node_ids.append(nodeid)  # current node position
            grad.append(g)
            hess.append(h)

            data_record += 1

        node_histograms = FeatureHistogram.generate_histogram_template(node_map, bin_split_points, feature_num,
                                                                       valid_features)

        for rid in range(data_record):
            nid = node_map.get(node_ids[rid])
            for feature_index in range(len(bin_split_points)):
                value_id = bisect_left(bin_split_points[feature_index], data_bins[rid][feature_index])
                if value_id >= len(bin_split_points[feature_index]):
                    value_id = len(bin_split_points[feature_index]) - 1
                node_histograms[nid][feature_index][value_id][0] += grad[rid]
                node_histograms[nid][feature_index][value_id][1] += hess[rid]
                node_histograms[nid][feature_index][value_id][2] += 1
                # for value_id in range(len(bin_split_points[feature_index])):
                #     if value <= bin_split_points[feature_index][value_id]:
                #         node_histograms[nid][feature_index][value_id][0] += grad[rid]
                #         node_histograms[nid][feature_index][value_id][1] += hess[rid]
                #         node_histograms[nid][feature_index][value_id][2] += 1
                #         break

        ret = FeatureHistogram.generate_histogram_key_value_list(node_histograms, node_map, feature_num)
        return ret

    @staticmethod
    def recombine_histograms(histograms_dict: dict, node_map, feature_num):

        histograms = [[[] for j in range(feature_num)] for k in range(len(node_map))]
        for key, value in histograms_dict.items():
            nid, fid = key
            histograms[int(nid)][int(fid)] = FeatureHistogram.tensor_histogram_cumsum(value)
        return histograms
