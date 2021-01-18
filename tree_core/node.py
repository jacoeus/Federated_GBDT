#!/usr/bin/env python    
# -*- coding: utf-8 -*-

# =============================================================================
# Decision Tree Node Struture
# =============================================================================

class Node(object):
    def __init__(self, id=None, fid=None,
                 bid=None, weight=0, is_leaf=False, sum_grad=None,
                 sum_hess=None, left_nodeid=-1, right_nodeid=-1,
                 missing_dir=1, sample_num=0, sibling_nodeid=None, parent_nodeid=None, is_left_node=False):

        self.id = id
        self.fid = fid
        self.bid = bid
        self.weight = weight
        self.is_leaf = is_leaf
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.left_nodeid = left_nodeid
        self.right_nodeid = right_nodeid
        self.missing_dir = missing_dir
        self.sibling_nodeid = sibling_nodeid
        self.parent_nodeid = parent_nodeid
        self.sample_num = sample_num
        self.is_left_node = is_left_node

    def __str__(self):
        return "id{}, fid:{},bid:{},weight:{},sum_grad:{},sum_hess:{},left_node:{},right_node:{} " \
               "is leaf {}".format(self.id,
                self.fid, self.bid, self.weight, self.sum_grad, self.sum_hess, self.left_nodeid, self.right_nodeid,
                self.is_leaf
        )

