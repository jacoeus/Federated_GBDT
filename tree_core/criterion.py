#!/usr/bin/env python    
# -*- coding: utf-8 -*-

# =============================================================================
# Criterion
# =============================================================================


class Criterion(object):
    def __init__(self, criterion_params):
        pass

    @staticmethod
    def split_gain(node_sum, left_node_sum, right_node_sum):
        raise NotImplementedError("node gain calculation method should be define!!!")


class XgboostCriterion(Criterion):
    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def split_gain(self, node_sum, left_node_sum, right_node_sum):
        sum_grad, sum_hess = node_sum
        left_node_sum_grad, left_node_sum_hess = left_node_sum
        right_node_sum_grad, right_node_sum_hess = right_node_sum
        return self.node_gain(left_node_sum_grad, left_node_sum_hess) + \
               self.node_gain(right_node_sum_grad, right_node_sum_hess) - \
               self.node_gain(sum_grad, sum_hess)

    def node_gain(self, sum_grad, sum_hess):
        return sum_grad * sum_grad / (sum_hess + self.reg_lambda)

    def node_weight(self, sum_grad, sum_hess):
        return -sum_grad / (self.reg_lambda + sum_hess)
