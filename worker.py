from tree_core.node import Node
from tree_core.quantile_summaries import quantile_summary_factory
from homo_decision_tree.homo_decision_tree_client import HomoDecisionTreeClient
import numpy as np
from typing import List
from preprocess import CompDataset
from preprocess import get_user_data
from collections import Counter
import pickle
import torch
import torch.nn.functional as F
from torch import Tensor


class Worker(object):
    SAVE_DIR = '/tmp/model/'

    def __init__(self, booster_dim, bin_num, feature_num, u):
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
        x, y = get_user_data(user_idx=u)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        self.data_bin = x
        self.y = np.array(y)
        self.u = u

    """
    Boosting Fit
    """

    def count_label(self):
        return self.u, Counter(self.y)

    def fit_get_quantile(self):
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

    def fit_aggregate_g_h(self, class_idx):  # for class_idx in range(self.booster_dim)
        return self.estimators.fit_send_g_h()

    def fit_distribute_global_g_h(self, class_idx, global_g_sum, global_h_sum):
        self.estimators.fit_get_global_g_h(global_g_sum, global_h_sum)

    def fit_tree_stop(self, class_idx):
        self.estimators.fit_break()

    def fit_cur_layer_node_num(self, class_idx):
        return self.estimators.fit_cur_layer_node_num()

    def fit_aggregate_local_h(self, class_idx, dep):
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
        pickle.dump(self.estimators, file)  # 把 this worker model永久保存到文件中
        file.close()  # 关闭文件

    # def fit_send_tree_list(self):
    #     return self.tree_node

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


class NNWorker(object):
    def __init__(self, user_idx):
        self.user_idx = user_idx
        self.data = get_user_data(self.user_idx)  # The worker can only access its own data
        self.ps_info = {}

    def preprocess_data(self):
        x, y = self.data
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        self.data = (x, y)

    def round_data(self, n_round, n_round_samples=-1):
        """Generate data for user of user_idx at round n_round.

        Args:
            n_round: int, round number
            n_round_samples: int, the number of samples this round
        """

        if n_round_samples == -1:
            return self.data

        n_samples = len(self.data[1])
        choices = np.random.choice(n_samples, min(n_samples, n_round_samples))

        return self.data[0][choices], self.data[1][choices]

    def receive_server_info(self, info):  # receive info from PS
        self.ps_info = info

    def process_mean_round_train_acc(self):  # process the "mean_round_train_acc" info from server
        mean_round_train_acc = self.ps_info["mean_round_train_acc"]
        # You can go on to do more processing if needed

    def user_round_train(self, model, device, n_round, batch_size, n_round_samples=-1, debug=False):

        X, Y = self.round_data(n_round, n_round_samples)
        data = CompDataset(X=X, Y=Y)
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
        )

        model.train()

        correct = 0
        prediction = []
        real = []
        total_loss = 0
        model = model.to(device)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # import ipdb
            # ipdb.set_trace()
            # print(data.shape, target.shape)
            data = torch.unsqueeze(data, dim=0)
            data = data.float()
            output = model(data)
            loss = F.nll_loss(output, target)
            total_loss += loss
            loss.backward()
            pred = output.argmax(
                dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            prediction.extend(pred.reshape(-1).tolist())
            real.extend(target.reshape(-1).tolist())

        grads = {'n_samples': data.shape[0], 'named_grads': {}}
        for name, param in model.named_parameters():
            # print('User {}'.format(self.user_idx))
            # print(type(param))
            # print(type(param.grad))
            if isinstance(param.grad, Tensor) == True:
                grads['named_grads'][name] = param.grad.detach().cpu().numpy()

        worker_info = {}
        worker_info["train_acc"] = correct / len(train_loader.dataset)

        if debug:
            print('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
                total_loss, 100. * correct / len(train_loader.dataset)))

        return grads, worker_info
