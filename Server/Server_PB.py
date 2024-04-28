import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import copy
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from stellargraph.core.graph import StellarGraph
from Models.Model_Ego import MyModel, GCN
from Datasets.Datasets import stellargraph_to_torch_geo
from Utils.log import log, binary_conversion


class Server:
    def __init__(self, args, graph:StellarGraph, node_subjects):
        self.args = args
        self.graph_sg = graph
        self.num_classes = args.num_classes
        self.client_adj_matrix = None

        self.feature_shape = self.graph_sg.node_features()[0].shape
        self.feature_len = self.feature_shape[0]
        self.node_ids = self.graph_sg.nodes()
        self.node_num = len(self.node_ids)
        self.graph_pyg = stellargraph_to_torch_geo(graph, node_subjects)

        self.model = MyModel(args.sageMode, self.feature_len, args.h_feats, args.h_feats,
                             args.linear, self.num_classes, layer_size=2).to(args.device)
        # self.model = GCN(self.feature_len, self.num_classes).to(args.device)

        self.loss_function = nn.CrossEntropyLoss().to(args.device)

        self.test_mask = torch.zeros(self.node_num, dtype=torch.bool)
        self.test_mask[:] = True
        self.data_loader = NeighborLoader(self.graph_pyg, num_neighbors=[8] * 2, batch_size=args.batch_size,
                                          input_nodes=self.test_mask, replace=True, directed=True,
                                          num_workers=args.num_workers)

    def aggregation_avg(self, list_parameters, weight=None):
        w_avg = copy.deepcopy(list_parameters[0])
        if weight is not None:
            for k in w_avg.keys():
                for i in range(1, len(list_parameters)):
                    w_avg[k] += list_parameters[i][k]*(1+self.args.lamuda_aggregation*weight[i])
                w_avg[k] = torch.div(w_avg[k], len(list_parameters))
        else:
            for k in w_avg.keys():
                for i in range(1, len(list_parameters)):
                    w_avg[k] += list_parameters[i][k]
                w_avg[k] = torch.div(w_avg[k], len(list_parameters))
        self.model.load_state_dict(w_avg)
        print(binary_conversion(w_avg))
        return w_avg

    def evaluate(self):
        self.model.eval()
        pred_all = []
        y_all = []
        total_examples = 0.0
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.data_loader:
                batch = batch.to(self.args.device)
                b_size = batch.batch_size
                x = batch.x
                y = batch.y
                edge_index = batch.edge_index

                out = self.model(x, edge_index)[:b_size]
                # out = self.model(batch)[:b_size] #GCN
                pred = out.argmax(dim=-1)

                pred_all += pred.tolist()
                y_all += y[:b_size].tolist()

                loss = self.get_loss(out, y[:b_size])

                total_examples += b_size
                total_loss += float(loss) * b_size

        acc = accuracy_score(y_all, pred_all)
        micro_f1 = f1_score(y_all, pred_all, average='macro')
        label = [i for i in range(self.num_classes)]
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_all, pred_all, labels=label)
        p_class = [round(x, 4) for x in p_class]
        r_class = [round(x, 4) for x in r_class]
        f_class = [round(x, 4) for x in f_class]
        log('\nPrecision:{} \nRecall:{} \nF1-score:{}'.format(p_class, r_class, f_class))
        return micro_f1, acc, total_loss/total_examples

    def get_loss(self, out, y):
        x = F.softmax(out, dim=1)
        loss = self.loss_function(x, y)
        return loss

    def get_adj_clients(self, client_id):
        adj_clients = [i for i, x in enumerate(self.client_adj_matrix[client_id]) if x == 1]
        return adj_clients


class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if alpha:
                #self.alpha = t.ones(class_num, 1, requires_grad=True)
                self.alpha = torch.tensor(alpha, requires_grad=True)
                # print('alpha初始\n', self.alpha)
                # print('alpha shape\n', self.alpha.shape)
        #             else:
        #                 self.alpha = t.ones(class_num, 1*alpha).cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        # input.shape = (N, C)
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)  # 經過softmax 概率
        # ---------one hot start--------------#
        class_mask = inputs.data.new(N, C).fill_(0)  # 生成和input一样shape的tensor
        # print('依照input shape制作:class_mask\n', class_mask)
        class_mask = class_mask.requires_grad_()  # 需要更新， 所以加入梯度计算
        ids = targets.view(-1, 1)  # 取得目标的索引
        # print('取得targets的索引\n', ids)
        class_mask.data.scatter_(1, ids.data, 1.)  # 利用scatter将索引丢给mask
        # print('targets的one_hot形式\n', class_mask)  # one-hot target生成
        # ---------one hot end-------------------#
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        #         alpha = self.alpha[ids.data.view(-1, 1)]
        #         alpha = self.alpha[ids.view(-1)]
        alpha = self.alpha
        # print('alpha值\n', alpha)
        # print('alpha shape\n', alpha.shape)

        probs = (P * class_mask).sum(1).view(-1, 1)
        # print('留下targets的概率（1的部分），0的部分消除\n', probs)
        # 将softmax * one_hot 格式，0的部分被消除 留下1的概率， shape = (5, 1), 5就是每个target的概率

        log_p = probs.log()
        # print('取得对数\n', log_p)
        # 取得对数

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p  # 對應下面公式
        # print('每一个batch的loss\n', batch_loss)
        # batch_loss就是取每一个batch的loss值

        # 最终将每一个batch的loss加总后平均
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        # print('loss值为\n', loss)
        return loss