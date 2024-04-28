import numpy as np
import pandas as pd
import stellargraph as sg
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch import optim
from tqdm import tqdm
import copy
from stellargraph.core.graph import StellarGraph

from Models.Model_Sage import LocalSage_Plus
from Models.Model_Ego import MyModel, GCN
from Datasets.Datasets import stellargraph_to_torch_geo
from sklearn.metrics import f1_score, accuracy_score
from Utils.log import log

class Client:
    def __init__(self, args, graph: StellarGraph, node_subjects, node_target, all_subj):
        super(Client, self).__init__()
        self.device = args.device
        self.args = args
        self.all_subj = all_subj
        self.num_classes = args.num_classes

        # whole graph
        self.node_subjects = node_subjects
        self.node_target = node_target
        self.graph_sg = graph
        self.graph_pyg = stellargraph_to_torch_geo(self.graph_sg, self.node_subjects)
        self.feat_shape = self.graph_sg.node_features()[0].shape
        self.feat_len = self.feat_shape[0]
        self.all_ids = self.graph_sg.nodes()
        self.node_num = len(self.all_ids)

        # hided graph
        self.hided_ids = []
        self.hided_node_targets = []
        self.all_targets_feat = []
        self.all_targets_subj = []
        self.num_pred = self.args.num_pred_node
        self.hided_graph = self.hide_graph()

        self.hided_node_targets = np.asarray(self.hided_node_targets).reshape((-1, 1))
        self.hided_node_targets = torch.tensor(self.hided_node_targets).cuda()
        self.all_targets_feat = np.asarray(self.all_targets_feat).reshape((-1, self.num_pred, self.feat_len))
        self.all_targets_feat = torch.tensor(self.all_targets_feat).cuda()
        self.all_targets_subj = np.asarray(self.all_targets_subj).reshape((-1, self.num_classes))
        self.all_targets_subj = torch.tensor(self.all_targets_subj).cuda()
        self.hided_feature_shape = self.hided_graph.node_features()[0].shape
        self.hided_feature_len = self.hided_feature_shape[0]
        self.hided_node_ids = self.hided_graph.nodes()
        self.hided_node_num = len(self.hided_node_ids)

        # model
        self.model = MyModel(args.sageMode, self.feat_len, args.h_feats, args.h_feats,
                             args.linear, self.num_classes, layer_size=2).to(args.device)
        # self.model = GCN(self.feat_len, self.num_classes).to(args.device)

        self.neighborhood_gen_model = LocalSage_Plus(self.args, feat_len=self.hided_feature_len, node_len=self.hided_node_num,
                                                n_classes=self.num_classes).cuda()
        self.neighborhood_gen_model2 = None
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.neighborhood_gen_model_optimizer = optim.SGD(self.neighborhood_gen_model.parameters(), lr=self.args.lr)
        self.neighborhood_gen_model2_optimizer = None
        self.loss_function = nn.CrossEntropyLoss().to(args.device)

        self.train_mask = torch.zeros(self.node_num, dtype=torch.bool)
        self.train_mask[:] = True
        self.dataloader_train = NeighborLoader(self.graph_pyg, num_neighbors=[6]*2, batch_size=args.batch_size,
                                         input_nodes=self.train_mask, replace=True, directed=True,
                                         num_workers=args.num_workers)

    @staticmethod
    def greedy_loss(args, pred_feats, true_feats, pred_missing, true_missing):
        true_missing_tmp = true_missing.cpu()
        pred_missing_tmp = pred_missing.cpu()

        loss = torch.zeros(pred_feats.shape).cuda()

        pred_len = len(pred_feats)
        pred_missing_np = true_missing_tmp.detach().numpy().reshape(-1).astype(np.int32)
        true_missing_np = pred_missing_tmp.detach().numpy().reshape(-1).astype(np.int32)

        true_missing_np = np.clip(true_missing_np, 0, args.num_pred_node)
        pred_missing_np = np.clip(pred_missing_np, 0, args.num_pred_node)

        for i in range(pred_len):
            for pred_j in range(min(args.num_pred_node, pred_missing_np[i])):
                if true_missing_np[i] > 0:
                    if isinstance(true_feats[i][true_missing_np[i] - 1], np.ndarray):
                        true_feats_tensor = torch.tensor(true_feats[i][true_missing_np[i] - 1]).cuda()
                    else:
                        true_feats_tensor = true_feats[i][true_missing_np[i] - 1]
                    loss[i][pred_j] += F.mse_loss(pred_feats[i][pred_j].unsqueeze(0).float(),
                                                  true_feats_tensor.unsqueeze(0).float()).squeeze(0)

                    for true_k in range(min(args.num_pred_node, true_missing_np[i])):
                        if isinstance(true_feats[i][true_k], np.ndarray):
                            true_feats_tensor = torch.tensor(true_feats[i][true_k]).cuda()
                        else:
                            true_feats_tensor = true_feats[i][true_k]

                        loss_ijk = F.mse_loss(pred_feats[i][pred_j].unsqueeze(0).float(),
                                              true_feats_tensor.unsqueeze(0).float()).squeeze(0)
                        if torch.sum(loss_ijk) < torch.sum(loss[i][pred_j].data):
                            loss[i][pred_j] = loss_ijk
                else:
                    continue
        return loss

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def get_adj(self, edges, node_len):
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(node_len, node_len), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def get_degree(self, graph:StellarGraph):
        id2degree = {}
        edges = np.asarray(graph.edges(use_ilocs=True))
        tmp = [0 for i in range(len(edges)*2)]
        for i, edge in enumerate(edges):
            tmp[i] = edge[0]
            tmp[i+1] = edge[1]
        for node in tmp:
            if node in id2degree:
                id2degree[node] += 1
            else:
                id2degree[node] = 1
        return id2degree

    def hide_graph(self):
        id2degree = self.get_degree(self.graph_sg)
        for key in id2degree.keys():
            if id2degree[key] == 1:
                self.hided_ids.append(key)
        self.wn_hide_ids = list(set(self.graph_sg.nodes()).difference(self.hided_ids))
        hided_graph = self.graph_sg.subgraph(self.wn_hide_ids)
        li = hided_graph.nodes().tolist()
        loc_list = []
        for id in li:
            loc = self.graph_sg.node_ids_to_ilocs([id])
            missing_ids = list(set(self.graph_sg.neighbors(id)).difference(list(hided_graph.neighbors(id))))
            missing_len = len(missing_ids)
            self.hided_node_targets.append(missing_len)
            if missing_len > 0:
                if missing_len <= self.num_pred:
                    zeros = np.zeros((max(0, self.num_pred - missing_len), self.feat_len))
                    missing_feat_all = np.vstack((self.graph_sg.node_features(missing_ids), zeros)). \
                        reshape((1, self.num_pred, self.feat_len))
                else:
                    missing_feat_all = np.copy(self.graph_sg.node_features(missing_ids[:self.num_pred])). \
                        reshape((1, self.num_pred, self.feat_len))
            else:
                missing_feat_all = np.zeros((1, self.num_pred, self.feat_len))
            self.all_targets_feat.append(missing_feat_all)
            self.all_targets_subj.append(self.node_target[loc][0])
        if len(self.all_targets_subj[0]) == 1:
            tmp_all = np.empty(shape=(len(self.all_targets_subj), 2))
            for index, item in enumerate(self.all_targets_subj):
                tmp_all[index] = np.append(not item[0], item)
            self.all_targets_subj = tmp_all
        return hided_graph

    def train_neighborhood_gen(self):
        for epoch in tqdm(range(self.args.num_epoch_gen), desc='train neighgen'):
            self.neighborhood_gen_model.train()
            faeture = torch.tensor(self.hided_graph.node_features()).cuda()
            edges = np.asarray(self.hided_graph.edges(use_ilocs=True))
            edges = torch.tensor(edges.astype(np.int32))
            adj   = self.get_adj(edges, len(self.hided_graph.nodes())).cuda()
            # input is hide graph
            output_missing, output_feat, output_nc = self.neighborhood_gen_model(faeture, edges.cuda(), adj)

            output_missing = torch.flatten(output_missing)
            output_feat = output_feat.view(self.hided_node_num, self.num_pred, self.hided_feature_len)
            output_nc = output_nc.view(self.hided_node_num + self.hided_node_num * self.num_pred, self.num_classes)
            # loss
            loss_train_missing = F.smooth_l1_loss(output_missing.float(),
                                                  self.hided_node_targets.reshape(-1).float())

            loss_train_feat = self.greedy_loss(self.args, output_feat, self.all_targets_feat, output_missing,
                                               self.hided_node_targets).unsqueeze(0).mean().float()

            true_nc_label = torch.argmax(self.all_targets_subj, dim=1).view(-1).cuda()
            loca = self.graph_sg.node_ids_to_ilocs(self.hided_node_ids)
            loss_train_label = F.cross_entropy(output_nc[loca.tolist()], true_nc_label)

            loss = (loss_train_missing + loss_train_feat + loss_train_label).float()
            self.neighborhood_gen_model_optimizer.zero_grad()
            loss.backward()
            self.neighborhood_gen_model_optimizer.step()

    def local_train(self):
        self.model.train()
        total_examples = 0.0
        total_loss = 0.0

        for batch in self.dataloader_train:
            batch = batch.to(self.device)
            b_size = batch.batch_size
            x = batch.x
            y = batch.y
            edge_index = batch.edge_index

            out = self.model(x, edge_index)[:b_size]
            # out = self.model(batch)[:b_size]# GCN
            loss = self.loss_function(out, y[:b_size])
            # loss = F.nll_loss(out, y[:b_size])

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            total_examples += b_size
            total_loss += float(loss) * b_size

            acc = accuracy_score(y[:b_size].cpu().detach().numpy(), out.argmax(axis=1).cpu().detach().numpy())
            micro_f1 = f1_score(y[:b_size].cpu().detach().numpy(), out.argmax(axis=1).cpu().detach().numpy(), average='macro')
        return total_loss / total_examples, self.model.state_dict(), acc, micro_f1

    def fill_graph(self, args, missing, new_feats, feat_len):
        new_feats = new_feats.reshape((-1, args.num_pred_node, feat_len))
        original_node_ids = [id_i for id_i in self.graph_sg.nodes()]
        fill_node_ids = [id_i for id_i in self.graph_sg.nodes()]
        fill_node_feats = []
        org_feats = self.graph_sg.node_features()

        for i in range(len(list(self.graph_sg.nodes()))):
            fill_node_feats.append(np.asarray(org_feats[i].reshape(-1)))

        org_edges = np.copy(self.graph_sg.edges())
        fill_edges_source = [edge[0] for edge in org_edges]
        fill_edges_target = [edge[1] for edge in org_edges]

        start_id = -1
        for new_i in range(len(missing)):
            if int(missing[new_i]) > 0:
                new_ids_i = np.arange(start_id, start_id - min(args.num_pred_node, int(missing[new_i])), -1)

                i_pred = 0
                for i in new_ids_i:
                    original_node_ids.append(int(i))
                    if isinstance(new_feats[new_i][i_pred], np.ndarray) == False:
                        new_feats = new_feats.cpu()
                        new_feats = new_feats.detach().numpy()
                    fill_node_feats.append(np.asarray(new_feats[new_i][i_pred].reshape(-1)))
                    i_pred += 1
                    fill_edges_source.append(fill_node_ids[new_i])
                    fill_edges_target.append(int(i))

                start_id = start_id - min(args.num_pred_node, int(missing[new_i]))

        fill_edges_source = np.asarray(fill_edges_source).reshape((-1))
        fill_edges_target = np.asarray(fill_edges_target).reshape((-1))
        fill_edges = pd.DataFrame()
        fill_edges['source'] = fill_edges_source
        fill_edges['target'] = fill_edges_target
        fill_node_feats_np = np.asarray(fill_node_feats).reshape((-1, feat_len))
        fill_node_ids_np = np.asarray(original_node_ids).reshape(-1)

        fill_nodes = sg.IndexedArray(fill_node_feats_np, fill_node_ids_np)
        fill_G = sg.StellarGraph(nodes=fill_nodes, edges=fill_edges)
        return fill_nodes, fill_G

    def build_new_neighborhood_gen(self):
        self.neighborhood_gen_model2 = LocalSage_Plus(self.args, feat_len=self.feat_len, node_len=self.node_num,
                                                n_classes=self.num_classes).cuda()
        self.neighborhood_gen_model2.reg_model = copy.deepcopy(self.neighborhood_gen_model.reg_model)
        self.neighborhood_gen_model2.gen = copy.deepcopy(self.neighborhood_gen_model.gen)
        self.neighborhood_gen_model2_optimizer = optim.Adam(self.neighborhood_gen_model2.parameters(), lr=self.args.lr,
                                                           weight_decay=self.args.weight_decay)


class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if alpha:
                #                 self.alpha = t.ones(class_num, 1, requires_grad=True)
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








