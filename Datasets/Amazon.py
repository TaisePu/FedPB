import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import copy as cp
import json
import scipy.sparse
import os
import pickle
from collections import defaultdict

class Amazon(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 返回数据集源文件名
    @property
    def raw_file_names(self):
        return []

    # 返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']

    # 用于从网上下载数据集
    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    # 生成数据集所用的方法
    def process(self):
        # adj_full == adj_train
        adj_full, adj_train, feats, class_map, role = load_npz_data('amazon')

        row1 = []
        row2 = []
        resorted_edge_list = []
        for u, v in zip(adj_train.nonzero()[0],adj_train.nonzero()[1]):
            row1.append(u)
            row2.append(v)
        resorted_edge_list.append(row1)
        resorted_edge_list.append(row2)

        list_label = [class_map[index] for index in range(len(feats))]

        x = torch.tensor(feats, dtype=torch.float)
        y = torch.tensor(list_label, dtype=torch.long)
        edge_index = torch.tensor(resorted_edge_list, dtype=torch.long)

        '''split datasets train/val/test'''
        index = list(range(len(list_label)))
        labels = np.array(list_label)
        idx_train, idx_test, y_train, y_test = train_test_split(index, list_label, stratify=None, test_size=0.20,
                                                                random_state=2, shuffle=True)
        idx_val, idx_test, y_val, y_test = train_test_split(idx_test, y_test, stratify=None, test_size=0.50,
                                                            random_state=2, shuffle=True)

        train_mask = [False for _ in index]
        val_mask = [False for _ in index]
        test_mask = [False for _ in index]

        for i in range(len(idx_train) - 1):
            train_mask[idx_train[i]] = True
        for i in range(len(idx_val) - 1):
            val_mask[idx_val[i]] = True
        for i in range(len(idx_test) - 1):
            test_mask[idx_test[i]] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        data_list = [data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_npz_data(datasets_name):
    """
    Load the various data files residing in the `prefix` directory.
    Files to be loaded:
        adj_full.npz        sparse matrix in CSR format, stored as scipy.sparse.csr_matrix
                            The shape is N by N. Non-zeros in the matrix correspond to all
                            the edges in the full graph. It doesn't matter if the two nodes
                            connected by an edge are training, validation or test nodes.
                            For unweighted graph, the non-zeros are all 1.

        adj_train.npz       sparse matrix in CSR format, stored as a scipy.sparse.csr_matrix
                            The shape is also N by N. However, non-zeros in the matrix only
                            correspond to edges connecting two training nodes. The graph
                            sampler only picks nodes/edges from this adj_train, not adj_full.
                            Therefore, neither the attribute information nor the structural
                            information are revealed during training. Also, note that only
                            a x N rows and cols of adj_train contains non-zeros. For
                            unweighted graph, the non-zeros are all 1.

        role.json           a dict of three keys. Key 'tr' corresponds to the list of all
                              'tr':     list of all training node indices
                              'va':     list of all validation node indices
                              'te':     list of all test node indices
                            Note that in the raw data, nodes may have string-type ID. You
                            need to re-assign numerical ID (0 to N-1) to the nodes, so that
                            you can index into the matrices of adj, features and class labels.

        class_map.json      a dict of length N. Each key is a node index, and each value is
                            either a length C binary list (for multi-class classification)
                            or an integer scalar (0 to C-1, for single-class classification).

        feats.npz           a numpy array of shape N by F. Row i corresponds to the attribute
                            vector of node i.

    Inputs:
        prefix              string, directory containing the above graph related files.

        normalize           bool, whether or not to normalize the node features.

    Outputs:
        adj_full            scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
                            the full graph, with N being total num of train + val + test nodes.

        adj_train           scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
                            the training graph. While the shape is the same as adj_full, the
                            rows/cols corresponding to val/test nodes in adj_train are all-zero.

        feats               np array (shape N x f), the node feature matrix, with f being the
                            length of each node feature vector.

        class_map           dict, where key is the node ID and value is the classes this node
                            belongs to.

        role                dict, where keys are: 'tr' for train, 'va' for validation and 'te'
                            for test nodes. The value is the list of IDs of nodes belonging to
                            the train/val/test sets.
    """
    adj_full = scipy.sparse.load_npz('./Folder/{}/raw/adj_full.npz'.format(datasets_name)).astype(np.bool)
    adj_train = scipy.sparse.load_npz('./Folder/{}/raw/adj_train.npz'.format(datasets_name)).astype(np.bool)
    role = json.load(open('./Folder/{}/raw/role.json'.format(datasets_name)))
    feats = np.load('./Folder/{}/raw/feats.npy'.format(datasets_name))
    class_map = json.load(open('./Folder/{}/raw/class_map.json'.format(datasets_name)))
    class_map = {int(k): v for k, v in class_map.items()}
    assert len(class_map) == feats.shape[0]

    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role