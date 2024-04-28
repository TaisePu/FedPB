import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import copy as cp
import os
import pickle
from collections import defaultdict


class Elliptic(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return []
    
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    #用于从网上下载数据集
    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        labels_csv = pd.read_csv('./Folder/elliptic/raw/elliptic_txs_classes.csv',sep=',',header=0)
        edge_csv = pd.read_csv('./Folder/elliptic/raw/elliptic_txs_edgelist.csv',sep=',',header=0)
        feature_csv = pd.read_csv('./Folder/elliptic/raw/elliptic_txs_features.csv',sep=',',header=0)

        labels_arr = np.array(labels_csv)
        edge_arr = np.array(edge_csv)
        feature_arr = np.array(feature_csv)
        '''节点特征少一条，随机补一条'''
        feature_arr = np.append(feature_arr, [feature_arr[-1]], axis=0)

        labels_list = labels_arr.tolist()
        edge_list = edge_arr.tolist()
        feature_list = feature_arr.tolist()
        print('Node', len(labels_list), 'Edge', len(edge_list), 'Feature', len(feature_list))

        labels_list = []
        id2index = []
        '''按label来定index'''
        for item in labels_arr:
            if item[1] == '1':
                labels_list.append(1)
            elif item[1] == '2':
                labels_list.append(0)
            elif item[1] == 'unknown':
                labels_list.append(2)
            id2index.append(np.int64(item[0]))

        labels = np.array(labels_list)
        print('Class imbalance racio:{:.4f}'.format(np.mean(labels)))

        adj_lists = defaultdict(set)
        prefix = './Folder/elliptic/raw/'
        filename = 'elliptic_adjlists_noeye.pickle'

        if os.path.exists(prefix + filename):
            with open(prefix + filename, 'rb') as file:
                adj_lists = pickle.load(file)
            file.close()
        else:
            for item in edge_arr:
                adj_lists[id2index.index(item[0])].add(id2index.index(item[1]))
                adj_lists[id2index.index(item[1])].add(id2index.index(item[0]))

            with open(prefix + filename, 'wb') as file:
                pickle.dump(adj_lists, file)
            file.close()
        
        row1 = []
        row2 = []
        resorted_edge_list = []
        for index in range(len(labels_list)):
            for i in range(len(adj_lists[index-1])):
                tmp = list(adj_lists[index-1])
                row1.append(index-1)
                row2.append(tmp[i-1])
        resorted_edge_list.append(row1)
        resorted_edge_list.append(row2)

        x = torch.tensor(feature_list, dtype=torch.float)
        y = torch.tensor(labels_list, dtype=torch.long)
        edge_index = torch.tensor(resorted_edge_list, dtype=torch.long)
        
        '''split datasets train/val/test'''
        index = list(range(len(labels_list)))
        idx_train, idx_test, y_train, y_test = train_test_split(index, labels_list, stratify=labels, test_size=0.20, random_state=2, shuffle=True)
        idx_val, idx_test, y_val, y_test = train_test_split(idx_test, y_test, stratify=y_test, test_size=0.50, random_state=2, shuffle=True)
        
        train_pos, train_neg = self.pos_neg_split(idx_train, y_train)
        val_pos, val_neg = self.pos_neg_split(idx_val,y_val)
        test_pos, test_neg = self.pos_neg_split(idx_test,y_test)
        print('train: pos{:.4f}:neg{:.4f}'.format(len(train_pos),len(train_neg)))
        print('train: pos{:.4f}:neg{:.4f}'.format(len(val_pos),len(val_neg)))
        print(' test: pos{:.4f}:neg{:.4f}'.format(len(test_pos),len(test_neg)))

        train_mask = [False for _ in range(len(labels_list))]
        val_mask = [False for _ in range(len(labels_list))]
        test_mask = [False for _ in range(len(labels_list))]
        
        for i in range(len(idx_train)-1):
            train_mask[idx_train[i]] = True
        for i in range(len(idx_val)-1):
            val_mask[idx_val[i]] = True
        for i in range(len(idx_test)-1):
            test_mask[idx_test[i]] = True

        train_mask = torch.tensor(train_mask,dtype=bool)
        val_mask = torch.tensor(val_mask,dtype=bool)
        test_mask = torch.tensor(test_mask,dtype=bool)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        data_list = [data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    # 将数据集按照类别的正负分开
    def pos_neg_split(self, nodes, labels):
        pos_nodes = []
        neg_nodes = cp.deepcopy(nodes)
        aux_nodes = cp.deepcopy(nodes)
        for idx, label in enumerate(labels):
            if label == 1:
                pos_nodes.append(aux_nodes[idx])
                neg_nodes.remove(aux_nodes[idx])

        return pos_nodes, neg_nodes







