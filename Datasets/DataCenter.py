import torch
import random
from collections import defaultdict
from torch_geometric.data.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class DataCenter(object):
    """docstring for DataCenter"""
    def __init__(self, args):
        super(DataCenter, self).__init__()
        self.args = args
        self.client_num = args.num_clients
        self.device = args.device

    def load_dataSet(self):
        data = Planetoid(name=self.args.datasets_name, root=self.args.root_path + '/Folder/')
        data = data[0]
        # graph = to_networkx(data)
        # graph = graph.to_undirected()
        # components = nx.connected_components(graph)
        # for sub_graph in components:
        #     print(sub_graph)

        x = data.x.numpy()
        y = torch.flatten(data.y).numpy()
        edge_index = data.edge_index.numpy()
        edge_cnt = edge_index.shape[1]

        """adj_lists split the dataset"""
        # what is adj?
        adj_lists = defaultdict(set)
        node_map = {}
        for i in range(edge_cnt):
            paper1 = edge_index[0][i]
            paper2 = edge_index[1][i]
            if not paper1 in node_map:
                node_map[paper1] = len(node_map)
            if not paper2 in node_map:
                node_map[paper2] = len(node_map)
            adj_lists[node_map[paper1]].add(node_map[paper2])
            adj_lists[node_map[paper2]].add(node_map[paper1])

        x = x[list(node_map)]
        y = y[list(node_map)]
        for i in range(edge_cnt):
            edge_index[0][i] = node_map[edge_index[0][i]]
            edge_index[1][i] = node_map[edge_index[1][i]]
        assert len(x) == len(y) == len(adj_lists)

        num_classes = len(set(y))
        num_nodes = x.shape[0]
        len_feats = x.shape[1]

        print("num_labels", num_classes)
        print("total node num", num_nodes)
        print("feature dimension", len_feats)
        print("total edges", edge_cnt // 2)

        setattr(self, self.args.datasets_name + '_num_classes', num_classes)
        setattr(self, self.args.datasets_name + '_in_feats', len_feats)

        sampling_node_nums = int(num_nodes * self.args.global_sample_rate)
        test_index = list(np.random.permutation(np.arange(num_nodes))[:sampling_node_nums])
        print("node in test", sampling_node_nums, flush=True)

        distibution = np.zeros(num_classes)
        for node in y[test_index]:
            distibution[node] += 1
        print("test distribution")
        print((np.array(distibution) / len(test_index)).tolist(), flush=True)

        pos = {}
        for i, node in enumerate(test_index):
            pos[node] = i

        test_edge_index_u = []
        test_edge_index_v = []
        for u in test_index:
            for v in test_index:
                if (v in adj_lists[u]):
                    """undirect edge"""
                    test_edge_index_u.append(pos[u])
                    test_edge_index_u.append(pos[v])
                    test_edge_index_v.append(pos[v])
                    test_edge_index_v.append(pos[u])

        assert len(test_edge_index_u) % 2 == 0 and len(
            test_edge_index_v) % 2 == 0

        test_edge_index = torch.stack([torch.tensor(test_edge_index_u),
                                       torch.tensor(test_edge_index_v)], 0)

        test_x = torch.tensor(x[test_index])
        test_y = torch.tensor(y[test_index])
        test_data = Data(x=test_x, edge_index=test_edge_index, y=test_y)
        setattr(self, self.args.datasets_name + '_test_data', test_data)
        setattr(self, self.args.datasets_name + '_test_index', test_index)

        # sort node by label
        delete_index = list(test_index)
        sampling_node_num = int(num_nodes * self.args.sample_rate)
        print("node in each client", sampling_node_num)

        # split dataset
        list_clients_data = []
        index_list = []
        for client_id in range(self.client_num):
            """sort nodes"""
            client_nodes = np.delete(np.arange(num_nodes), delete_index)
            node_by_label = defaultdict(list)

            for node in client_nodes:
                node_by_label[y[node]].append(node)
            """major label nodes"""
            holding_label = np.random.permutation(np.arange(num_classes))[:self.args.major_label]
            holding_label_index = []
            print("Major label of", "client", client_id, ":", holding_label)

            for label in holding_label:
                holding_label_index += node_by_label[label]
            major_num = int(sampling_node_num * self.args.major_rate)

            if (major_num > len(holding_label_index)):
                print("Major label not enough nodes")

            major_num = min(major_num, len(holding_label_index))
            major_index = list(np.random.permutation(holding_label_index)[:major_num])
            major_pos = []

            for pos, node in enumerate(client_nodes):
                if node in major_index:
                    major_pos.append(pos)
            major_pos = np.array(major_pos, dtype=int)

            # other label
            rest_num = sampling_node_num - major_num
            rest_index = np.delete(client_nodes, major_pos)
            other_index = list(np.random.permutation(rest_index)[:rest_num])
            index = major_index + other_index

            connect = set()
            for u in index:
                for v in index:
                    if (u in adj_lists[v]):
                        connect.add(u)
                        connect.add(v)
            index = list(connect)
            print("node num in client", client_id + 1, ":", len(index))
            random.shuffle(index)
            """delete the local test nodes"""
            delete_index += index[len(index) - self.args.test_num:]

            pos = {}  # mapping from the original node to the new node
            for i, node in enumerate(index):
                pos[node] = i
            index_list += [index]

            client_edge_index_u = []
            client_edge_index_v = []
            for u in index:
                for v in index:
                    if (u in adj_lists[v]):
                        """双向边"""
                        client_edge_index_u.append(pos[u])
                        client_edge_index_u.append(pos[v])
                        client_edge_index_v.append(pos[v])
                        client_edge_index_v.append(pos[u])
            print(sorted(client_edge_index_u))
            assert len(client_edge_index_u) % 2 == 0 and len(client_edge_index_v) % 2 == 0
            client_edge_index = torch.stack([torch.tensor(client_edge_index_u),
                                             torch.tensor(client_edge_index_v)], 0)
            # client local data
            client_x = torch.tensor(x[index])
            client_y = torch.tensor(y[index])
            client_data = Data(x=client_x, edge_index=client_edge_index, y=client_y)
            list_clients_data.append(client_data)
            print(f"Client {client_id + 1} finish loading data", flush=True)

            """print the distibution"""
            distibution = np.zeros(num_classes)
            for node in client_y:
                distibution[node] += 1
            print("distribution")
            print((np.array(distibution) / len(client_y)).tolist(), flush=True)

        setattr(self, self.args.datasets_name + '_local_train_data', list_clients_data)
        setattr(self, self.args.datasets_name + '_index_list', index_list)
        setattr(self, self.args.datasets_name + '_total_data', data)
