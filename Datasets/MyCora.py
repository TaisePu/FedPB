import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from stellargraph.core.graph import StellarGraph
from sklearn import preprocessing


class MyCora(InMemoryDataset):
    def __init__(self, root, graph:StellarGraph, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.graph = graph

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)

    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        edges = self.graph.edges()
        nodes = self.graph.nodes()
        features = self.graph.node_features()

        target_encoding = preprocessing.LabelBinarizer()
        target = target_encoding.fit_transform(node_label)
        label = []
        for i in target:
            for j in range(len(i)):
                if i[j] == 1:
                    label.append(j + 1)

        sorted_nodes = sorted(nodes)
        index_nodes = [i for i in range(len(sorted_nodes))]
        sorted_dict = {k: v for k, v in zip(sorted_nodes, index_nodes)}
        source_nodes = []
        target_nodes = []
        for edge in edges:
            source_nodes.append(sorted_dict[edge[1]])
            target_nodes.append(sorted_dict[edge[0]])
            source_nodes.append(sorted_dict[edge[0]])
            target_nodes.append(sorted_dict[edge[1]])

        edges_tensor = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        feature_tensor = torch.tensor(features, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)

        data = Data(x=feature_tensor, edge_index=edges_tensor, y=label_tensor)
        data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])