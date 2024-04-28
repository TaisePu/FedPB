import copy
import os
import sys
import random
import numpy as np
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
import stellargraph as sg
from stellargraph import datasets
from stellargraph.core.graph import StellarGraph
from torch_geometric.datasets import Planetoid
from sklearn import preprocessing, model_selection
from Datasets.Elliptic import Elliptic
from Datasets.Amazon import Amazon
from tqdm import tqdm
sys.path.append('//')


def load_datasets_stellargraph(args):
    if args.datasets_name == 'cora':
        dataset = datasets.Cora()
        graph, node_subjects = dataset.load()
    elif args.datasets_name == 'citeseer':
        dataset = datasets.CiteSeer()
        graph, node_subjects = dataset.load()
    elif args.datasets_name == 'pubmeddiabetes':
        dataset = datasets.PubMedDiabetes()
        graph, node_subjects = dataset.load()
    else:
        print(args.datasets_name)
        print("dataset name does not exist!")
        exit()

    # resort index
    features = graph.node_features()
    nodes_id = sorted(graph.nodes())
    edges = graph.edges()
    index = [i for i in range(len(nodes_id))]
    nodes2index = {k:v for k, v in zip(nodes_id, index)}

    nodes_index = [nodes2index[id] for id in nodes_id]
    nodes = sg.IndexedArray(features, nodes_index)
    df = pd.DataFrame()
    tmp = []
    for edge in edges:
        if edge[0] in nodes_id and edge[1] in nodes_id:
            tmp.append((nodes2index[edge[0]], nodes2index[edge[1]]))
    df['source'] = [edge[0] for edge in tmp]
    df['target'] = [edge[1] for edge in tmp]
    graph = StellarGraph(nodes=nodes, edges=df)

    keys = node_subjects.keys()
    new_keys = [nodes2index[id] for id in keys]
    values = node_subjects.values
    node_subjects = pd.Series(values, new_keys)
    return graph, node_subjects


def load_datasets_torch(args):
    if args.datasets_name == 'elliptic':
        dataset = Elliptic(root='./Folder/' + args.datasets_name + '/')
        dataset = clustering_unknown_node(args, dataset[0])
    elif args.datasets_name == 'cora':
        dataset = Planetoid(root='./Folder/' + args.datasets_name + '/', name=args.datasets_name)
    elif args.datasets_name == 'citeseer':
        dataset = Planetoid(root='./Folder/' + args.datasets_name + '/', name=args.datasets_name)
    elif args.datasets_name == 'amazon':
        dataset = Amazon(root='./Folder/' + args.datasets_name + '/')
    else:
        print(args.datasets_name)
        print("dataset name does not exist!")
        exit()
    data = dataset[0]
    return data


def stellargraph_to_torch_geo_list(graph:StellarGraph, node_label):
    data_list = []
    nodes = graph.nodes()
    nodex_index = [i for i in range(len(nodes))]
    features = graph.node_features()
    edges = graph.edges()
    labels = node_label
    nodes_to_index = {k:v for k, v in zip(nodes, nodex_index)}

    target_encoding = preprocessing.LabelBinarizer()
    target = target_encoding.fit_transform(labels)

    graph_nx = StellarGraph.to_networkx(graph)
    components_list = nx.connected_components(graph_nx)

    for component in components_list:# set
        local_nodes = [i for i in component]
        local_features = []
        local_labels = []
        local_edges = []

        for node in local_nodes:
            feature = features[nodes_to_index[node]]
            local_features.append(feature)
            for edge in edges:
                if node == edge[0] and edge[1] in local_nodes:
                    local_edges.append(edge)
            y = target[nodes_to_index[node]]
            local_labels.append(y)

        label = []
        for i in local_labels:
            for j in range(len(i)):
                if i[j] == 1:
                    label.append(j+1)

        sorted_nodes = sorted(local_nodes)
        index_nodes = [i for i in range(len(sorted_nodes))]
        sorted_dict = {k: v for k, v in zip(sorted_nodes, index_nodes)}
        source_nodes = []
        target_nodes = []

        for edge in local_edges:
            source_nodes.append(sorted_dict[edge[1]])
            target_nodes.append(sorted_dict[edge[0]])
            source_nodes.append(sorted_dict[edge[0]])
            target_nodes.append(sorted_dict[edge[1]])

        assert len(source_nodes) == len(target_nodes)
        edges_tensor = torch.stack([torch.tensor(source_nodes), torch.tensor(target_nodes)], 0)
        feature_tensor = torch.tensor(torch.tensor(local_features), dtype=torch.float)
        label_tensor = torch.tensor(torch.tensor(label), dtype=torch.float)

        data = Data(x=feature_tensor, edge_index=edges_tensor, y=label_tensor)
        data_list.append(data)
    return data_list


def stellargraph_to_torch_geo(graph:StellarGraph, node_label):
    nodes = graph.nodes()
    nodex_index = [i for i in range(len(nodes))]
    features = graph.node_features()
    edges = graph.edges()
    labels = node_label
    nodes_to_index = {k:v for k, v in zip(nodes, nodex_index)}

    target_encoding = preprocessing.LabelBinarizer()
    target = target_encoding.fit_transform(node_label)
    label = []
    for i in target:
        # elliptic
        if len(i) == 1:
            label.append(i[0])
        else:
            for j in range(len(i)):
                if i[j] == 1:
                    label.append(j)

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

    assert len(source_nodes) == len(target_nodes)
    edges_tensor = torch.stack([torch.tensor(source_nodes), torch.tensor(target_nodes)], 0)
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.int64)

    data = Data(x=feature_tensor, edge_index=edges_tensor, y=label_tensor)
    return data


def torch_to_stellargraph(args, data: Data):
    node_subjects = data.y.tolist()
    keys = [i for i in range(len(node_subjects))]
    values = [i for i in node_subjects]
    node_label = pd.Series(values, keys)

    nodes_id = []
    feats_i = []
    for index, feature in enumerate(data.x.tolist()):
        nodes_id.append(index)
        feats_i.append(feature)
    feats_i = np.array(feats_i)
    nodes = sg.IndexedArray(feats_i, nodes_id)

    pth = './Folder/' + args.datasets_name + '/whole_edges.csv'
    if not os.path.exists(pth):
        edges = data.edge_index #tensor
        tmp = []
        # print(type(edges[0][0]))
        # print(edges[0][0])
        # for u, v in zip(edges[0], edges[1]):
        #     if (v, u) in tmp:
        #         continue
        #     tmp.append((u, v))

        for i in tqdm(range(edges.shape[1])):
            u, v = edges[0][i], edges[1][i]
            if (v, u) in tmp:
                continue
            tmp.append((u, v))

        df = pd.DataFrame()
        df['source'] = [edge[0] for edge in tmp]
        df['target'] = [edge[1] for edge in tmp]
        df.to_csv('./Folder/' + args.datasets_name + '/whole_edges.csv')
    else:
        df = pd.read_csv('./Folder/' + args.datasets_name + '/whole_edges.csv')
    graph = StellarGraph(nodes=nodes, edges=df)
    return graph, node_label


def clustering_unknown_node(args, data:Data):
    features = data.x.tolist()
    y = data.y.tolist()
    neg_feature = []
    pos_feature = []
    unknown_feature = []
    for i, label in enumerate(y):
        if label == 0:
            pos_feature.append(features[i])
        elif label == 1:
            neg_feature.append(features[i])
        elif label == 2:
            unknown_feature.append(features[i])

    pos_centroid = calculate_centroid(pos_feature)
    neg_centroid = calculate_centroid(neg_feature)

    euclidean = []
    cosine = []
    manhattan = []
    new_y = copy.deepcopy(y)

    for i, label in enumerate(y):
        if label == 2:
            pos_eu = euclidean_distance(pos_centroid, features[i])
            neg_eu = euclidean_distance(neg_centroid, features[i])
            euclidean.append((pos_eu, neg_eu))
            pos_cosin = cosine_similarity(pos_centroid, features[i])
            neg_cosin = cosine_similarity(neg_centroid, features[i])
            cosine.append((pos_cosin, neg_cosin))
            pos_man = manhattan_distance(pos_centroid, features[i])
            neg_man = manhattan_distance(neg_centroid, features[i])
            manhattan.append((pos_man, neg_man))

            # new_y[i] = 0 if pos_cosin < neg_cosin else 1
            new_y[i] = random.randint(0, 1)
            continue
        new_y[i] = label
    new_y = torch.tensor(new_y, dtype=torch.long)
    new_data = Data(x=data.x, edge_index=data.edge_index, y=new_y, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
    return [new_data]


def calculate_centroid(features):
    # Initialize an empty array to store the centroid
    centroid = np.zeros(len(features[0]))
    # For each feature dimension:
    for i in range(len(features[0])):
        # Calculate the mean of the feature values for all nodes
        mean = np.mean([node[i] for node in features])
        # Store the mean in the centroid array
        centroid[i] = mean
    # Return the centroid array
    return centroid


def euclidean_distance(centroid, feature):
    # Calculate the Euclidean distance between the centroid and the feature
    distance = np.linalg.norm(centroid - feature)
    # Return the Euclidean distance
    return distance


def cosine_similarity(centroid, feature):
    # Calculate the dot product of the centroid and the feature
    dot_product = np.dot(centroid, feature)
    # Calculate the magnitudes of the centroid and the feature
    magnitude_centroid = np.linalg.norm(centroid)
    magnitude_feature = np.linalg.norm(feature)
    # Calculate the cosine similarity
    similarity = dot_product / (magnitude_centroid * magnitude_feature)
    # Return the cosine similarity
    return similarity


def manhattan_distance(centroid, feature):
    # Calculate the Manhattan distance between the centroid and the feature
    distance = np.sum(np.abs(centroid - feature))
    # Return the Manhattan distance
    return distance


def train_test_split(args, graph:StellarGraph, node_label):
    path = './Folder/' + args.datasets_name + '/saved'
    # if os.path.exists(path):
    #     return load_train_test_graph(path)
    # os.mkdir(path)

    all_nodes_ids = graph.nodes().tolist()
    train_nodes, test_nodes = model_selection.train_test_split(all_nodes_ids, train_size=0.6, stratify=node_label)
    _, test_nodes = model_selection.train_test_split(all_nodes_ids, test_size=0.4, stratify=node_label)

    train_nodes.sort()
    train_loc = graph.node_ids_to_ilocs(train_nodes)
    train_label_tmp = node_label[train_loc] #Series
    data = train_label_tmp.values
    train_label = pd.Series(data)
    feats_i = graph.node_features()[train_loc]
    nodes_id = [i for i in range(len(train_nodes))]
    nodes = sg.IndexedArray(feats_i, nodes_id)
    edges = graph.edges()
    tmp = []
    for (u, v) in edges:
        if u in train_nodes and v in train_nodes:
            tmp.append((train_nodes.index(u), train_nodes.index(v)))
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in tmp]
    df['target'] = [edge[1] for edge in tmp]
    train_graph = StellarGraph(nodes=nodes, edges=df)

    # save datasets
    df.to_csv('./Folder/' + args.datasets_name + '/saved/edges_train.csv')
    df_feature = pd.DataFrame()
    # df_feature['node'] = [node for node in nodes_id]
    for id, feature in enumerate(feats_i):
        df_feature['node_{}'.format(id)] = [float(i) for i in feature]
    df_feature.to_csv('./Folder/' + args.datasets_name + '/saved/feature_train.csv')
    df_subj = pd.DataFrame()
    df_subj['node'] = [node for node in nodes_id]
    df_subj['subj'] = [subj for subj in train_label]
    df_subj.to_csv('./Folder/' + args.datasets_name + '/saved/subj_train.csv')
    del edges
    del tmp
    del df

    test_nodes.sort()
    test_loc = graph.node_ids_to_ilocs(test_nodes)
    test_label_tmp = node_label[test_loc]
    data = test_label_tmp.values
    test_label = pd.Series(data)
    feats_i = graph.node_features()[test_loc]
    nodes_id = [i for i in range(len(test_nodes))]
    nodes = sg.IndexedArray(feats_i, nodes_id)

    edges = graph.edges()
    tmp = []
    for (u, v) in edges:
        if u in test_nodes and v in test_nodes:
            tmp.append((test_nodes.index(u), test_nodes.index(v)))
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in tmp]
    df['target'] = [edge[1] for edge in tmp]
    test_graph = StellarGraph(nodes=nodes, edges=df)

    # save datasets
    df.to_csv('./Folder/' + args.datasets_name + '/saved/edges_test.csv')
    df_feature = pd.DataFrame()
    # df_feature['node'] = [node for node in nodes_id]
    for id, feature in enumerate(feats_i):
        df_feature['node_{}'.format(id)] = [float(i) for i in feature]
    df_feature.to_csv('./Folder/' + args.datasets_name + '/saved/feature_test.csv')
    df_subj = pd.DataFrame()
    df_subj['node'] = [node for node in nodes_id]
    df_subj['subj'] = [subj for subj in test_label]
    df_subj.to_csv('./Folder/' + args.datasets_name + '/saved/subj_test.csv')
    return train_graph, test_graph, train_label, test_label


def load_train_test_graph(path):
    train_feats_df = pd.read_csv(path+'/feature_train.csv')
    train_edges_df = pd.read_csv(path+'/edges_train.csv')
    train_subj_df  = pd.read_csv(path+'/subj_train.csv')

    test_feats_df = pd.read_csv(path+'/feature_test.csv')
    test_edges_df = pd.read_csv(path+'/edges_test.csv')
    test_subj_df  = pd.read_csv(path+'/subj_test.csv')

    train_feats = [np.array(train_feats_df['node_{}'.format(i)]) for i in range(train_feats_df.shape[1]-1)]
    train_nodes = [i for i in train_subj_df['node']]
    train_subj  = [i for i in train_subj_df['subj']]
    # target_encoding = preprocessing.LabelBinarizer()
    # train_target = target_encoding.fit_transform(train_subj)
    train_target = pd.Series(train_subj)
    train_nodes = sg.IndexedArray(np.array(train_feats), np.array(train_nodes))
    train_graph = StellarGraph(nodes=train_nodes, edges=train_edges_df)

    test_feats = [np.array(test_feats_df['node_{}'.format(i)]) for i in range(test_feats_df.shape[1]-1)]
    test_nodes = [i for i in test_subj_df['node']]
    test_subj  = [i for i in test_subj_df['subj']]
    # target_encoding = preprocessing.LabelBinarizer()
    # test_target = target_encoding.fit_transform(test_subj)
    test_target = pd.Series(test_subj)
    test_nodes = sg.IndexedArray(np.array(test_feats), np.array(test_nodes))
    test_graph = StellarGraph(nodes=test_nodes, edges=test_edges_df)
    return train_graph, test_graph, train_target, test_target



















