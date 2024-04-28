import os
import louvain.community as community_louvain
import networkx as nx
from stellargraph.core.graph import StellarGraph
import stellargraph as sg
import numpy as np
from sklearn import preprocessing
import pandas as pd
from random import sample


def louvain_graph_cut(args, whole_graph:StellarGraph, node_subjects):
    edges = np.copy(whole_graph.edges())
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in edges]
    df['target'] = [edge[1] for edge in edges]
    # multi graph
    graph = StellarGraph.to_networkx(whole_graph)

    # community partition
    # partition is a dict (node, communication), then transform to groups(List)
    partition = community_louvain.best_partition(graph)
    groups = []

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])

    partition_groups = {group_i:[] for group_i in groups}
    # partition_groups is a dictionary{partition:[nodes]}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)
    group_len_max = len(list(whole_graph.nodes()))//args.num_clients-args.delta#20

    # groups is the index of communication
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups)+1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}
    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]

    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)}
    owner_node_ids = {owner_id:[] for owner_id in range(args.num_clients)}

    owner_nodes_len = len(list(graph.nodes()))//args.num_clients
    owner_list = [i for i in range(args.num_clients)]
    owner_ind = 0

    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) >= owner_nodes_len + args.delta:
            owner_ind = (owner_ind + 1) % len(owner_list)
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]

    for owner_i in owner_node_ids.keys():
        print('nodes len for '+str(owner_i)+' = '+str(len(owner_node_ids[owner_i])))

    local_graph = []
    local_node_subj = []
    local_nodes_ids = []
    local_target = []
    local_node_subj_0 = []

    target_encoding = preprocessing.LabelBinarizer()
    target = target_encoding.fit_transform(node_subjects)
    subj_set = list(set(node_subjects.values))

    for client_id in range(args.num_clients):
        partition_i = owner_node_ids[client_id]
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        sbj_i = node_subjects.copy(deep=True)
        sbj_i.values[:] = "" if node_subjects.values[0].__class__ == str else 0
        sbj_i.values[locs_i] = node_subjects.values[locs_i]
        local_node_subj_0.append(sbj_i)

    count = []
    for client_id in range(args.num_clients):
        count_i = {k:[] for k in subj_set}
        sbj_i = local_node_subj_0[client_id]
        for i in sbj_i.index:
            if sbj_i[i] != 0 and sbj_i[i] != "":
                count_i[sbj_i[i]].append(i)
        count.append(count_i)

    for k in subj_set:
        for client_id in range(args.num_clients):
            if len(count[client_id][k]) < 2:
                for j in range(args.num_clients):
                    if len(count[j][k]) > 2:
                        id = count[j][k][-1]
                        count[j][k].remove(id)
                        count[client_id][k].append(id)
                        owner_node_ids[client_id].append(id)
                        owner_node_ids[j].remove(id)

    # split nodes & edges & features
    for client_id in range(args.num_clients):
        partition_i = owner_node_ids[client_id]
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        sbj_i = node_subjects.copy(deep=True)
        sbj_i.values[:] = "" if node_subjects.values[0].__class__ == str else 0
        sbj_i.values[locs_i] = node_subjects.values[locs_i]
        local_node_subj.append(sbj_i)
        local_target_i = np.zeros(target.shape, np.int32)
        local_target_i[locs_i] += target[locs_i]
        local_target.append(local_target_i)
        local_nodes_ids.append(partition_i)
        feats_i = np.zeros(whole_graph.node_features().shape)
        feats_i[locs_i] = feats_i[locs_i] + whole_graph.node_features()[locs_i]
        nodes_id = whole_graph.nodes()
        nodes = sg.IndexedArray(feats_i, nodes_id)
        graph_i = StellarGraph(nodes=nodes, edges=df)
        local_graph.append(graph_i)

    return local_graph, local_node_subj, local_target, local_nodes_ids


def graph_cut_community(args, whole_graph:StellarGraph, node_subjects):
    # path = './Folder/' + args.datasets_name + '/saved'
    # if os.path.exists(path):
    #     return load_saved_subgraph(args, path)

    edges = np.copy(whole_graph.edges()).tolist()
    # multi sub graph
    graph = StellarGraph.to_networkx(whole_graph)

    # community partition
    # partition is a dict (node, communication), then transform to groups(List)
    partition = community_louvain.best_partition(graph)
    groups = []

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])

    partition_groups = {group_i:[] for group_i in groups}
    # partition_groups is a dictionary{partition:[nodes]}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)
    group_len_max = len(list(whole_graph.nodes()))//args.num_clients-args.delta#20

    # groups is the index of communication
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups)+1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}
    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]

    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)}
    owner_node_ids = {owner_id:[] for owner_id in range(args.num_clients)}

    owner_nodes_len = len(list(graph.nodes()))//args.num_clients
    owner_list = [i for i in range(args.num_clients)]
    owner_ind = 0

    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) >= owner_nodes_len + args.delta:
            owner_ind = (owner_ind + 1) % len(owner_list)
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]

    for owner_i in owner_node_ids.keys():
        print('nodes len for '+str(owner_i)+' = '+str(len(owner_node_ids[owner_i])))

    local_graph = []
    local_node_subj = []
    local_nodes_ids = []
    local_target = []
    local_node_subj_0 = []

    target_encoding = preprocessing.LabelBinarizer()
    target = target_encoding.fit_transform(node_subjects)
    subj_set = list(set(node_subjects.values))

    for client_id in range(args.num_clients):
        partition_i = owner_node_ids[client_id]
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        sbj_i = node_subjects.copy(deep=True)
        sbj_i.values[:] = "" if node_subjects.values[0].__class__ == str else 0
        sbj_i.values[locs_i] = node_subjects.values[locs_i]
        local_node_subj_0.append(sbj_i)

    count = []
    for client_id in range(args.num_clients):
        count_i = {k:[] for k in subj_set}
        sbj_i = local_node_subj_0[client_id]
        for i in sbj_i.index:
            if sbj_i[i] != 0 and sbj_i[i] != "":
                count_i[sbj_i[i]].append(i)
        count.append(count_i)

    for k in subj_set:
        for client_id in range(args.num_clients):
            if len(count[client_id][k]) < 2:
                for j in range(args.num_clients):
                    if len(count[j][k]) > 2:
                        id = count[j][k][-1]
                        count[j][k].remove(id)
                        count[client_id][k].append(id)
                        owner_node_ids[client_id].append(id)
                        owner_node_ids[j].remove(id)

    """split nodes & edges & features"""
    for client_id in range(args.num_clients):
        partition_i = owner_node_ids[client_id]
        local_nodes_ids.append(partition_i)

        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        keys = node_subjects.keys()[locs_i]
        values = node_subjects.values[locs_i]
        sbj_i = pd.Series(values, keys)
        local_node_subj.append(sbj_i)

        local_target_i = target[locs_i]
        local_target.append(local_target_i)

        feats_i = whole_graph.node_features()[locs_i]

        nodes_id = partition_i
        nodes = sg.IndexedArray(feats_i, nodes_id)
        df = pd.DataFrame()
        tmp = []
        for edge in edges:
            if edge[0] in nodes_id and edge[1] in nodes_id:
                tmp.append(edge)
        df['source'] = [edge[0] for edge in tmp]
        df['target'] = [edge[1] for edge in tmp]
        graph_i = StellarGraph(nodes=nodes, edges=df)
        local_graph.append(graph_i)
    del df

    """bridge subgraph by adding edge"""
    whole_graph_nx = StellarGraph.to_networkx(whole_graph)
    local_graph_new = []
    for i, graph in enumerate(local_graph):
        tmp = []
        local_edges = np.copy(graph.edges()).tolist()
        graph_nx = StellarGraph.to_networkx(graph)
        for set_graph in nx.connected_components(graph_nx):
            tmp.append(set_graph.pop())
        for index in range(len(tmp)-1):
            local_edges.append([tmp[index], tmp[index+1]])

        nodes_id = graph.nodes()
        feats_i = graph.node_features()
        nodes = sg.IndexedArray(feats_i, nodes_id)
        df_edges = pd.DataFrame()
        df_edges['source'] = [edge[0] for edge in local_edges]
        df_edges['target'] = [edge[1] for edge in local_edges]
        graph_new = StellarGraph(nodes=nodes, edges=df_edges)
        local_graph_new.append(graph_new)

        """save datasets"""
        df_edges.to_csv('./Folder/' + args.datasets_name + '/saved/edges_{}.csv'.format(i))
        df_feature = pd.DataFrame()
        for id, feature in enumerate(feats_i):
            df_feature['node_{}'.format(id)] = [float(i) for i in feature]
        df_feature.to_csv('./Folder/' + args.datasets_name + '/saved/feature_{}.csv'.format(i))
        tmp = local_node_subj[i].tolist()
        df_subj = pd.DataFrame()
        df_subj['node'] = [node for node in nodes_id]
        df_subj['subj'] = [subj for subj in tmp]
        df_subj.to_csv('./Folder/' + args.datasets_name + '/saved/subj_{}.csv'.format(i))
    return local_graph, local_node_subj, local_target, local_nodes_ids


def graph_cut_by_number(args, whole_graph:StellarGraph, node_subjects):
    local_nodes_ids = []
    local_node_subj = []
    local_target = []
    local_graph = []

    nodes = whole_graph.nodes()
    edges = whole_graph.edges()
    feature = whole_graph.node_features()
    target_encoding = preprocessing.LabelBinarizer()
    target = target_encoding.fit_transform(node_subjects)
    lens = len(nodes)
    nodes_list = [i for i in range(lens)]

    # split nodes & edges & features
    for client_id in range(args.num_clients):
        partition_i = sample(nodes_list, lens//(5-client_id))
        local_nodes_ids.append(partition_i)
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        keys = node_subjects.keys()[locs_i]
        values = node_subjects.values[locs_i]
        sbj_i = pd.Series(values, keys)
        local_node_subj.append(sbj_i)

        local_target_i = target[locs_i]
        local_target.append(local_target_i)

        feats_i = feature[locs_i]
        nodes = sg.IndexedArray(feats_i, partition_i)
        df = pd.DataFrame()
        tmp = []
        for edge in edges:
            if edge[0] in partition_i and edge[1] in partition_i:
                tmp.append(edge)
        df['source'] = [edge[0] for edge in tmp]
        df['target'] = [edge[1] for edge in tmp]
        graph_i = StellarGraph(nodes=nodes, edges=df)
        local_graph.append(graph_i)

    """bridge subgraph by adding edge"""
    whole_graph_nx = StellarGraph.to_networkx(whole_graph)
    local_graph_new = []
    for i, graph in enumerate(local_graph):
        tmp = []
        local_edges = np.copy(graph.edges()).tolist()
        graph_nx = StellarGraph.to_networkx(graph)
        for set_graph in nx.connected_components(graph_nx):
            tmp.append(set_graph.pop())
        for index in range(len(tmp)-1):
            local_edges.append([tmp[index], tmp[index+1]])

        nodes_id = graph.nodes()
        feats_i = graph.node_features()
        nodes = sg.IndexedArray(feats_i, nodes_id)
        df_edges = pd.DataFrame()
        df_edges['source'] = [edge[0] for edge in local_edges]
        df_edges['target'] = [edge[1] for edge in local_edges]
        graph_new = StellarGraph(nodes=nodes, edges=df_edges)
        local_graph_new.append(graph_new)

    for id, owner_i in enumerate(local_nodes_ids):
        print('nodes len for '+str(id)+' = '+str(len(owner_i)))
    return local_graph, local_node_subj, local_target, local_nodes_ids


def graph_cut_random(args, whole_graph:StellarGraph, node_subjects):
    local_nodes_ids = []
    local_node_subj = []
    local_target = []
    local_graph = []

    nodes = whole_graph.nodes()
    edges = whole_graph.edges()
    feature = whole_graph.node_features()
    target_encoding = preprocessing.LabelBinarizer()
    target = target_encoding.fit_transform(node_subjects)
    lens = len(nodes)
    nodes_list = [i for i in range(lens)]

    # split nodes & edges & features
    for client_id in range(args.num_clients):
        partition_i = sample(nodes_list, lens//5)
        local_nodes_ids.append(partition_i)
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        keys = node_subjects.keys()[locs_i]
        values = node_subjects.values[locs_i]
        sbj_i = pd.Series(values, keys)
        local_node_subj.append(sbj_i)

        local_target_i = target[locs_i]
        local_target.append(local_target_i)

        feats_i = feature[locs_i]
        nodes = sg.IndexedArray(feats_i, partition_i)
        df = pd.DataFrame()
        tmp = []
        for edge in edges:
            if edge[0] in partition_i and edge[1] in partition_i:
                tmp.append(edge)
        df['source'] = [edge[0] for edge in tmp]
        df['target'] = [edge[1] for edge in tmp]
        graph_i = StellarGraph(nodes=nodes, edges=df)
        local_graph.append(graph_i)

    for id, owner_i in enumerate(local_nodes_ids):
        print('nodes len for '+str(id)+' = '+str(len(owner_i)))
    return local_graph, local_node_subj, local_target, local_nodes_ids


def load_saved_subgraph(args, path: str):
    local_graph = []
    local_node_subj = []
    local_target = []
    local_nodes_ids = []

    for client in range(args.num_clients):
        nodes_df = pd.read_csv(path+'/subj_{}.csv'.format(client))
        feats_df = pd.read_csv(path+'/feature_{}.csv'.format(client))
        edges_df = pd.read_csv(path+'/edges_{}.csv'.format(client))
        subj_df  = pd.read_csv(path+'/subj_{}.csv'.format(client))

        feats_i = [np.array(feats_df['node_{}'.format(i)]) for i in range(feats_df.shape[1]-1)]
        nodes_i = [i for i in nodes_df['node']]
        subj_i  = [i for i in subj_df['subj']]
        target_encoding = preprocessing.LabelBinarizer()
        target = target_encoding.fit_transform(subj_i)

        nodes = sg.IndexedArray(np.array(feats_i), np.array(nodes_i))
        graph_i = StellarGraph(nodes=nodes, edges=edges_df)
        local_graph.append(graph_i)
        local_node_subj.append(subj_i)
        local_target.append(target)
        local_nodes_ids.append(nodes_i)
    return local_graph, local_node_subj, local_target, local_nodes_ids
