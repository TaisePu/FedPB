import os
import sys
import torch
import random
import time
import numpy as np
import pandas as pd
import stellargraph as sg
from stellargraph.core.graph import StellarGraph
from Methods.FedPB import fedpb

from options import args_parser, save_options
from Datasets.Datasets import load_datasets_stellargraph, load_datasets_torch, torch_to_stellargraph, train_test_split
from Utils.louvain_networkx import graph_cut_community, louvain_graph_cut, graph_cut_by_number, graph_cut_random
from Utils.log import log, save_training_log
from torch_geometric.utils.convert import to_networkx
import networkx as nx
sys.path.append('//')


def check_each_graph_legal(local_graph, local_label, local_target, local_nodes_ids):
    for index, target in enumerate(local_target):
        target = np.array(target)
        print(f'client:{index}, Class imbalance racio:{np.mean(target)}')
    for index, graph in enumerate(local_graph):
        graph_nx = StellarGraph.to_networkx(graph)
        components = nx.connected_components(graph_nx)
        # components is a generator object.
        if not sum(1 for _ in components) == 1:
            print(f'client:{index}\'s dataset is not a connect graph!')
            exit()
    for graph, nodes in zip(local_graph, local_nodes_ids):
        edges = graph.edges()
        sub_nodes = graph.nodes()
        for item in edges:
            if item[0] not in nodes or item[1] not in nodes:
                print('edge not in nodes!')
                exit()
            if item[0] not in sub_nodes or item[1] not in sub_nodes:
                print('edge not in sub_nodes!')
                exit()


if __name__ == '__main__':
    args = args_parser()

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    save_options(args)

    # Cora, Citeseer, Pubmeddiabetes
    graph, node_label = load_datasets_stellargraph(args)

    # train/test graph is not a connect graph
    # some nodes in train/test graph are overlapping.
    train_graph, test_graph, train_label, test_label = train_test_split(args, graph, node_label)

    # loouvain: graph split based communication found
    local_graph, local_label, local_target, local_nodes_ids = graph_cut_by_number(args, train_graph, train_label)
    # check_each_graph_legal(local_graph, local_label, local_target, local_nodes_ids)


    # FedPB
    log('FedPB')
    start_time = time.time()
    fedpb(args, test_graph, test_label, local_graph, local_label, local_target, local_nodes_ids)
    end_time = time.time()
    log(f'\nTotal time: {end_time - start_time}')
    save_training_log(args, 'log_fedpb.txt')



