import copy
import torch
import numpy as np
from sklearn import preprocessing
from Server.Server_PB import Server
from Client.Client_PB import Client
from tqdm import tqdm
from Utils.log import log

times = 0

def fedpb(args, graph, node_subjects, local_graph,
            local_subj, local_target, local_nodes_ids):
    target_encoding = preprocessing.LabelBinarizer()
    global_targets = target_encoding.fit_transform(node_subjects)
    num_classes = target_encoding.classes_

    # server & clients
    server = Server(args, graph, node_subjects)
    list_clients = [Client(args, local_graph[i], local_subj[i], local_target[i], num_classes) for i in range(args.num_clients)]

    # participant relationship contruction
    # participant & alliance
    list_pre_features = []
    list_filled_graphs = []
    for client in list_clients:
        client.train_neighborhood_gen()
        # the shape of hided/whole graph is different
        client.build_new_neighborhood_gen()

        faeture = torch.tensor(client.graph_sg.node_features(), dtype=torch.float32).cuda()
        edges = np.asarray(client.graph_sg.edges(use_ilocs=True))
        edges = torch.tensor(edges.astype(np.int32))
        adj = client.get_adj(edges, len(client.graph_sg.nodes())).cuda()
        pred_missing, pred_feats, pre_target = client.neighborhood_gen_model2(faeture, edges.cuda(), adj)

        tmp_feature = pred_feats.reshape((-1, args.num_pred_node, client.hided_feature_len))
        tmp = tmp_feature.cpu()
        tmp_feature = tmp.detach().numpy().tolist()
        list_pre_features.append(tmp_feature)

        filled_nodes, filled_graph = client.fill_graph(args, pred_missing, pred_feats, client.hided_feature_len)
        list_filled_graphs.append(filled_graph)

    # calculate the similarity of each client
    matrix_distance = calculate_similarity(local_graph, list_pre_features)
    tmp = normalize(matrix_distance)
    server.client_adj_matrix = get_adjacency_matrix(tmp)

    parameters_list = [[] for _ in range(args.num_clients)]
    loss_dict = {}
    global times
    participant_acc_list = [0.0 for _ in range(args.num_clients)]

    # shared, unified initial parameters.
    parameters = server.aggregation_avg([client.model.state_dict() for client in list_clients])
    for round in tqdm(range(args.num_rounds_global), desc='global rounds'):
        # step 1: local training with updated parameters.
        for id, client in enumerate(list_clients):
            client.model.load_state_dict(server.model.state_dict())
            loss, parameters, acc, f1 = client.local_train()
            participant_acc_list[id] = acc
            times += 2
            loss_dict[str(id)] = (f1, parameters)
            parameters_list[id].append(parameters)

        # step 2: broadcast parameters and local training again.
        # broadcast_step = 2
        for id, client in enumerate(list_clients):
            # step 1
            for neighborhood in server.get_adj_clients(id):
                list_clients[neighborhood].model.load_state_dict(client.model.state_dict())
                loss, parameters, acc, f1 = list_clients[neighborhood].local_train()
                times += 2
                # acc or f1
                loss_dict[str(id)+str(neighborhood)] = (loss, parameters)
                parameters_list[neighborhood].append(parameters)
                # step 2
                for second_neighborhood in server.get_adj_clients(neighborhood):
                    list_clients[second_neighborhood].model.load_state_dict(list_clients[neighborhood].model.state_dict())
                    loss, parameters, acc, f1 = list_clients[second_neighborhood].local_train()
                    times += 2
                    # acc or f1
                    loss_dict[str(id) + str(neighborhood) + str(second_neighborhood)] = (loss, parameters)
                    parameters_list[second_neighborhood].append(parameters)
        # Shaply
        shaply_values = [0.1 for _ in range(args.num_clients)]
        count = 1
        for id in range(args.num_clients):
            for key in loss_dict.keys():
                if key == str(id):
                    continue
                elif key[-1] == str(id):
                    shaply_values[id] += abs(loss_dict[str(key)][0]-loss_dict[str(key[:-1])][0])
                    count += 1
            shaply_values[id] /= count
        v_sum = sum(shaply_values)
        new_shaply_values = [v/v_sum for v in shaply_values]
        log(f'Shaply Value:{new_shaply_values}')

        # Aggregation with weight(Shaply).
        w_last = [client_para[-1] for client_para in parameters_list]
        w_load = []
        for client_id in range(args.num_clients):
            cur_load = copy.deepcopy(w_last[client_id])
            for k in cur_load.keys():
                for i in range(-1, 2):
                    if i == 0: continue
                    cur_load[k] += w_last[(client_id + i + args.num_clients) % args.num_clients][k]
                cur_load[k] = torch.div(cur_load[k], 3)
            w_load += [cur_load]

        # data number weight
        data_number_weight = [len(local_nodes_ids[i]) for i in range(args.num_clients)]

        # performance weight
        parameters = server.aggregation_avg(w_load)
        f1, acc, loss = server.evaluate()
        performance_weight = [abs(p_acc-acc) for p_acc in participant_acc_list]
        performance_weight = [acc/sum(performance_weight) for acc in performance_weight]

        # arrgregation with weight
        parameters = server.aggregation_avg(w_load, new_shaply_values)
        f1, acc, loss = server.evaluate()
        log('FedPB! round:{}, acc:{:.4f}, f1:{:.4f}, loss:{:.4f}'.format(round, acc, f1, loss))
    print(times)


def calculate_similarity(list_original_graph, list_filled_feature):
    list_centroid = []
    dis = 0.0
    matrix_distance = np.zeros((len(list_original_graph), len(list_original_graph)))
    for graph in list_original_graph:
        features = graph.node_features()
        list_centroid.append(calculate_centroid(features))
    for i, centroid in enumerate(list_centroid):
        for j, client in enumerate(list_filled_feature):
            for c, node in enumerate(client):
                for f in node:
                    dis += cosine_similarity(centroid, f)
            matrix_distance[i][j] = dis
            dis = 0.0
    return matrix_distance


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


def normalize(numpy_matrix):
    normalizer = preprocessing.MinMaxScaler()
    normalized_matrix = normalizer.fit_transform(numpy_matrix)
    return normalized_matrix


def get_adjacency_matrix(normalize_matrix):
    adj_matrix = copy.deepcopy(normalize_matrix)
    for i, row in enumerate(normalize_matrix):
        for j, client in enumerate(row):
            adj_matrix[i][j] = 1 if client < normalize_matrix[i][i] else 0
    return adj_matrix