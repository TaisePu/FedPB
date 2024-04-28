import argparse
import os


def args_parser():
    path_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=path_dir)
    parser.add_argument('--test_name', type=str, default='test200')
    parser.add_argument('--describe', type=str, help='you can write something about this test.')
    parser.add_argument('--device', type=str, default='cuda', help='CPU / GPU device.')
    parser.add_argument('--seed', type=int, default=17, help='seed for randomness')
    parser.add_argument('--datasets_name', type=str, default='cora', help='cora, citeseer, pubmeddiabetes,'
                                                                                    'elliptic, amazon')
    parser.add_argument('--path_save_keypoints', type=str,
                        default=os.path.join(path_dir, 'Results/'))

    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--num_samples', type=list, default=[5, 5])

    parser.add_argument('--num_rounds_global', type=int, default=100, help='number of rounds to simulate;')
    parser.add_argument('--num_epoch_local', type=int, default=4, help='number of local epochs;')
    parser.add_argument('--num_epoch_gen', type=int, default=100)

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--delta', type=int, default=20)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--classifier_layer_sizes', type=list, default=[64,32])

    parser.add_argument('--num_pred_node', type=int, default=5)
    parser.add_argument('--hidden_portion', type=float, default=0.5)

    parser.add_argument('--nlayer', type=int, default=3, help='Number of GINconv layers')
    parser.add_argument('--type_init', type=str, default='rw_dg', choices=['rw','dg','rw_dg','ones'], help='the type of positional initialization')
    parser.add_argument('--n_rw', type=int, default=16, help='Size of position encoding (random walk).')
    parser.add_argument('--n_dg', type=int, default=16, help='Size of position encoding (max degree).')
    parser.add_argument('--n_se', type=int, default=32, help='n_rw+n_dg')

    parser.add_argument("--global_sample_rate", type=float, default=0.3)
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--major_label", type=int, default=3)
    parser.add_argument("--major_rate", type=float, default=0.8)
    parser.add_argument("--test_num", type=int, default=200)
    parser.add_argument("--sageMode", type=str, default="GraphSAGE", help="GraphSAGE, GAT")
    parser.add_argument('--h_feats', type=int, default=512, help='hidden features')
    parser.add_argument("--mixup", type=int, default=1)
    parser.add_argument("--linear", action="store_true", help="linear personalization layers")
    parser.add_argument("--server_train_epoch", type=int, default=5)
    parser.add_argument("--lamb_c", type=float, default=0.5)
    parser.add_argument("--lamb_fixed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument('--broadcast_step', type=int, default=2)
    parser.add_argument('--lamuda_aggregation', type=float, default=1e-6)
    
    args = parser.parse_args()
    if args.datasets_name == 'cora':
        args.num_classes = 7
    elif args.datasets_name == 'citeseer':
        args.num_classes = 6
    elif args.datasets_name == 'elliptic':
        args.num_classes = 2
    elif args.datasets_name == 'pubmeddiabetes':
        args.num_classes = 3
    return args


def save_options(args):
    args_dict = args.__dict__
    pth = '{}{}'.format(args.path_save_keypoints, args.test_name)
    if not os.path.exists(pth):
        os.mkdir(pth)

    with open(f'{pth}/log_options.txt', 'a') as f:
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.close()