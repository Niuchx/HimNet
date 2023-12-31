import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.svm import OneClassSVM
import argparse
import load_data
import networkx as nx
import torch
import torch.nn as nn
import time
from GCN_embedding import *
from torch.autograd import Variable
from graph_sampler import GraphSampler
from numpy.random import seed
import random
import copy
import torch.nn.functional as F
import torch_geometric
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from loss import *

def arg_parse():
    parser = argparse.ArgumentParser(description='HimNet Arguments.')
    parser.add_argument('--datadir', dest='datadir', default ='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default ='AIDS', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0, help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--batch-size', dest='batch_size', default=2000, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=512, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=256, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=3, type=int, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nodemem-num', dest='mem_num_node', default=4, type=int, help='Node Memory blocks')
    parser.add_argument('--graphmem-num', dest='mem_num_graph', default=3, type=int, help='Graph Memory blocks')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=False, help='Whether to add bias. Default to True.')
    parser.add_argument('--lr', dest='lr', default= 0.01, type=float, help='Learning Rate')
    parser.add_argument('--epoch', dest='epoch', default=100, type=int, help='total epoch number')
    parser.add_argument('--feature', dest='feature', default='deg-num', help='use what node feature')
    parser.add_argument('--alpha', dest='alpha', default= 0.01, type=float, help='weight parameter')
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='seed')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)

def train(dataset, data_test_loader, model, k, args):
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    model.train()
    tr_entropy_loss_func = EntropyLoss()
    auroc_final = []

    for epoch in range(args.epoch):
        loss_epoch = 0
        num_train = 0
        
        for batch_idx, data in enumerate(dataset):
            optimizer.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            adj_label = Variable(data['adj_label'].float(), requires_grad=False).cuda()

            recon_node, recon_adj, att_node, att_graph, graph_embed, recon_graph_embed = model(h0, adj)

            loss_recon_adj, loss_recon_node = loss_func(adj_label, recon_adj, h0, recon_node)
            entropy_loss_node = tr_entropy_loss_func(att_node)
            entropy_loss_graph = tr_entropy_loss_func(att_graph)
            graph_embed_loss = graphembloss(graph_embed, recon_graph_embed)

            loss = loss_recon_adj.mean() + loss_recon_node.mean() + graph_embed_loss.mean() + args.alpha*entropy_loss_node + args.alpha*entropy_loss_graph
            
            loss_epoch += loss.item() * adj.shape[0]
            num_train += adj.shape[0]
            loss.backward()
            optimizer.step()
            
        print("Epoch: %d Train AE Loss: %f" % (epoch+1, loss_epoch / num_train))
        
        if (epoch+1)%args.epoch == 0:
            model.eval()   
            loss = []
            y=[]
            for batch_idx, data in enumerate(data_test_loader):
               adj = Variable(data['adj'].float(), requires_grad=False).cuda()
               h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
               adj_label = Variable(data['adj_label'].float(), requires_grad=False).cuda()

               recon_node, recon_adj, _, _, graph_embed, recon_graph_embed = model(h0, adj)

               loss_recon_adj, loss_recon_node = loss_func(adj_label, recon_adj, h0, recon_node)
               graph_embed_loss = graphembloss(graph_embed, recon_graph_embed)
               lossall = loss_recon_adj + loss_recon_node + graph_embed_loss
            
               loss_ = lossall
               loss_ = np.array(loss_.cpu().detach())
               loss.append(loss_)
               y.append(data['label'].cpu().numpy())

            label_test = []
            for loss_ in loss:
               label_test.append(loss_)
            label_test = np.array(label_test)                  
            fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)  
            auroc_final.append(test_roc_ab) 
            print('Epoch: {} Abnormal Detection: auroc_ab: {}'.format(epoch+1, test_roc_ab))
        if epoch == (args.epoch-1):
            auroc_final =  test_roc_ab
    return auroc_final

    
if __name__ == '__main__':
    args = arg_parse()
    DS = args.DS

    graphs_train_ = load_data.read_graphfile(args.datadir, 'Tox21_' + args.DS+'_training', max_nodes=args.max_nodes)  
    graphs_test = load_data.read_graphfile(args.datadir, 'Tox21_' + args.DS+'_testing', max_nodes=args.max_nodes)  
    datanum = len(graphs_train_) + len(graphs_test)    
  
    if args.max_nodes == 0:
        max_nodes_num_train = max([G.number_of_nodes() for G in graphs_train_])
        max_nodes_num_test = max([G.number_of_nodes() for G in graphs_test])
        max_nodes_num = max([max_nodes_num_train, max_nodes_num_test])
    else:
        max_nodes_num = args.max_nodes
        
    print(datanum, max_nodes_num)

    graphs_train = []
    for graph in graphs_train_:
        if graph.graph['label'] == 1:
            graphs_train.append(graph)
    for graph in graphs_train:
        graph.graph['label'] = 0
            
    graphs_test_nor = []
    graphs_test_ab = []
    for graph in graphs_test:
        if graph.graph['label'] == 0:
            graphs_test_nor.append(graph)
        else:
            graphs_test_ab.append(graph)
    for graph in graphs_test_nor:
        graph.graph['label'] = 0
    for graph in graphs_test_ab:
        graph.graph['label'] = 1
        graphs_test_nor.append(graph)
    graphs_test = graphs_test_nor
                
    num_train = len(graphs_train)
    num_test = len(graphs_test)
    print(num_train, num_test)

    dataset_sampler_train = GraphSampler(graphs_train, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)    
    data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, shuffle=True, batch_size=args.batch_size)

    dataset_sampler_test = GraphSampler(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
    data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, shuffle=False, batch_size=1)
    result_auc = []
    for i in range(5):
        setup_seed(i)
        model = GNNet(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, args.num_gc_layers, args.mem_num_node, args.mem_num_graph, max_nodes_num, args=args).cuda()
        results = train(data_train_loader, data_test_loader, model, i, args)
        result_auc.append(results)
            
    result_auc = np.array(result_auc)    
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)

    print(' auroc {}, average: {}, std: {}'.format(result_auc, auc_avg, auc_std))
    
    
