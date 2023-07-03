import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import math

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False, dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y
    

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers,
            pred_hidden_dims=[], concat=False, bn=True, dropout=0.0, args=None):
        super(GraphEncoder, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(input_dim, hidden_dim, embedding_dim, num_layers, add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self, normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self, normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList([GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self, normalize_embedding=normalize, dropout=dropout, bias=self.bias) for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self, normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        x = conv_first(x, adj)
        x = self.act(x)#relu
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out = torch.mean(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out = torch.mean(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)
        out = torch.mean(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        return x, output
    
class attr_Decoder(nn.Module):
    def __init__(self, feat_size,hiddendim,outputdim,dropout):
        super(attr_Decoder, self).__init__()

        self.gc1 = nn.Linear(outputdim, hiddendim, bias=False)
        self.gc2 = nn.Linear(hiddendim, feat_size, bias=False)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)    

    def forward(self, x, adj):
        x = self.leaky_relu(self.gc1(torch.matmul(adj, x)))
        x = self.gc2(torch.matmul(adj, x))

        return x    

class stru_Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(stru_Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1=x.permute(0, 2, 1)
        x = torch.matmul(x,x1) 
        x = nn.ReLU()(x)
        return x

class MemModule(nn.Module):
    def __init__(self, mem_num_node, mem_num_graph, node_num, feat_num):
        super(MemModule, self).__init__()
        
        self.weight_node = Parameter(torch.Tensor(mem_num_node, node_num, feat_num))
        self.weight_graph = Parameter(torch.Tensor(mem_num_graph, feat_num))
        self.bias = None
        self.shrink_thres = 0.2 / mem_num_node
        self.shrink_thres_graph = 0.2 / mem_num_graph
        self.reset_parameters_node()
        self.reset_parameters_graph()
        
    def reset_parameters_node(self):
        stdv = 1. / math.sqrt(self.weight_node.size(1))
        self.weight_node.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)   

    def reset_parameters_graph(self):
        stdv = 1. / math.sqrt(self.weight_graph.size(1))
        self.weight_graph.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  
            
    def forward(self, node_embed, graph_embed):
        att_weight_node = torch.einsum('bnd,mnd->bm', [node_embed, self.weight_node])
        att_weight_node = F.softmax(att_weight_node, dim=1)
        if(self.shrink_thres>0):
            att_weight_node = hard_shrink_relu(att_weight_node, lambd=self.shrink_thres)
            att_weight_node = F.normalize(att_weight_node, p=1, dim=1)
        output_node = torch.einsum('bm,mnd->bnd', [att_weight_node, self.weight_node])

        att_weight_graph = torch.einsum('bd,md->bm', [graph_embed, self.weight_graph])
        att_weight_graph = F.softmax(att_weight_graph, dim=1)
        if(self.shrink_thres_graph>0):
            att_weight_graph = hard_shrink_relu(att_weight_graph, lambd=self.shrink_thres_graph)
            att_weight_graph = F.normalize(att_weight_graph, p=1, dim=1)
        output_graph = torch.einsum('bm,md->bd', [att_weight_graph, self.weight_graph])

        return output_node, output_graph, att_weight_node, att_weight_graph

def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class GNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, mem_num_node, mem_num_graph, node_num, dropout=0.1, args=None):
        super(GNNet, self).__init__()
        
        self.encoder = GraphEncoder(input_dim, hidden_dim, embedding_dim, num_layers, bn=args.bn, args=args)
        self.memory = MemModule(mem_num_node, mem_num_graph, node_num, embedding_dim)
        self.feat_dec = attr_Decoder(input_dim, hidden_dim, embedding_dim, dropout)
        self.adj_dec = stru_Decoder(embedding_dim)
        
    def forward(self, x, adj):
        node_embed, graph_embed = self.encoder(x, adj)
        output_node, output_graph, att_weight_node, att_weight_graph = self.memory(node_embed, graph_embed)

        recon_node = self.feat_dec(output_node, adj)
        recon_adj = self.adj_dec(output_node)
        
        return recon_node, recon_adj, att_weight_node, att_weight_graph, graph_embed, output_graph
