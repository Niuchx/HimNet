import torch
from torch import nn

def loss_func(adj, A_hat, attrs, X_hat):

    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sum(diff_attribute, -1)
    attribute_cost = torch.mean(attribute_reconstruction_errors, 1)

    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sum(diff_structure, -1)
    structure_cost = torch.mean(structure_reconstruction_errors, 1)
    
    return structure_cost, attribute_cost


def graphembloss(graph_embed, recon_graph_emb):
    
    diff_graph_emb = torch.pow(graph_embed - recon_graph_emb, 2)
    graph_emb_reconstruction_errors = torch.sum(diff_graph_emb, -1)
    graph_emb_cost = graph_emb_reconstruction_errors

    return graph_emb_cost


class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b
