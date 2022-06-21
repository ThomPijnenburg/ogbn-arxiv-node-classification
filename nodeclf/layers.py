import torch
import torch.nn as nn


class GCNLayer(nn.Module):
  def __init__(self, c_in, c_out):
    super().__init__()
    self.projection = nn.Linear(c_in, c_out)


  def forward(self, node_feats, adj_matrix):
    """Inputs"""
    num_neighbours = torch.sparse.sum(adj_matrix, dim=-1).unsqueeze(-1).to_dense()
    node_feats = self.projection(node_feats)
    
    # node_feats = torch.sparse.mm(adj_matrix, node_feats)
    node_feats = torch.mm(adj_matrix, node_feats)
    
    # node_feats = node_feats.to_dense()
    node_feats = torch.div(node_feats, num_neighbours)
    return node_feats