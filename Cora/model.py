#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from gatLayer import GraphAttentionLayer
from treeLayer import RecursiverLayer



class FewGatModel(nn.Module):
    def __init__(self,feat_dim,hidden_dim,dropout,alpha,num_class,weight_dict,concat=True):
        super(FewGatModel, self).__init__()
        self.gat = GraphAttentionLayer(feat_dim,2*hidden_dim,dropout,alpha,concat=True)
        self.tree = RecursiverLayer(2*hidden_dim,2*hidden_dim,num_class,weight_dict,alpha)
        #outgat out_tree
        self.outgat = GraphAttentionLayer(2*hidden_dim,num_class,dropout,alpha,concat=False)
        self.dropout = dropout
    def forward(self,inputs,adj):
        attention_false = torch.zeros(size=[100,100])
        inputs = F.dropout(inputs,self.dropout,training=self.training)
        res_embedding = self.gat(inputs,adj,attention_false,False)
        tree_attention = self.tree(res_embedding,adj)
        all_res2 = self.outgat(res_embedding,adj,tree_attention,True)
        r = F.log_softmax(all_res2,dim=1)
        return r

