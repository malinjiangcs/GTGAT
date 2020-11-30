#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
class GraphAttentionLayer(nn.Module):
    def __init__(self,feat_dim,hidden_dim,dropout,alpha,concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.concat = concat
        # 压缩数据的参数
        # 注意：这里扩展了隐层的编码向量，前一半作为表示，后一半作为其的置信度向量
        self.W = nn.Parameter(torch.zeros(size=(feat_dim, hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # 增加控制attention的参数
        self.beta = nn.Parameter(torch.zeros(size=[1,1]))
        nn.init.xavier_uniform_(self.beta.data,gain=1.414)
        self.gama = nn.Parameter(torch.zeros(size=[1,1]))
        nn.init.xavier_uniform_(self.gama.data,gain=1.414)
    def forward(self,inputs,adj,tree_attention,controlFlag=False):
        h = torch.mm(inputs,self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.hidden_dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj>0,e,zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        if controlFlag:
            attention = self.beta * attention + self.gama * tree_attention
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.feat_dim) + '->' + str(self.hidden_dim) + ')'