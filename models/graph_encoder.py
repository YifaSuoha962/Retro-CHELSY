import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from utils.data_loader import graph_to_seq, seq_to_graph


class DMPNN_layer(nn.Module):
    '''DMPNN layer, the aggregate direction is k->i->j'''

    def __init__(
        self,
        f_node: int,
        f_bond: int,
        d_model: int,
        **kwargs
    ):
        super(DMPNN_layer, self).__init__(**kwargs)
        self.W_z = nn.Linear(f_node + f_bond + d_model, d_model)
        self.W_r = nn.Linear(f_node + f_bond + d_model, d_model)
        self.U = nn.Linear(d_model, d_model, bias=False)
        self.W = nn.Linear(f_node + f_bond, d_model)

    def forward(
        self,
        h_ij: torch.Tensor,  # bond (i->j) initial feature
        h_ki: torch.Tensor,  # bond (k->i) initial feature
        mess: torch.Tensor,  # the previous layer message
        src_idx: torch.Tensor,  # bond(i->j) index
        nei_idx: torch.Tensor   # bond(k->i) index
    ):
        bond_num = h_ij.size(0)
        mess_ki = mess.index_select(0, nei_idx)
        s_ij = scatter(mess_ki, src_idx, dim=0, dim_size=bond_num, reduce='sum')
        z_ij = torch.sigmoid(self.W_z(torch.cat([h_ij, s_ij], dim=-1)))

        r_ki = torch.sigmoid(self.W_r(torch.cat([h_ki, mess_ki], dim=-1)))
        r_ij = scatter(r_ki * mess_ki, src_idx, dim=0, dim_size=bond_num, reduce='sum')

        m_ij = torch.tanh(self.W(h_ij) + self.U(r_ij))
        mess = (1 - z_ij) * s_ij + z_ij * m_ij

        return mess


class DMPNN_Encoder(nn.Module):
    def __init__(
        self,
        f_node: int,
        f_bond: int,
        d_model: int,
        layer: int,
        dropout=0.1,
        **kwargs
    ):
        super(DMPNN_Encoder, self).__init__(**kwargs)
        self.layer = layer
        self.d_model = d_model
        self.mpnn = DMPNN_layer(f_node, f_bond, d_model)
        self.W_in = nn.Sequential(nn.Linear(f_node + f_bond, d_model), nn.GELU())
        self.W_on = nn.Sequential(nn.Linear(f_node + d_model, d_model), nn.GELU())
        self.W_oe = nn.Sequential(nn.Linear(f_bond + d_model, d_model), nn.GELU())
        self.dropout = nn.Dropout(dropout)

        self._init_param()

    def _init_param(self):
        for param in self.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_normal_(param)

    def forward(
        self,
        node: torch.Tensor,
        connect: torch.Tensor,
        bond: torch.Tensor,
        bond_neighbour: torch.Tensor
    ):
        src, tgt = connect[0].to(torch.long), connect[1].to(torch.long)
        # indices of neighboring bonds
        src_idx, nei_idx = bond_neighbour[0].to(torch.long),\
            bond_neighbour[1].to(torch.long)
        f_node, f_bond = node.clone(), bond.clone()
        mess = self.W_in(torch.cat([f_node.index_select(0, src), f_bond], dim=-1))
        f_ij = torch.cat([f_node.index_select(0, src), f_bond], dim=-1)
        f_ki = f_ij.index_select(0, src_idx)

        for i in range(self.layer):
            mess = self.mpnn(
                h_ij=f_ij,
                h_ki=f_ki,
                mess=mess,
                src_idx=src_idx,
                nei_idx=nei_idx
            )
            if i == 0:
                mess = self.dropout(mess)

        f_bond = self.W_oe(torch.cat([f_bond, mess], dim=-1))
        mess_boost = scatter(mess, tgt, dim=0, dim_size=node.size(0), reduce='max')
        mess = scatter(mess, tgt, dim=0, dim_size=node.size(0), reduce='sum')  # m_ij -> m_j
        mess = mess * mess_boost
        f_node = self.W_on(torch.cat([f_node, mess], dim=-1))
        return f_node, f_bond


class CMPNN_GRU(nn.Module):
    def __init__(
        self,
        d_model
    ):  
        super(CMPNN_GRU, self).__init__()
        self.d_model = d_model
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.bias = nn.Parameter(torch.Tensor(self.d_model),
                                 requires_grad=True)
        with torch.no_grad():
            self.bias.normal_(0., 1 / math.sqrt(self.d_model))
    
    def forward(
        self,
        node: torch.Tensor,
        node_num: torch.Tensor
    ):
        mess = F.hardswish(node + self.bias)
        gate_mess = graph_to_seq(
            node_feat=mess,
            node_num=node_num
        )
        batch_size = len(node_num)
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=node_num.device).\
            repeat_interleave(node_num, dim=0)
        hidden = scatter(node, batch_idx, dim=0, dim_size=batch_size, reduce='max')

        hidden = hidden.unsqueeze(0).repeat(2, 1, 1)
        gate_mess, hidden = self.gru(gate_mess, hidden)
        gate_mess = seq_to_graph(
            node_feat=gate_mess,
            node_num=node_num
        )
        return gate_mess


class CMPNN_Encoder(nn.Module):
    def __init__(
        self,
        f_node: int,
        f_bond: int,
        d_model: int,
        layer: int,
        head = 8,
        dropout=0.1,
        **kwargs
    ):
        super(CMPNN_Encoder, self).__init__(**kwargs)
        self.layer = layer
        self.head = head
        self.d_model = d_model
        self.d_head = int(d_model // head)
        self.W_node = nn.Linear(f_node, d_model)
        self.W_node_final = nn.Linear(f_node + d_model, d_model)
        self.W_bond = nn.Linear(f_bond + f_node, d_model)
        self.W_bond_final = nn.Linear(f_bond + f_node + d_model, d_model)
        self.layer_activation = nn.Hardswish()
        self.node_final_dropout = nn.Dropout(dropout)
        self.bond_final_dropout = nn.Dropout(dropout)

        self.W_z = nn.Linear(f_node + f_bond + d_model, d_model)
        self.W_r = nn.Linear(f_node + f_bond + d_model, d_model)
        self.U = nn.Linear(d_model, d_model, bias=False)
        self.W = nn.Linear(f_node + f_bond, d_model)

        self.W_zn = nn.Linear(f_node + d_model * 2, d_model)
        self.W_rn = nn.Linear(f_node + d_model * 2, d_model)
        self.U_n = nn.Linear(d_model * 2, d_model, bias=False)
        self.W_n = nn.Linear(f_node, d_model)

        self._init_param()
    
    def _init_param(self):
        for param in self.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_normal_(param)
    
    def forward(
        self,
        node: torch.Tensor,
        connect: torch.Tensor,  # [i, j] i->j
        bond: torch.Tensor,
        bond_neighbour: torch.Tensor  # [i->j, k->i]
    ):
        i_idx, j_idx, reverse = connect[0].to(torch.long), connect[1].to(torch.long),\
            connect[2].to(torch.long)
        ij_idx, ki_idx = bond_neighbour[0].to(torch.long), bond_neighbour[1].to(torch.long)
        
        node_size, bond_size = node.size(0), bond.size(0)
        mess_bond = torch.cat([node.index_select(0, i_idx), bond], dim=-1)
        mess_node = node.clone()
        init_bond, init_node = mess_bond.clone(), mess_node.clone()
        mess_bond = self.layer_activation(self.W_bond(mess_bond))
        mess_node = self.layer_activation(self.W_node(mess_node))

        for i in range(self.layer):
            # bond message update
            mess_ki = mess_bond.index_select(0, ki_idx)
            s_ij = scatter(mess_ki, ij_idx, dim=0, dim_size=bond_size, reduce='sum')
            z_ij = torch.sigmoid(self.W_z(torch.cat([init_bond, s_ij], dim=-1)))

            r_ki = torch.sigmoid(self.W_r(torch.cat([init_bond.index_select(0, ij_idx), mess_ki], dim=-1)))
            r_ij = scatter(r_ki * mess_ki, ij_idx, dim=0, dim_size=bond_size, reduce='sum')

            m_ij = torch.tanh(self.W(init_bond) + self.U(r_ij))
            mess_bond = (1 - z_ij) * s_ij + z_ij * m_ij

            # node message update
            aggr_node = scatter(mess_bond, j_idx, dim=0, dim_size=node_size, reduce='sum')
            mess_node = self.W_n(init_node) + self.U_n(torch.cat([mess_node, aggr_node], dim=-1))
            mess_node = self.layer_activation(mess_node)

        mess_bond = torch.cat([init_bond, mess_bond], dim=-1)
        mess_bond = self.layer_activation(self.W_bond_final(mess_bond))
        mess_node = torch.cat([init_node, mess_node], dim=-1)
        mess_node = self.layer_activation(self.W_node_final(mess_node))
        mess_node, mess_bond = self.node_final_dropout(mess_node),\
            self.bond_final_dropout(mess_bond)

        return mess_node, mess_bond


class Graph_Readout(nn.Module):
    '''the Set2Set Readout function, code is from pyg'''

    def __init__(
        self,
        d_model: int,
        layer=1,
        step=4,
        **kwargs
    ):
        super(Graph_Readout, self).__init__(**kwargs)
        self.d_model = d_model
        self.lstm = nn.LSTM(d_model * 2, d_model, layer, batch_first=True)
        self.W_o = nn.Linear(d_model * 2, d_model)
        self.step = step

        self._init_param()

    def _init_param(self):
        for param in self.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_normal_(param)

    def forward(self, node: torch.Tensor, node_num: torch.Tensor):
        batch_size = node_num.size(0)
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=node.device).\
            repeat_interleave(node_num, dim=0)

        h = None
        q_star = torch.zeros(batch_size, self.d_model * 2,
                             dtype=torch.float, device=node.device)
        for _ in range(self.step):
            q, h = self.lstm(q_star.unsqueeze(1), h)
            q = q.squeeze(1)
            e = (node * q[batch_idx]).sum(dim=-1, keepdim=True)
            e = (e - scatter(e, batch_idx, 0, dim_size=node.size(0), reduce='max')
                .index_select(dim=0, index=batch_idx)).exp()
            a = e / (scatter(e, batch_idx, 0, dim_size=node.size(0), reduce='sum')
                .index_select(dim=0, index=batch_idx) + 1e-6)
            r = scatter(a * node, batch_idx, dim=0, reduce='sum')
            q_star = torch.cat([q, r], dim=-1)

        return self.W_o(q_star)
        

