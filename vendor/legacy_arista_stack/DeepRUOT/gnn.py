import torch
import torch.nn as nn
import math
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple
import pickle
import os
#from torch_scatter import scatter

class GNNInteraction(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, num_heads, num_layers, activation='Tanh', num_rbf = 8, cutoff = 0.2, use_spatial = True, edge_predictor_path = None, edge_predictor_thre = 0.5):
        super().__init__()
        if activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_spatial = use_spatial
        self.cutoff = cutoff
        self.rbf_expansion = ExpNormalSmearing(cutoff=cutoff, num_rbf = num_rbf)
        self.edge_path = os.path.join(os.path.dirname(__file__), '..','edge_classifier')
        # self.load_metadata(self.edge_path)
        self.link_predictor = LinkPredictorMLP(input_dim=in_out_dim * 2,)
        self.link_predictor.load_state_dict(torch.load(os.path.join(self.edge_path, edge_predictor_path),map_location=torch.device('cpu')))
        #self.link_predictor.load_state_dict(torch.load(os.path.join(self.edge_path, 'heart.pt'),map_location=torch.device('cpu')))
        for param in self.link_predictor.parameters():
            param.requires_grad = False
        self.link_predictor.eval()
        self.gene_embed = nn.Sequential(
            nn.Linear(in_out_dim - 2, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_predictor_thre = edge_predictor_thre

        # self.edge_embedding = nn.Embedding(len(self.final_pathway_list), hidden_dim)
        
        self.distance_projection = nn.Linear(num_rbf, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.gnn_layers.append(GraphAttentionLayer(hidden_dim, num_heads, activation=activation))
        
        #self.spatial_readout = nn.Linear(hidden_dim, hidden_dim)
        
        # self.spatial_readout = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     self.activation,
        #     nn.Linear(hidden_dim, 2),
        # )

        self.gene_readout = nn.Sequential(
            nn.Linear(hidden_dim, in_out_dim - 2),
        )

    def forward(self, x, lnw, t, return_attn = False):
        if self.use_spatial:
            x_expanded = x[:, :2].unsqueeze(1)  # Shape: (batch_size, 1, dim)
            x_expanded_t = x[:, :2].unsqueeze(0)  # Shape: (1, batch_size, dim)
            pairwise_distances = torch.norm(x_expanded - x_expanded_t, dim=2)  # Shape: (batch_size, batch_size)
            # Create mask based on cutoff distance
            mask = (pairwise_distances < self.cutoff).float()
            spatial_coord = x[:, :2]
        else:
            # Calculate pairwise distances between all points
            x_expanded = x.unsqueeze(1)  # Shape: (batch_size, 1, dim)
            x_expanded_t = x.unsqueeze(0)  # Shape: (1, batch_size, dim)
            pairwise_distances = torch.norm(x_expanded - x_expanded_t, dim=2)  # Shape: (batch_size, batch_size)
            # Create mask based on cutoff distance
            mask = (pairwise_distances < self.cutoff).float()
            spatial_coord = None
        
        # Predict edges
        rows, cols = torch.where(mask)
        features_i = x[rows]
        features_j = x[cols]
        pair_features = torch.cat([features_i, features_j], dim=1)
        pred_probs = self.link_predictor(pair_features)
        pred_probs = torch.sigmoid(pred_probs)
        pred_probs = pred_probs.reshape(-1)
        connected = (pred_probs >= self.edge_predictor_thre) 

        # edge_index_list = []
        # edge_label_list = []
        # for i in range(len(self.final_pathway_list)):
        #     pathway_name = self.final_pathway_list[i]
        #     threshold = self.best_thresholds[pathway_name]
        #     connected = (pred_probs[:, i] >= 0.97)
        #     edge_index_list.append(torch.stack([rows[connected], cols[connected]], dim=0))
        #     edge_label_list.append(torch.ones(connected.sum(), device = x.device) * i)
        # edge_index = torch.cat(edge_index_list, dim=1)
        # edge_label = torch.cat(edge_label_list, dim=0).long()
        edge_index = torch.stack([rows[connected], cols[connected]], dim=0)
        # edge_index = torch.stack([rows[connected], cols[connected]], dim=0) #[connected]``

        # Exclude self loop
        indices = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, indices]
        r_ij = pairwise_distances[edge_index[0], edge_index[1]]
        edge_index = edge_index[:, r_ij > 1e-6]
        r_ij = r_ij[r_ij > 1e-6]
        # edge_label = edge_label[indices]
        self.edge_index = edge_index
        del mask

        # Embed gene features
        num = x.shape[0]
        t = t.expand(num, 1)  # 保持 t 的梯度信息并扩展其形状
        # x_embed = self.gene_embed(torch.cat([t, x[:, 2:]], dim=1))
        x_embed= self.gene_embed(x[:, 2:])
        vec = torch.zeros(x_embed.size(0), 2, x_embed.size(1), device=x.device)


        # Calculate edge features
        
        vec_ij = (x[edge_index[0], :2] - x[edge_index[1], :2]) / r_ij.unsqueeze(1) # 0 to 1
        rbf_ij = self.rbf_expansion(r_ij)
        # edge_embedding = self.edge_embedding(edge_label)
        edge_attr = (x_embed[edge_index[0]] + x_embed[edge_index[1]]) * self.distance_projection(rbf_ij) # + edge_embedding
        

        for i in range(self.num_layers):
            x_embed, vec = self.gnn_layers[i](x_embed, vec, lnw, edge_index, edge_attr, vec_ij, return_attn = return_attn)
        

        x_spatial = vec.mean(dim=-1)

        # x_spatial = self.spatial_readout(x_embed) #(self.spatial_readout(x_embed).unsqueeze(1) * vec).sum(dim=-1) #vec.mean(dim=-1) #
        x_gene = self.gene_readout(x_embed)

        x_out = torch.cat([x_spatial, x_gene], dim=1)


        return x_out
    
    def load_metadata(self, edge_path):
        with open(os.path.join(edge_path, 'prediction_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            self.pathway_to_idx = metadata['pathway_to_idx']
            self.final_pathway_list = metadata['final_pathway_list']
            self.num_features = metadata['num_features']
            self.best_thresholds = metadata['best_thresholds']

class GraphAttentionLayer(MessagePassing):
    def __init__(self, hidden_dim, num_heads, activation='Tanh'):
        super(GraphAttentionLayer, self).__init__(node_dim=0)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        self.attn_activation = nn.SiLU()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dk_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dv_proj = nn.Linear(hidden_dim, hidden_dim)
        self.s_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.res_proj = nn.Linear(hidden_dim, hidden_dim)
        self.vec_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim//num_heads, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim//num_heads),
        )
    
    def forward(self, x, vec, lnw, edge_index, edge_attr, edge_vec, return_attn = False):
        x_res = self.res_proj(x).reshape(-1, self.num_heads, self.head_dim)
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.dk_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)
        dv = self.dv_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)
        w = torch.exp(lnw) * lnw.shape[0]

        self.return_attn = return_attn
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            w=w,
            x_orig=x_res,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=edge_attr,
            d_ij=edge_vec,
            size=None,
        )

        # vec1, vec2, vec3 = torch.split(self.vec_proj(vec_out), self.hidden_dim, dim=-1)
        # vec_dot = (vec1 * vec2).sum(dim=1)
        # o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_dim, dim=1)
        # x = vec_dot * o2 + o3
        # vec = vec3 * o1.unsqueeze(1) + vec_out

        
        return x, vec_out
        

    def message(self, q_i, k_j, v_j, vec_j, w_j, x_orig_i, dk, dv, r_ij, d_ij):

        attn = (q_i * k_j * dk).sum(dim=-1) #(num_edges, num_heads)
        attn = self.attn_activation(attn)
        #attn = torch.exp(attn)
        if self.return_attn:
            self.attn = attn
        v_j = v_j * dv + x_orig_i
        v_j = self.out_transform(v_j)
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_dim)

        s1, s2 = torch.split(self.activation(self.s_proj(v_j)), self.hidden_dim, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        v_j = v_j * w_j
        vec_j = vec_j * w_j.unsqueeze(1)
        return v_j, vec_j, w_j

    def manual_scatter_mean(self, src: torch.Tensor, weight: torch.Tensor, index: torch.Tensor, dim: int, output_size: int) -> torch.Tensor:
        """
        一个手动实现的、绝对可靠的 scatter_mean 函数。
        
        Args:
            src: 源张量，可以是任意维度 (e.g., 2D or 3D)。
            index: 1D的索引张量。
            dim: 进行聚合的维度 (对于GNN通常是 0)。
            output_size: 输出张量在聚合维度上的大小 (通常是节点总数)。
        """
        # a. 准备输出形状和扩展后的index
        #    这是解决“维度不匹配”错误的关键步骤
        output_shape = list(src.shape)
        output_shape[dim] = output_size
        
        # 扩展 index 来匹配 src 的维度
        # e.g., src是 [E, F], index是 [E] -> index_expand是 [E, F]
        # e.g., src是 [E, 2, F], index是 [E] -> index_expand是 [E, 1, 1] -> [E, 2, F]
        index_expand_shape = [1] * src.dim()
        index_expand_shape[dim] = -1 # -1表示保持这个维度的大小不变
        index_expanded = index.view(index_expand_shape).expand_as(src)

        # b. 计算 scatter_sum
        #    先创建一个全零的输出张量
        output_sum = torch.zeros(output_shape, dtype=src.dtype, device=src.device)
        #    使用 scatter_add_ (in-place) 来计算和
        output_sum.scatter_add_(dim, index_expanded, src)

        # c. 计算每个位置的计数
        count = torch.zeros(output_shape, dtype=src.dtype, device=src.device)
        # weight 现在的形状是 [num_edges]，需要扩展到和 src 一样的形状
        # 例如 src 是 [E, F]，weight 是 [E]，weight_expanded 需要是 [E, F]
        # 例如 src 是 [E, 2, F]，weight 是 [E]，weight_expanded 需要是 [E, 2, F]
        weight_expand_shape = [ -1 ] + [1] * (src.dim() - 1)
        weight_expanded = weight.view(weight_expand_shape).expand_as(src)
        count.scatter_add_(dim, index_expanded, weight_expanded)

        # d. 计算均值，并用 clamp(min=1) 防止除以零
        count_safe = count.clone()
        count_safe[count_safe == 0] = 1
        return output_sum / count_safe

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        最终的 aggregate 函数, 调用手动的 scatter_mean。
        """
        x, vec, w = features
        aggregation_dim = self.node_dim # 通常是 0

        # 对 x (2D) 和 vec (3D) 调用同一个可靠的函数
        aggregated_x = self.manual_scatter_mean(x, w, index, dim=aggregation_dim, output_size=dim_size)
        aggregated_vec = self.manual_scatter_mean(vec, w, index, dim=aggregation_dim, output_size=dim_size)
        return aggregated_x, aggregated_vec
        # x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        # vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        # return x, vec
        

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
    
class LinkPredictorMLP_pathway(nn.Module):
    """ 可动态配置层数的通用MLP模型 """
    def __init__(self, input_dim, output_dim, hidden_dims, activation_fn=nn.ReLU, use_batchnorm=True, dropout_rate=0.4):
        super(LinkPredictorMLP, self).__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if use_batchnorm: layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

class LinkPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(LinkPredictorMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x)

class CosineCutoff(nn.Module):
    
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, cutoff_sr = 0.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.cutoff_sr = cutoff_sr
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = 1.0
        self.cutoff_fn = CosineCutoff(cutoff)
        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        if self.cutoff == 0:
            return torch.zeros(self.num_rbf), torch.tensor(0.0)
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        end_value = torch.exp(torch.scalar_tensor(-self.cutoff_sr))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)
        
