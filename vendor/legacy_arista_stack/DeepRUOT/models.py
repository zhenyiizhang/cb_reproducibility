import torch, torch.nn as nn
import math
import joblib


import torch.nn.init as init
from DeepRUOT.interaction import cal_interaction
from DeepRUOT.gnn import GNNInteraction


class velocityNet(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh', use_spatial = False):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        
        self.use_spatial = use_spatial

        if use_spatial:
            # Spatial velocity
            self.spatial_net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
            )
            self.spatial_out = nn.Linear(Layers[-2], 2)

            # Gene velocity
            self.gene_net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
            )
            self.gene_out = nn.Linear(Layers[-2], in_out_dim - 2)
        else:
            self.net = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(Layers[i], Layers[i + 1]),
                    self.activation,
                )
                    for i in range(len(Layers) - 2)
                ]
            )
            self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        #print(num)
        t = t.expand(num, 1)  # 保持 t 的梯度信息并扩展其形状
        #print(t)
        state  = torch.cat((t,x),dim=1)
        #print(state)
        if self.use_spatial:
            ii = 0
            for layer in self.spatial_net:
                if ii == 0:
                    x = layer(state)
                else:
                    x = layer(x)
                ii =ii+1
            spatial_x = self.spatial_out(x)

            ii = 0
            for layer in self.gene_net:
                if ii == 0:
                    x = layer(state)
                else:
                    x = layer(x)
                ii =ii+1
            gene_x = self.gene_out(x)
            x = torch.cat([spatial_x, gene_x], dim = 1)
        else:
            ii = 0
            for layer in self.net:
                if ii == 0:
                    x = layer(state)
                else:
                    x = layer(x)
                ii =ii+1
            x = self.out(x)
        return x

class growthNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  # 保持 t 的梯度信息并扩展其形状
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
        #return torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

class scoreNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  # 保持 t 的梯度信息并扩展其形状
        state  = torch.cat((t,x),dim=1)
        return self.net(state)


class dediffusionNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  # 保持 t 的梯度信息并扩展其形状
        state  = torch.cat((t,x),dim=1)
        return self.net(state)

class indediffusionNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  # 保持 t 的梯度信息并扩展其形状
        #state  = torch.cat((t,x),dim=1)
        return self.net(t)

class InteractionModel_vanilla(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
        
    def forward(self, x):
        zero_input = torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)
        baseline = self.net(zero_input)
        potential = self.net(x) + self.net(-x) - 2 * baseline
        return potential

class InteractionModel(nn.Module):
    def __init__(self, x_dim, hidden_dim, activation='Tanh', num_rbf=16, cutoff=1, dim_reduce = False):
        super().__init__()
        self.num_rbf = num_rbf
        if activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation type: {}".format(activation))
        self.rbf_expansion = ExpNormalSmearing(cutoff=cutoff, num_rbf = self.num_rbf, trainable= True)
        self.net = nn.Sequential(
            nn.Linear(self.num_rbf, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1)
        )
        self.cutoff = cutoff
        self.eps = 1e-6
        self.dim_reduce = dim_reduce
        if self.dim_reduce:
            self.pca = nn.Linear(x_dim, 10, bias=False)
            #pca_sklearn = joblib.load('./pca_weinreb.pkl')
            #self.pca.weight = nn.Parameter(torch.tensor(pca_sklearn.components_, dtype=torch.float32))

        #self.pca.weight.copy_(torch.tensor(pca_sklearn.components_))
        #self.pca = self.pca = joblib.load('./pca_weinreb.pkl')
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


    def forward(self, x_t):
        # zero_input = torch.zeros(1, x_t.shape[-1], device=x_t.device, dtype=x_t.dtype)
        # baseline = self.net(zero_input)
        # potential = self.net(x_t) + self.net(-x_t) - 2*baseline

        # dis = torch.norm(x_t, dim = -1)
        # potential = self.net(dis[dis!=0].unsqueeze(1))

        if self.cutoff == 0:
            return 0 * x_t.sum()
        if self.dim_reduce:
            x_t = self.pca(x_t)
        dis = self.compute_distance(x_t)
        dis_exp = self.rbf_expansion(dis[dis != 0])
        potential = self.net(dis_exp)
        return potential
    
    def compute_distance(self, x):
        return torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + self.eps)
    

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
    
class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, use_spatial = False):

        super(SelfAttention, self).__init__()
        assert hidden_dim % n_heads == 0
        
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.d_k = hidden_dim // n_heads  

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.use_spatial = use_spatial

        if use_spatial:
            self.num_rbf = 8
            self.rbf_expansion = ExpNormalSmearing(cutoff=0.5, num_rbf = self.num_rbf, trainable= True)
            self.edge_embed = nn.Sequential(
                nn.Linear(self.num_rbf, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, self.n_heads)
            )

    def forward(self, x1, x2, x3, lnw, mask_dist, spatial_coord = None):

        batch_size, _ = x1.size()

        q = self.q_proj(x1).view(batch_size, self.n_heads, self.d_k).permute(1, 0, 2)  
        k = self.k_proj(x2).view(batch_size, self.n_heads, self.d_k).permute(1, 0, 2)  
        v = self.v_proj(x3).view(batch_size, self.n_heads, self.d_k).permute(1, 0, 2)  

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (n_heads, batch_size, batch_size)
        mask = torch.eye(attn.size(-1), device=attn.device).bool()  
        mask = mask.unsqueeze(0).expand(attn.size(0), -1, -1)  
        attn = attn.masked_fill(mask, float('-inf'))
        mask_dist = mask_dist.unsqueeze(0).expand(attn.size(0), -1, -1)
        attn = attn.masked_fill(mask_dist == 0, float('-inf'))
        attn = attn + lnw.reshape(-1).unsqueeze(0).unsqueeze(1)
        # if self.use_spatial:
        #     # Calculate pairwise distances between spatial coordinates
        #     spatial_expanded = spatial_coord.unsqueeze(1)  # Shape: (batch_size, 1, 2)
        #     spatial_expanded_t = spatial_coord.unsqueeze(0)  # Shape: (1, batch_size, 2)
        #     diff = spatial_expanded - spatial_expanded_t  # Shape: (batch_size, batch_size, 2)
        #     spatial_distances = torch.sqrt(torch.sum(diff ** 2, dim=2, keepdim=True) + 1e-8)  # Shape: (batch_size, batch_size, 1)
            
        #     # Reshape distances for RBF expansion
        #     spatial_distances = spatial_distances.view(-1, 1)  # Shape: (batch_size * batch_size, 1)
            
        #     # Apply RBF expansion to distances
        #     rbf_features = self.rbf_expansion(spatial_distances)  # Shape: (batch_size * batch_size, num_rbf)
            
        #     # Edge embedding
        #     edge_weights = self.edge_embed(rbf_features)  # Shape: (batch_size * batch_size, n_heads)

        #     # Reshape back to batch_size x batch_size
        #     edge_weights = edge_weights.view(batch_size, batch_size, -1)  # Shape: (batch_size, batch_size, num_heads)
            
        #     # Reshape to match attention shape
        #     edge_weights = edge_weights.permute(2, 0, 1)  # Shape: (n_heads, batch_size, batch_size)
            
        #     # Add to attention scores
        #     attn = attn + edge_weights
        attn = torch.exp(attn)
        print(attn.sum(dim=-1, keepdim=True).shape)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        #attn = torch.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)  #（n_heads, batch_size, hidden_dim//n_heads)
        output = output.permute(1, 0, 2).reshape(batch_size, self.hidden_dim) 

        return output, attn, q, k, v
    


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim,
                 num_heads,
                 use_spatial = False,
                 cutoff = 100,
                 edge_threshold = 0.7):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(dim, num_heads, hidden_dim, use_spatial = use_spatial)
        self.use_spatial = use_spatial
        self.cutoff = cutoff
        self.link_predictor = LinkPredictorMLP(dim * 2)
        self.link_predictor.load_state_dict(torch.load('/lustre/home/2100011778/CellSync/classifier/spatial_injury_model.pth'))
        for param in self.link_predictor.parameters():
            param.requires_grad = False
        self.edge_threshold = edge_threshold
        if use_spatial:
            self.embed = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            self.spatial_readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 2)
            )
            self.gene_readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, dim - 2)
            )
        else:
            self.embed = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, dim)
            )

        #self.use_FFN = use_FFN
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, lnw):
        if self.use_spatial:
            # Calculate pairwise distances between all points
            x_expanded = x[:, :2].unsqueeze(1)  # Shape: (batch_size, 1, dim)
            x_expanded_t = x[:, :2].unsqueeze(0)  # Shape: (1, batch_size, dim)
            pairwise_distances = torch.norm(x_expanded - x_expanded_t, dim=2)  # Shape: (batch_size, batch_size)
            # Create mask based on cutoff distance
            mask = (pairwise_distances < self.cutoff).float()
            spatial_coord = x[:, :2]
            #x = x[:, 2:]

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
        not_connected = (pred_probs < self.edge_threshold)
        mask[cols[not_connected], rows[not_connected]] = 0 # col represents receptor, row represents ligand
        
        # Calculate attention score
        x_embed = self.embed(x)
        #x_embed = self.norm1(x_embed)
        x, weights, q, k, v = self.attn(x_embed, x_embed, x_embed, lnw, mask, spatial_coord)
        x = x + x_embed
        #x = self.norm2(x)
        if self.use_spatial:
            x_spatial = self.spatial_readout(x)
            x_gene = self.gene_readout(x)
            x = torch.cat([x_spatial, x_gene], dim = 1)
        else:
            x = self.readout(x)

        return x, weights, q, k, v
  

class InteractionTransformer(torch.nn.Module):
    def __init__(self, in_out_dim, hidden_dim, num_heads, num_layers, cutoff = 100, use_spatial = False):
        super(InteractionTransformer, self).__init__()
        self.cutoff = cutoff
        self.net = nn.ModuleList(
                            [TransformerBlock(dim=in_out_dim, num_heads=num_heads, hidden_dim=hidden_dim, cutoff = cutoff, use_spatial = use_spatial) 
                            for _ in range(num_layers)]
                            )

    def forward(self, x, lnw, return_attn = False):

        for layer in self.net:
            output = layer(x, lnw)
            x = output[0]
            attn = output[1]
        if return_attn:
            return x, attn
        else:
            del attn
            return x

class LinkPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(LinkPredictorMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x)
    

class FNet_interaction(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation, num_rbf = 8, thre = 100, dim_reduce = False, use_spatial = False, num_heads = 8, num_layers = 1, edge_predictor_path = None, edge_predictor_thre = 0.5):
        super(FNet_interaction, self).__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.v_net = velocityNet(in_out_dim, hidden_dim, n_hiddens, activation, use_spatial = use_spatial)  # v = dx/dt
        self.g_net = growthNet(in_out_dim, hidden_dim, activation)  # g
        self.s_net = scoreNet(in_out_dim, hidden_dim, activation)  # s = log rho
        self.d_net = indediffusionNet(in_out_dim, hidden_dim, activation)  # d = sigma(t)
        # self.interaction_net = InteractionModel(in_out_dim, hidden_dim, activation, num_rbf = num_rbf, cutoff = thre, dim_reduce = dim_reduce)  # v = dx/dt
        # self.interaction_net = InteractionTransformer(in_out_dim, hidden_dim, num_heads, num_layers, cutoff = thre, use_spatial = use_spatial)
        self.interaction_net = GNNInteraction(in_out_dim, hidden_dim, num_heads, num_layers, activation = activation, num_rbf = num_rbf, cutoff = thre, use_spatial = use_spatial, edge_predictor_path = edge_predictor_path, edge_predictor_thre = edge_predictor_thre)
    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)

            v = self.v_net(t, z).float()
            g = self.g_net(t, z).float()
            s = self.s_net(t, z).float()
            d = self.d_net(t, z).float()

        return v, g, s, d


class FNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super(FNet, self).__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.v_net = velocityNet(in_out_dim, hidden_dim, n_hiddens, activation)  # v = dx/dt
        self.g_net = growthNet(in_out_dim, hidden_dim, activation)  # g
        self.s_net = scoreNet(in_out_dim, hidden_dim, activation)  # s = log rho
        self.d_net = indediffusionNet(in_out_dim, hidden_dim, activation)  # d = sigma(t)

    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)

            v = self.v_net(t, z).float()
            g = self.g_net(t, z).float()
            s = self.s_net(t, z).float()
            d = self.d_net(t, z).float()

        return v, g, s, d

class ODEFunc2_interaction_energy(nn.Module):
    def __init__(self, f_net, use_mass = True, thre = 0.5, mass_detach = False):
        super(ODEFunc2_interaction_energy, self).__init__()
        self.f_net = f_net
        self.use_mass = use_mass
        self.thre = thre
        self.interaction_potential=f_net.interaction_net
        self.mass_detach = mass_detach

    def forward(self, t, state):
        z, lnw, _= state
        # print(z)
        # w = torch.exp(lnw)
        v, g, _, _ = self.f_net(t, z)
        
        dz_dt = v
        dlnw_dt = g
        # net_force = self.interaction_potential(z, lnw)
        net_force = cal_interaction(z, lnw, self.interaction_potential, t, m=1024, mass_detach = self.mass_detach).float()
        w = torch.exp(lnw)
        if self.use_mass:
            dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1)**2 + g**2) * w
        else:
            dm_dt = torch.norm(v, p=2,dim=1).unsqueeze(1)**2
        
        return dz_dt.float()+net_force.float(), dlnw_dt.float(), dm_dt.float()
    

    
class ODEFunc2_interaction(nn.Module):
    def __init__(self, f_net):
        super(ODEFunc2_interaction, self).__init__()
        self.f_net = f_net
        self.interaction_potential=f_net.interaction_net

    def forward(self, t, state):
        z, lnw,= state
        z.requires_grad_(True)
        lnw.requires_grad_(True)
        t.requires_grad_(True)
        # w = torch.exp(lnw)
        v, g, _, _ = self.f_net(t, z)
        
        dz_dt = v
        dlnw_dt = g
        # net_force = self.interaction_potential(z, lnw)
        net_force = cal_interaction(z, lnw, self.interaction_potential, t, m=1024).float()
        w = torch.exp(lnw)
        dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1)**2 + g**2) * w
        
        return dz_dt.float()+net_force.float(), dlnw_dt.float()


class ODEFunc_interaction(nn.Module):
    def __init__(self, f_net):
        super(ODEFunc_interaction, self).__init__()
        self.v_net = f_net.v_net
        self.interaction_potential=f_net.interaction_net

    def forward(self, t, z):
        dz_dt = self.v_net(t, z)
        batch_size, embed_dim = z.shape
        device = z.device

        # 随机打乱粒子索引
        perm = torch.randperm(batch_size, device=device)
        z_shuffled = z[perm]

        
        # 计算分组数量
        if batch_size % 2 == 0:
            num_pairs = batch_size // 2
            num_triples = 0
        else:
            num_pairs = (batch_size - 3) // 2
            num_triples = 1

        # 初始化力张量
        net_force = torch.zeros_like(z)

        # 处理两粒子组
        if num_pairs > 0:
            pairs_z = z_shuffled[:num_pairs * 2].view(num_pairs, 2, embed_dim)
            # 计算每对粒子之间的差值
            diff_pairs = pairs_z[:, 0] - pairs_z[:, 1]  # 形状: (num_pairs, embed_dim)
            diff_pairs.requires_grad_(True)
            potentials_pairs = self.interaction_potential(diff_pairs)
            grad_potential = torch.autograd.grad(
                outputs=potentials_pairs,
                inputs=diff_pairs,
                grad_outputs=torch.ones_like(potentials_pairs),
                create_graph=True,
            )[0]
            force_pairs = -grad_potential  # 形状: (num_pairs, embed_dim)
            # 分配力：粒子 i 受 -force，粒子 j 受 force
            force_assigned = torch.zeros(num_pairs * 2, embed_dim, device=device)
            force_assigned[0::2] = force_pairs  # 第一个粒子受正力
            force_assigned[1::2] = -force_pairs  # 第二个粒子受反向力
            # 系数 p-1 = 2-1 = 1，无需额外调整
            net_force[perm[:num_pairs * 2]] = force_assigned

        # 处理三粒子组（如果有）
        if num_triples > 0:
            triple_z = z_shuffled[num_pairs * 2:].view(1, 3, embed_dim)
            # 计算三粒子组内所有差值
            diff_triple = triple_z.unsqueeze(1) - triple_z.unsqueeze(0)  # 形状: (1, 3, 3, embed_dim)
            diff_triple_flat = diff_triple.view(-1, embed_dim)  # 形状: (9, embed_dim)
            diff_triple_flat.requires_grad_(True)
            potentials_triple = self.interaction_potential(diff_triple_flat)
            grad_potential = torch.autograd.grad(
                outputs=potentials_triple,
                inputs=diff_triple_flat,
                grad_outputs=torch.ones_like(potentials_triple),
                create_graph=True,
            )[0]
            force_triple_flat = -grad_potential  # 形状: (9, embed_dim)
            force_matrix = force_triple_flat.view(3, 3, embed_dim)
            # 排除自身作用力
            eye = torch.eye(3, device=device).unsqueeze(-1)
            force_matrix = force_matrix * (1 - eye)
            # 对每个粒子求和，系数 p-1 = 3-1 = 2
            force_triple = force_matrix.sum(dim=1) * 2
            net_force[perm[num_pairs * 2:]] = force_triple
        #w = torch.exp(lnw)
        #dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1) + g**2) * w
        return dz_dt.float()+net_force.float()

class ODEFunc2(nn.Module):
    def __init__(self, f_net, use_mass = True):
        super(ODEFunc2, self).__init__()
        self.f_net = f_net
        self.use_mass = use_mass

    def forward(self, t, state):
        z, lnw, _= state
        v, g, _, _ = self.f_net(t, z)
        
        dz_dt = v
        dlnw_dt = g
        w = torch.exp(lnw)
        if self.use_mass:
            dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1)**2 + g**2) * w
        else:
            dm_dt = torch.norm(v, p=2,dim=1).unsqueeze(1)**2
        
        return dz_dt.float(), dlnw_dt.float(), dm_dt.float()




class ODEFunc(nn.Module):
    def __init__(self, v_net):
        super(ODEFunc, self).__init__()
        self.v_net = v_net

    def forward(self, t, z):
        dz_dt = self.v_net(t, z)
        return dz_dt.float()

# %%
class scoreNet2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):

        state  = torch.cat((t,x),dim=1)
        return self.net(state)
    
    def compute_gradient(self, t, x):
        x = x.requires_grad_(True)
        output = self.forward(t, x)
        gradient = torch.autograd.grad(outputs=output, inputs=x,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=True)[0]
        return gradient

import torch
import torch.nn as nn

class scoreNet2_res(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, num_layers = 10, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.num_layers = num_layers

        self.input_layer = nn.Linear(in_out_dim + 1, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, t, x):
        state = torch.cat((t, x), dim=1)
        
        z = self.activation(self.input_layer(state))
        
        for i in range(self.num_layers):
            dz = self.activation(self.hidden_layers[i](z))
            z = z + dz  # Use out-of-place operation to avoid in-place modification
        
        out = self.output_layer(z)
        return out
    
    def compute_gradient(self, t, x):
        x = x.requires_grad_(True)
        output = self.forward(t, x)
        gradient = torch.autograd.grad(outputs=output, inputs=x,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=True)[0]
        return gradient

# Example usage:
# model = scoreNet2(in_out_dim=10, hidden_dim=20, num_layers=4, activation='relu')
# t = torch.randn(5, 1)
# x = torch.randn(5, 10)
# output = model.forward(t, x)
# print(output)


# %%
class ODEFunc3(nn.Module):
    def __init__(self, f_net,sf2m_score_model,sigma, use_mass, thre=1000):
        super(ODEFunc3, self).__init__()
        self.f_net = f_net
        self.interaction_potential = f_net.interaction_net
        self.sf2m_score_model = sf2m_score_model
        self.sigma=sigma
        self.use_mass = use_mass
        self.thre = thre


    def forward(self, t, state):
        z, lnw, m = state
        w = torch.exp(lnw)
        z.requires_grad_(True)
        lnw.requires_grad_(True)
        m.requires_grad_(True)
        t.requires_grad_(True)

        
        v, g, _, _ = self.f_net(t, z)
        v.requires_grad_(True)
        g.requires_grad_(True)
        #s.requires_grad_(True)
        time=t.expand(z.shape[0],1)
        time.requires_grad_(True)
        s=self.sf2m_score_model(time,z)
        
        dz_dt = v
        dlnw_dt = g

        z=z.requires_grad_(True)
        #grad_s = torch.autograd.grad(s.sum(), z)[0].requires_grad_() #need to change 
        grad_s = torch.autograd.grad(outputs=s, inputs=z,grad_outputs=torch.ones_like(s),create_graph=True)[0]

        norm_grad_s = torch.norm(grad_s, dim=1).unsqueeze(1).requires_grad_(True)
        # net_force = self.interaction_potential(z, lnw)
        net_force = cal_interaction(z, lnw, self.interaction_potential, t, m=1024).float()
        
        
        #w = torch.exp(lnw)
        #dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1) + g**2) * w
        if self.use_mass:
            if self.interaction_potential.cutoff != 0:

                dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                        (norm_grad_s ** 2) / 2 + torch.norm(v, p=2, dim=1).unsqueeze(1) * torch.norm(grad_s, p=2, dim=1).unsqueeze(1) + g ** 2) * w #torch.sum(v * grad_s, dim = 1, keepdim = True)
            else:
                dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                 (norm_grad_s ** 2) / 2 -
                 (1 / 2 * self.sigma ** 2 *g + s* g) + g ** 2) * w
        else:
            dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                    (norm_grad_s ** 2) / 2 + torch.norm(v, p=2, dim=1).unsqueeze(1) * torch.norm(grad_s, p=2, dim=1).unsqueeze(1))
        
        return dz_dt.float()+net_force.float(), dlnw_dt.float(), dm_dt.float()
