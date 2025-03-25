import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, SAGEConv
from torch.nn import Conv1d

# --------------------------------------------------------------------------- #
#                     CGDG + STFMamba Integrated Cell
# --------------------------------------------------------------------------- #
class CGDG_STFMambaCell(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 1,
        graph_layer: str = 'GATConv',
        dropout: float = 0.0,
        self_loop: bool = True,
        head_concat: bool = False,
        aggr: str = 'mean',
        scene_feature_dim: int = 3,
        embed_dim: int = 8
    ):
        """
        Args:
            in_channels (int): Node feature dimension.
            out_channels (int): Hidden state dimension.
            K (int): Number of attention heads (for GAT/GATv2/TransformerConv).
            graph_layer (str): Graph layer type ('GATConv', 'GATv2Conv', 'TransformerConv', 'SAGEConv').
            dropout (float): Dropout rate for graph layers.
            self_loop (bool): Whether to add self-loops (for GAT-based layers).
            head_concat (bool): If True, concatenate attention heads before projecting back to out_channels.
            aggr (str): Aggregation strategy for SAGEConv (e.g., 'mean').
            scene_feature_dim (int): Dimensionality of the global scene feature vector f_t.
            embed_dim (int): Hidden dimension for projecting scene features and computing softmax scores.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.graph_layer = graph_layer
        self.dropout = dropout
        self.self_loop = self_loop
        self.head_concat = head_concat
        self.aggr = aggr

        # ------------------------- Graph Layers (Update/Reset) ------------------------- #
        if self.graph_layer == 'GATConv':
            self.conv_x_z = GATConv(in_channels + out_channels, out_channels,
                                    heads=K, concat=head_concat, dropout=dropout,
                                    add_self_loops=self_loop)
            self.conv_x_r = GATConv(in_channels + out_channels, out_channels,
                                    heads=K, concat=head_concat, dropout=dropout,
                                    add_self_loops=self_loop)
        elif self.graph_layer == 'GATv2Conv':
            self.conv_x_z = GATv2Conv(in_channels + out_channels, out_channels,
                                      heads=K, concat=head_concat, dropout=dropout,
                                      add_self_loops=self_loop)
            self.conv_x_r = GATv2Conv(in_channels + out_channels, out_channels,
                                      heads=K, concat=head_concat, dropout=dropout,
                                      add_self_loops=self_loop)
        elif self.graph_layer == 'TransformerConv':
            self.conv_x_z = TransformerConv(in_channels + out_channels, out_channels,
                                            heads=K, concat=head_concat, dropout=dropout)
            self.conv_x_r = TransformerConv(in_channels + out_channels, out_channels,
                                            heads=K, concat=head_concat, dropout=dropout)
        elif self.graph_layer == 'SAGEConv':
            self.conv_x_z = SAGEConv(in_channels + out_channels, out_channels, aggr=aggr)
            self.conv_x_r = SAGEConv(in_channels + out_channels, out_channels, aggr=aggr)
        else:
            raise NotImplementedError(f"{graph_layer} not implemented.")

        # If head_concat=True, map from (out_channels*K) -> out_channels
        if head_concat and self.graph_layer in ['GATConv', 'GATv2Conv', 'TransformerConv']:
            self.fc_z = nn.Linear(out_channels * K, out_channels)
            self.fc_r = nn.Linear(out_channels * K, out_channels)

        # ------------------------- Graph Layer (Candidate State) ------------------------- #
        if self.graph_layer == 'GATConv':
            self.conv_x_h = GATConv(in_channels + out_channels, out_channels,
                                    heads=K, concat=head_concat, dropout=dropout,
                                    add_self_loops=self_loop)
        elif self.graph_layer == 'GATv2Conv':
            self.conv_x_h = GATv2Conv(in_channels + out_channels, out_channels,
                                      heads=K, concat=head_concat, dropout=dropout,
                                      add_self_loops=self_loop)
        elif self.graph_layer == 'TransformerConv':
            self.conv_x_h = TransformerConv(in_channels + out_channels, out_channels,
                                            heads=K, concat=head_concat, dropout=dropout)
        elif self.graph_layer == 'SAGEConv':
            self.conv_x_h = SAGEConv(in_channels + out_channels, out_channels, aggr=aggr)

        if head_concat and self.graph_layer in ['GATConv', 'GATv2Conv', 'TransformerConv']:
            self.fc_h = nn.Linear(out_channels * K, out_channels)

        # ------------------------- STFMamba Modules ------------------------- #
        # (1) Linear projection: from out_channels -> 2*out_channels
        self.win = nn.Linear(out_channels, 2 * out_channels)
        # (2) 1D convolution for "main" branch in STFMamba
        self.conv1d = Conv1d(out_channels, out_channels, kernel_size=1)
        # (3) GSSSM for global/long-range feature modeling
        self.gsssm = GSSSM(hidden_dim=out_channels)
        # (4) Output projection after the residual fusion
        self.wout = nn.Linear(out_channels, out_channels)

        # ------------------------- CGDG Attention Weights ------------------------- #
        # We assume a global scene feature f_t in R^(scene_feature_dim).
        # We'll project it to 'embed_dim' and have 3 embeddings for NA, SD, CR.
        self.scene_fc = nn.Linear(scene_feature_dim, embed_dim)
        self.u_na = nn.Parameter(torch.randn(embed_dim))
        self.u_sd = nn.Parameter(torch.randn(embed_dim))
        self.u_cr = nn.Parameter(torch.randn(embed_dim))

        # Additional parameters (e.g., sigma_dist, sigma_F) can be added here
        self.sigma_dist = 1.0
        self.sigma_F = 1.0

    def _set_hidden_state(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Initialize hidden state if None."""
        if H is None:
            H = torch.zeros(X.size(0), self.out_channels, device=X.device)
        return H

    # ----------------------------------------------------------------------- #
    #  CGDG: compute A^NA, A^SD, A^CR, then combine via dynamic attention
    # ----------------------------------------------------------------------- #
    def compute_adaptive_dynamic_edge_weight(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        scene_feature: torch.Tensor
    ) -> torch.Tensor:
        """
          1) A^NA: neighbor attributes
          2) A^SD: social distance
          3) A^CR: collision risk
        Then merges them via attention-based weighting w_t,k.

        Args:
            X (torch.Tensor): Node features [num_nodes, in_channels].
                              Assume X[:,0], X[:,1] are positions, X[:,2] is velocity, etc.
            edge_index (torch.Tensor): [2, E] representing source->target edges.
            scene_feature (torch.Tensor): [batch_size=1, scene_feature_dim], 
                                          e.g., traffic density, average speed, etc.

        Returns:
            edge_weight (torch.Tensor): [E], final normalized edge weights in [0,1].
        """
        src, dst = edge_index
        pos_src = X[src, :2]  # (x, y) for source node
        pos_dst = X[dst, :2]  # (x, y) for target node
        dist = torch.norm(pos_src - pos_dst, dim=1)  # Euclidean distance

        # --- 1) A^NA (neighbor attributes) ---
        #   = 1 if v^i precedes v^j and lateral distance is small
        #     otherwise 0
        #   We'll assume "precedes" means pos_src[:,0] < pos_dst[:,0].
        #   And lateral distance < 1 lane as an example.
        cond_na = (pos_src[:, 0] < pos_dst[:, 0]) & (torch.abs(pos_src[:, 1] - pos_dst[:, 1]) < 1.0)
        A_NA = cond_na.float()  # [E]

        # --- 2) A^SD (social distance) ---
        #   = exp( - (dist / sigma_dist)^2 )
        A_SD = torch.exp(- (dist / self.sigma_dist) ** 2)

        # --- 3) A^CR (collision risk) ---
        #   = tanh( Force / sigma_F )
        #   Here we simplify Force as |velocity_src - velocity_dst|
        #   or use the formula from the snippet if you have mass, etc.
        #   We'll assume velocity is in X[:,2].
        if X.size(1) > 2:
            v_src = X[src, 2]
            v_dst = X[dst, 2]
            force = torch.abs(v_src - v_dst)
        else:
            force = torch.zeros_like(dist)

        A_CR = torch.tanh(force / self.sigma_F)

        # --- Compute dynamic attention weights w_t,NA, w_t,SD, w_t,CR ---
        # scene_feature: [1, scene_feature_dim]
        # project to embed_dim
        f_t = self.scene_fc(scene_feature)  # [1, embed_dim]
        # dot with each adjacency type embedding
        score_na = (f_t * self.u_na).sum(dim=-1)  # scalar
        score_sd = (f_t * self.u_sd).sum(dim=-1)
        score_cr = (f_t * self.u_cr).sum(dim=-1)
        scores = torch.stack([score_na, score_sd, score_cr], dim=-1)  # [1, 3]

        # softmax over the 3 adjacency types
        weights = F.softmax(scores / math.sqrt(f_t.size(-1)), dim=-1)  # [1, 3]
        w_na, w_sd, w_cr = weights[0, 0], weights[0, 1], weights[0, 2]

        # --- Combine adjacency matrices ---
        edge_weight = w_na * A_NA + w_sd * A_SD + w_cr * A_CR

        # --- Normalize to [0,1] ---
        max_val = edge_weight.max().detach()
        if max_val > 0:
            edge_weight = edge_weight / (max_val + 1e-9)

        return edge_weight

    # ----------------------------------------------------------------------- #
    #                        Forward Pass
    # ----------------------------------------------------------------------- #
    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        H: torch.Tensor = None,
        scene_feature: torch.Tensor = None,
        external_edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass integrating CGDG and STFMamba.

        Args:
            X (torch.Tensor): [num_nodes, in_channels] node features.
            edge_index (torch.Tensor): [2, E] graph edges (source->target).
            H (torch.Tensor, optional): [num_nodes, out_channels], previous hidden state. Defaults to None.
            scene_feature (torch.Tensor, optional): [1, scene_feature_dim] global scene descriptor
                                                    (e.g., traffic density, avg speed). Defaults to None.
            external_edge_weight (torch.Tensor, optional): [E], user-provided edge weights. If None,
                                                           we'll compute them via CGDG.

        Returns:
            torch.Tensor: Updated hidden state [num_nodes, out_channels].
        """
        # 1) Initialize hidden state if needed
        H = self._set_hidden_state(X, H)

        # 2) Compute or use provided edge weights
        if external_edge_weight is not None:
            edge_weight = external_edge_weight
        else:
            if scene_feature is None:
                # If no scene feature is provided, you can default to zero or random
                scene_feature = torch.zeros(1, self.scene_fc.in_features, device=X.device)
            edge_weight = self.compute_adaptive_dynamic_edge_weight(X, edge_index, scene_feature)

        # 3) Update gate Z
        Z_in = torch.cat([X, H], dim=1)
        Z = self.conv_x_z(Z_in, edge_index, edge_weight)
        if self.head_concat and self.graph_layer in ['GATConv', 'GATv2Conv', 'TransformerConv']:
            Z = F.relu(Z)
            Z = self.fc_z(Z)
        Z = torch.sigmoid(Z)

        # 4) Reset gate R
        R_in = torch.cat([X, H], dim=1)
        R = self.conv_x_r(R_in, edge_index, edge_weight)
        if self.head_concat and self.graph_layer in ['GATConv', 'GATv2Conv', 'TransformerConv']:
            R = F.relu(R)
            R = self.fc_r(R)
        R = torch.sigmoid(R)

        # 5) Candidate state H_tilde
        H_tilde_in = torch.cat([X, H * R], dim=1)
        H_tilde = self.conv_x_h(H_tilde_in, edge_index, edge_weight)
        if self.head_concat and self.graph_layer in ['GATConv', 'GATv2Conv', 'TransformerConv']:
            H_tilde = F.relu(H_tilde)
            H_tilde = self.fc_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)

        # 6) STFMamba residual fusion
        #    (7) Linear projection -> split into h_main and res
        h_main_res = self.win(H_tilde)  # [num_nodes, 2*out_channels]
        h_main, res = torch.split(h_main_res, self.out_channels, dim=1)

        #    1D conv over h_main (demo: kernel_size=1)
        h_main_1d = h_main.unsqueeze(-1)  # shape: [num_nodes, out_channels, 1]
        h_prime_main = self.conv1d(h_main_1d).squeeze(-1)  # [num_nodes, out_channels]
        h_prime_main = F.silu(h_prime_main)

        #    GSSSM for global/long-range modeling
        h_sssm = self.gsssm(h_prime_main)  # [num_nodes, out_channels]

        #    (8) Residual fusion with res
        fused = h_sssm * F.silu(res)  # element-wise multiplication
        h_mamba_out = self.wout(fused)  # [num_nodes, out_channels]

        # 7) Final hidden state update
        H_new = Z * H + (1 - Z) * h_mamba_out
        return H_new
# --------------------------------------------------------------------------- #
#                         GSSSM (Global Selective SSM)
# --------------------------------------------------------------------------- #
class GSSSM(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [num_nodes, hidden_dim]
        # This example simply applies a linear layer + tanh activation.
        return torch.tanh(self.linear(x))