import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) Import your new CGDG+STFMamba cell
from STFMamba import CGDG_STFMambaCell
from util import merge_edges_over_time

class DATMamba(nn.Module): 
    def __init__(
        self,
        input_feat_dim,
        hidden_dim,
        encode_dim,
        n_head=3,
        embed_dim=5,
        graph_layer='GATv2Conv',
        decode_steps=10,
        use_edge_attr=False,
        use_lane=True,
        n_lane=4,
        dropout=0,
        edge_dropout=0,
        self_loop=False,
        head_concat=False,
        graph_aggr='mean',
        decode_graph=True,
        scene_feature_dim=3
    ):
        """
        Args:
            input_feat_dim (int): Dimension of input numeric features (excluding lane embeddings).
            hidden_dim (int): Hidden state dimension.
            encode_dim (int): Final embedding dimension after encoding.
            n_head (int): Number of attention heads.
            embed_dim (int): Embedding dimension for lane IDs.
            graph_layer (str): Which GNN to use (GATv2Conv, GATConv, TransformerConv, SAGEConv).
            decode_steps (int): Sequence length for decoding.
            use_edge_attr (bool): Whether to use pre-defined edge weights or not.
            use_lane (bool): Whether to use lane ID as input.
            n_lane (int): Number of lane categories for embedding.
            dropout (float): Dropout rate for fully-connected layers.
            edge_dropout (float): Dropout for graph edges in GNN.
            self_loop (bool): Add self-loop in GNN layers.
            head_concat (bool): Concatenate attention heads or average them.
            graph_aggr (str): Aggregation function (e.g., 'mean') for SAGEConv.
            decode_graph (bool): Whether to use graph connections in decoding.
            scene_feature_dim (int): Dimensionality of the global scene feature vector.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_feat_dim = input_feat_dim
        self.encode_dim = encode_dim
        self.dropout_layer = nn.Dropout(dropout)
        self.use_edge_attr = use_edge_attr
        self.use_lane = use_lane
        self.n_lane = n_lane
        self.edge_dropout = edge_dropout
        self.decode_graph = decode_graph
        self.scene_feature_dim = scene_feature_dim

        # 2) Lane embedding if needed
        enncode_input_dim = input_feat_dim
        if use_lane:
            self.lane_embedding = nn.Embedding(n_lane, embed_dim)
            enncode_input_dim += embed_dim

        # 3) Encoder cell (CGDG+STFMamba)
        self.GRNN_Cell_encode = CGDG_STFMambaCell(
            in_channels=enncode_input_dim,
            out_channels=hidden_dim,
            K=n_head,
            graph_layer=graph_layer,
            dropout=edge_dropout,
            self_loop=self_loop,
            head_concat=head_concat,
            aggr=graph_aggr,
            scene_feature_dim=scene_feature_dim
        )
        self.fc_encode = nn.Linear(hidden_dim, encode_dim)

        # 4) Decoder cell
        self.decode_steps = decode_steps
        self.output_dim_dist = 6  # (mean, logvar) for distance, speed, acceleration => 2 x 3
        self.output_dim_dist += n_lane  # lane classification
        self.fc_decode = nn.Linear(encode_dim, hidden_dim)
        self.fc_decode_output = nn.Linear(hidden_dim, self.output_dim_dist)

        self.GRNN_Cell_decode = CGDG_STFMambaCell(
            in_channels=self.output_dim_dist,
            out_channels=hidden_dim,
            K=n_head,
            graph_layer=graph_layer,
            dropout=edge_dropout,
            self_loop=self_loop,
            head_concat=head_concat,
            aggr=graph_aggr,
            scene_feature_dim=scene_feature_dim
        )

    def compute_scene_feature(self, graph):
        """
        Example function to compute or stub a global scene feature vector for CGDG.
        For instance, we can compute average speed or traffic density.

        Return shape: [1, scene_feature_dim]
        """
        # Let's say graph.x has shape [n_nodes, input_feat_dim+...].
        # We'll compute a naive "average speed" + "std speed" + "density" as an example.
        # In practice, adapt to your domain.
        x = graph.x
        # Suppose speed is at index 4, or adapt as needed
        speed_col = 4
        speed_vals = x[:, speed_col]
        avg_speed = speed_vals.mean()
        std_speed = speed_vals.std() if speed_vals.numel() > 1 else 0.0
        density = float(x.size(0))  # naive measure: number of vehicles

        # [1, scene_feature_dim]
        scene_feat = torch.tensor([[density, avg_speed, std_speed]], device=x.device, dtype=torch.float)
        return scene_feat

    def encode(self, graphs):
        """
        graphs - list of [graph_0, ..., graph_(time_steps-1)]
        graph_i.x - shape [n_nodes, input_feat_dim + 2], e.g. [Lane, class, x, speed, accel, mileage].
        returns encode_state, shape [n_nodes, encode_dim]
        """
        hidden_state = None
        input_seq_len = self.decode_steps

        for graph_i in graphs[:input_seq_len]:
            # 1) Prepare node features
            x_nodes = graph_i.x[:, 2:]  # numeric features after lane & class
            if self.use_lane:
                x_lane = self.lane_embedding(graph_i.x[:, 0].long())
                x_nodes = torch.cat((x_nodes, x_lane), dim=-1)

            # 2) Prepare scene feature
            scene_feat = self.compute_scene_feature(graph_i)

            # 3) Possibly use external edge weights
            edge_attr = graph_i.edge_attr if self.use_edge_attr else None

            # 4) Update hidden state with CGDG+STFMamba
            hidden_state = self.GRNN_Cell_encode(
                X=x_nodes,
                edge_index=graph_i.edge_index,
                H=hidden_state,
                scene_feature=scene_feat,
                external_edge_weight=edge_attr
            )

        output = self.fc_encode(hidden_state)
        return output

    def graph_x_to_decode_input(self, x):
        """
        Convert the last time-step observation into initial decoder input (distribution parameters).
        """
        n_nodes = x.shape[0]
        decoder_input = torch.zeros(n_nodes, self.output_dim_dist, device=x.device)

        # Mean for distance
        decoder_input[:, 0] = x[:, -1]  # distance in the last column
        # Mean for speed
        decoder_input[:, 2 + self.n_lane] = x[:, 3]
        # Mean for acceleration
        decoder_input[:, 4 + self.n_lane] = x[:, 4]

        # Lane classification distribution
        lane_rows = list(range(n_nodes))
        lane_cols = [2 + int(l) for l in x[:, 0]]
        decoder_input[lane_rows, lane_cols] = 1.0
        return decoder_input

    def decode(self, encode_state, decoder_input, graphs):
        """
        encode_state: [n_nodes, encode_dim]
        decoder_input: [n_nodes, output_dim_dist]
        graphs: list of Graph objects to possibly merge edges from or get scene features.

        If self.decode_graph is True, we merge edges over time.
        Otherwise, we skip or create an empty edge_index.
        """
        # 1) Prepare hidden state
        hidden_state = self.fc_decode(encode_state)
        hidden_state = self.dropout_layer(hidden_state)
        hidden_state = F.relu(hidden_state)

        # 2) Merge edges if needed
        if self.decode_graph:
            edge_index = merge_edges_over_time(graphs)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=encode_state.device)

        # 3) We can also pick one graph (e.g., the last) for scene features if needed
        #    or compute some aggregated scene feature for the entire sequence
        scene_feat = self.compute_scene_feature(graphs[-1])

        outputs = []
        for i in range(self.decode_steps):
            # CGDG+STFMamba in decoding
            hidden_state = self.GRNN_Cell_decode(
                X=decoder_input,
                edge_index=edge_index,
                H=hidden_state,
                scene_feature=scene_feat,
                external_edge_weight=None  # or pass a custom edge_attr if desired
            )
            output = self.fc_decode_output(hidden_state)
            decoder_input = output
            outputs.append(output)

        # stack outputs in [n_nodes, decode_steps, output_dim_dist]
        outputs = torch.stack(outputs[::-1], dim=1)
        return outputs

    def forward(self, graphs):
        """
        graphs: a list of Graph objects with length >= decode_steps + 1
        1) encode over the first decode_steps
        2) decode from the last state
        """
        # 1) Encode
        encode_state = self.encode(graphs)

        # 2) Use last graph's x to init decoder
        last_graph = graphs[-1]
        decoder_input = self.graph_x_to_decode_input(last_graph.x)

        # 3) Decode
        outputs = self.decode(encode_state, decoder_input, graphs)
        return outputs
