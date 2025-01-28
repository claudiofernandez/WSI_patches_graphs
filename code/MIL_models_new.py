import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, GINConv, GENConv
from torch_geometric.nn import LayerNorm


class ImprovedPatchGCN(torch.nn.Module):
    def __init__(self,
                 input_dim=512,
                 hidden_dims=[384, 256, 128],
                 num_layers=3,
                 n_classes=4,
                 dropout=0.2,
                 gnn_layer_type='GCNConv',
                 include_edge_features=False,
                 pooling="attention", num_features=512):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.gnn_type = gnn_layer_type
        self.num_classes = n_classes
        self.pooling = pooling

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        curr_dim = hidden_dims[1]

        for i in range(num_layers):
            layer_dict = {}

            if gnn_layer_type == 'GCNConv':
                layer_dict['conv'] = GCNConv(curr_dim, curr_dim)
            elif gnn_layer_type == 'GAT':
                layer_dict['conv'] = GATConv(
                    curr_dim,
                    curr_dim // 8,
                    heads=8,
                    concat=True,
                    dropout=dropout
                )
            elif gnn_layer_type == 'GINConv':
                # GINConv requires an MLP as the internal transformation
                gin_mlp = nn.Sequential(
                    nn.Linear(curr_dim, curr_dim),
                    nn.BatchNorm1d(curr_dim),
                    nn.GELU(),
                    nn.Linear(curr_dim, curr_dim)
                )
                layer_dict['conv'] = GINConv(gin_mlp)
            elif gnn_layer_type == 'GENConv':
                layer_dict['conv'] = GENConv(
                    curr_dim,
                    curr_dim,
                    aggr='softmax',
                    t=1.0,
                    learn_t=True,
                    num_layers=2,
                    norm='layer'
                )
            else:
                raise ValueError(f"Unsupported GNN layer type: {gnn_layer_type}")

            # Common l

            layer_dict.update({
                'norm': LayerNorm(curr_dim),
                'dropout': nn.Dropout(dropout),
                'residual': nn.Linear(curr_dim, curr_dim) if i > 0 else None
            })

            self.gnn_layers.append(nn.ModuleDict(layer_dict))

        # Attention pooling
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(curr_dim, curr_dim // 2),
                nn.Tanh(),
                nn.Linear(curr_dim // 2, 1)
            )

        # Classifier
        self.pre_classifier = nn.Sequential(
            nn.Linear(curr_dim * 2, hidden_dims[2]),
            LayerNorm(hidden_dims[2]),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_dims[2], n_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use ReLU gain for GELU as an approximation
            gain = nn.init.calculate_gain('relu')
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def attention_pool(self, x, batch):
        scores = self.attention(x).squeeze(-1)
        scores = F.softmax(scores, dim=0)

        out = None
        for b in torch.unique(batch):
            mask = (batch == b)
            scored_x = x[mask] * scores[mask].unsqueeze(-1)
            pooled = scored_x.sum(dim=0, keepdim=True)
            out = pooled if out is None else torch.cat([out, pooled], dim=0)
        return out

    def forward(self, data):
        x, edge_index = data['x'], data['edge_index']
        batch = data.get('batch', None)

        # Initial feature projection
        x = self.feature_projection(x)

        # Store multi-scale features
        features = []

        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            identity = x

            x = layer['conv'](x, edge_index)
            x = layer['norm'](x)
            x = F.gelu(x)
            x = layer['dropout'](x)

            if i > 0 and layer['residual'] is not None:
                x = x + layer['residual'](identity)

            features.append(x)

        # Multi-scale feature aggregation
        x = torch.stack(features, dim=0).mean(dim=0)

        # Pooling
        if batch is not None:
            if self.pooling == "attention":
                x_pooled = self.attention_pool(x, batch)
            else:
                x_mean = global_mean_pool(x, batch)
                x_max = global_max_pool(x, batch)
                x_pooled = torch.cat([x_mean, x_max], dim=1)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]
            x_pooled = torch.cat([x_mean, x_max], dim=1)

        # Classification
        x = self.pre_classifier(x_pooled)
        logits = self.classifier(x)

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        return Y_prob, Y_hat, logits, x