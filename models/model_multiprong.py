
# import torch
# import torch.nn as nn
# from models.protas_model import MultiStageModel

# from utils.model_util import get_linear_layers_w_activations
# from utils.logging import configure_logging_format

# logger = configure_logging_format()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class MultiProngAttDropoutModel(nn.Module):
#     def __init__(
#         self, 
#         base_model_class,
#         base_model_params,
#         output_dimensionality,
#         num_heads,
#         dropping=False,
#         attn_layers = [512, 256],
#         drop_layers = [512, 128, 256],
#         use_protas = False,
#         protas_params = None,
#     ):
#         super(MultiProngAttDropoutModel, self).__init__()
#         self.num_classes = num_heads
#         self.dropping = dropping
#         self.output_dimensionality = output_dimensionality

#         # --- ADD THIS LOGIC ---
#         self.use_protas = use_protas
#         if self.use_protas:
#             from models.protas_model import MultiStageModel
#             self.protas_model = MultiStageModel(**protas_params)

#         else:
#             # shared base model
#             self.base_model = base_model_class(**base_model_params)
#             # all prongs
#             self.head_models = nn.ModuleList({
#                 HeadModel(output_dimensionality, class_name=head_id) for head_id in range(num_heads)
#             })
#             # Attention Mechanism
#             self.attention_layer = nn.Sequential(
#                 nn.Linear(output_dimensionality * self.num_classes, attn_layers[0]),  # Adjust the architecture as needed
#                 # nn.Dropout(p=0.2),
#                 *get_linear_layers_w_activations(attn_layers, activation_at_end=True, activation=nn.Tanh()),
#                 nn.Linear(attn_layers[-1], self.num_classes),
#                 nn.Softmax(dim=1)  # Apply softmax to get attention weights
#             ).to(device)
#             if self.dropping:
#                 self.dropout = nn.Sequential(
#                     nn.Linear(output_dimensionality, drop_layers[0]),  # Adjust the architecture as needed
#                     *get_linear_layers_w_activations(drop_layers, activation_at_end=True, activation=nn.ReLU()),
#                     nn.Linear(drop_layers[-1], output_dimensionality + 1)
#                 ).to(device)

#     def forward(self, videos):
#         print(f"\n[MODEL DEBUG] Input type: {type(videos)}")
#         if isinstance(videos, (list, tuple)):
#             print(f"[MODEL DEBUG] Input is list, unpacking index 0. Type: {type(videos[0])}")
#             videos = videos[0]

#         print(f"[MODEL DEBUG] Final Tensor Shape before processing: {videos.shape}")

#         if videos.dim() == 4:
#             print(f"[MODEL DEBUG] Squeezing dim 1 from {videos.shape}")
#             videos = videos.squeeze(1)

#         if self.use_protas:
#             # Expected: [Batch, Time, Channels]
            
#             # GTCC input x: [Batch, Time, Channels]
#             # ProTAS expects: [Batch, Channels, Time]
#             x = videos.transpose(1, 2) 
#             print(f"[MODEL DEBUG] Shape for Conv1D: {x.shape}")
#             mask = torch.ones(x.size(0), 1, x.size(2)).to(x.device)
            
#             # ProTAS model returns (action_out, progress_out)
#             # action_out: [Stages, Batch, Classes, Time]
#             # progress_out: [Stages, Batch, 1, Time]
#             _, progress_out = self.protas_model(x, mask)
            
#             # Take progress from the final stage and remove extra dims
#             # Result shape: [Batch, Time]
#             final_progress = progress_out[-1].squeeze(1)
#             print(f"[MODEL DEBUG] Output Progress Shape: {final_progress.shape}")
            
#             return {'progress': final_progress, 'outputs': [None]}

#         general_features = self.base_model(videos)['outputs']
#         prong_outputs = [prong(general_features) for prong in self.head_models]
#         prong_outputs = list(map(list, zip(*prong_outputs))) # list transpose to get (batch, video_heads)
#         outputs = []
#         attentions = []
#         dropouts = []
#         for prong_output in prong_outputs:
#             T = prong_output[0].shape[0]
#             prong_output_t = torch.stack(prong_output, dim=0)
#             concatenated_prongs = torch.stack(prong_output, dim=0).view(T, -1)
#             attention_weights = self.attention_layer(concatenated_prongs)
#             weighted_combination = prong_output_t.permute(2,1,0) * attention_weights
#             combined_embedding = weighted_combination.sum(dim=2).T
#             outputs.append(combined_embedding)
#             attentions.append(attention_weights)
#             if self.dropping:
#                 dout = self.dropout(combined_embedding).mean(dim=0)
#                 dropouts.append(dout)

#         if self.dropping:
#             return {'outputs': outputs, 'attentions': attentions, 'dropouts': dropouts}
#         else:
#             return {'outputs': outputs, 'attentions': attentions}

# class HeadModel(nn.Module):
#     def __init__(self, output_dimensionality, class_name, layers=[512, 128, 256]):
#         super(HeadModel, self).__init__()
#         self.class_name = class_name
#         self.fc_layers = nn.Sequential(
#                 nn.Linear(output_dimensionality, layers[0]),  # Adjust the architecture as needed
#                 *get_linear_layers_w_activations(layers, activation_at_end=True, activation=nn.ReLU()),
#                 nn.Linear(layers[-1], output_dimensionality)
#             ).to(device)
        
#     def forward(self, x):
#         outputs = []
#         for i, sequence in enumerate(x):
#             this_video = self.fc_layers(sequence)
#             # print(this_video.shape, sequence.shape)
#             outputs.append(this_video)
#         return outputs


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_util import get_linear_layers_w_activations
from utils.logging import configure_logging_format

logger = configure_logging_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Progress Head Factory Function
# =============================================================================

def create_progress_head(input_dim, config):
    """
    Factory function to create the appropriate ProgressHead based on config.

    Args:
        input_dim: Input embedding dimension (e.g., 128)
        config: Dictionary with progress head configuration

    Returns:
        ProgressHead module (GRU, Transformer, or DilatedConv based)
    """
    architecture = config.get('architecture', 'gru')

    if architecture == 'transformer':
        transformer_config = config.get('transformer_config', {})
        return TransformerProgressHead(
            input_dim=input_dim,
            d_model=transformer_config.get('d_model', 64),
            num_heads=transformer_config.get('num_heads', 4),
            num_layers=transformer_config.get('num_layers', 2),
            ffn_dim=transformer_config.get('ffn_dim', 128),
            dropout=transformer_config.get('dropout', 0.1),
        )
    elif architecture == 'dilated_conv':
        dilated_config = config.get('dilated_conv_config', {})
        return DilatedConvProgressHead(
            input_dim=input_dim,
            hidden_dim=dilated_config.get('hidden_dim', 64),
            kernel_size=dilated_config.get('kernel_size', 3),
            dilations=dilated_config.get('dilations', [1, 2, 4, 8, 16, 32]),
            dropout=dilated_config.get('dropout', 0.1),
        )
    else:
        # Default: GRU-based ProgressHead (backward compatible)
        return ProgressHead(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 64),
            use_gru=config.get('use_gru', True),
            use_position_encoding=config.get('use_position_encoding', False),
        )


class ProgressHead(nn.Module):
    """
    Learnable progress prediction head.

    Supports two modes for backward compatibility:
    - Legacy mode (use_position_encoding=False): Original GRU without position encoding (v2 checkpoints)
    - New mode (use_position_encoding=True): With position encoding and learnable h0 (v3+)

    Takes segment embeddings and predicts progress at the final frame.
    """
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True, use_position_encoding=False):
        super(ProgressHead, self).__init__()
        self.use_gru = use_gru
        self.input_dim = input_dim
        self.use_position_encoding = use_position_encoding

        # GRU input dimension: +1 only if using position encoding
        gru_input_dim = input_dim + 1 if use_position_encoding else input_dim

        if use_gru:
            self.gru = nn.GRU(gru_input_dim, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=False)
            # Learnable h0 only for new mode
            if use_position_encoding:
                self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(gru_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, segment_embeddings):
        T = segment_embeddings.shape[0]
        device = segment_embeddings.device

        if self.use_position_encoding:
            # Position encoding: normalized frame index (0 to ~1)
            positions = torch.arange(T, device=device, dtype=torch.float32) / max(T, 1)
            positions = positions.unsqueeze(1)  # (T, 1)
            x = torch.cat([segment_embeddings, positions], dim=1)  # (T, D+1)
        else:
            # Legacy mode: no position encoding
            x = segment_embeddings  # (T, D)

        if self.use_gru:
            x = x.unsqueeze(0)  # (1, T, D) or (1, T, D+1)
            if self.use_position_encoding:
                _, h_n = self.gru(x, self.h0)  # Use learnable h0
            else:
                _, h_n = self.gru(x)  # Legacy: zero h0
            progress = self.fc(h_n.squeeze())
        else:
            x = x.mean(dim=0)
            progress = self.fc(x)
        return progress.squeeze()


# =============================================================================
# TransformerProgressHead with ALiBi
# =============================================================================

class TransformerProgressHead(nn.Module):
    """
    Transformer-based progress prediction head with ALiBi (Attention with Linear Biases).

    ALiBi provides relative positional awareness without explicit position tokens,
    avoiding the contradictory signals that occur with absolute position encoding.

    Architecture:
    - Input projection: Linear(input_dim -> d_model)
    - Causal Transformer blocks with ALiBi attention
    - Take last token (causal aggregation)
    - Output MLP: d_model -> 32 -> 1 -> Sigmoid
    """

    def __init__(self, input_dim=128, d_model=64, num_heads=4, num_layers=2,
                 ffn_dim=128, dropout=0.1):
        super(TransformerProgressHead, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlockWithALiBi(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # ALiBi slopes (geometric sequence)
        # For 4 heads: [1.0, 0.5, 0.25, 0.125]
        slopes = []
        for i in range(num_heads):
            slopes.append(1.0 / (2 ** i))
        self.register_buffer('alibi_slopes', torch.tensor(slopes))

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _get_alibi_bias(self, T, device):
        """
        Compute ALiBi bias matrix.

        For position i attending to position j (where j <= i for causal):
        bias[i, j] = -slope * (i - j)
        """
        # Create distance matrix
        positions = torch.arange(T, device=device)
        # distance[i, j] = i - j
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)  # (T, T)

        # ALiBi bias: -slope * distance for each head
        # Shape: (num_heads, T, T)
        alibi_bias = -self.alibi_slopes.view(-1, 1, 1) * distance.unsqueeze(0).float()

        return alibi_bias

    def forward(self, segment_embeddings):
        """
        Args:
            segment_embeddings: (T, input_dim) - variable length sequence

        Returns:
            progress: scalar [0, 1]
        """
        T = segment_embeddings.shape[0]
        device = segment_embeddings.device

        # Handle single frame case
        if T == 1:
            x = self.input_proj(segment_embeddings)  # (1, d_model)
            return self.output_mlp(x.squeeze(0)).squeeze()

        # 1. Input projection
        x = self.input_proj(segment_embeddings)  # (T, d_model)
        x = x.unsqueeze(0)  # (1, T, d_model)

        # 2. Create causal mask (True = masked/ignored)
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        # 3. Compute ALiBi bias
        alibi_bias = self._get_alibi_bias(T, device)  # (num_heads, T, T)

        # 4. Apply transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask, alibi_bias)

        # 5. Take last token (causal aggregation)
        final = x[0, -1, :]  # (d_model,)

        # 6. Output MLP
        progress = self.output_mlp(final)

        return progress.squeeze()


class TransformerBlockWithALiBi(nn.Module):
    """
    Single transformer block with ALiBi attention.
    Uses pre-norm architecture for training stability.
    """

    def __init__(self, d_model, num_heads, ffn_dim, dropout):
        super(TransformerBlockWithALiBi, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, alibi_bias):
        """
        Args:
            x: (1, T, d_model)
            causal_mask: (T, T) - True means masked
            alibi_bias: (num_heads, T, T)
        """
        B, T, D = x.shape

        # Pre-norm + attention
        x_norm = self.norm1(x)

        # Compute Q, K, V
        Q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        K = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores with ALiBi
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)
        attn_scores = attn_scores + alibi_bias.unsqueeze(0)  # Add ALiBi bias

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and apply to values
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, V)  # (B, H, T, head_dim)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        # Residual connection
        x = x + attn_out

        # Pre-norm + FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        ffn_out = self.dropout(ffn_out)

        # Residual connection
        x = x + ffn_out

        return x


# =============================================================================
# DilatedConvProgressHead
# =============================================================================

class DilatedConvProgressHead(nn.Module):
    """
    Dilated convolution-based progress prediction head.

    Uses causal dilated convolutions to capture multi-scale temporal patterns
    without explicit position encoding. Similar architecture to ProTAS.

    Architecture:
    - Input projection: Linear(input_dim -> hidden_dim)
    - Causal dilated residual blocks (dilations: 1, 2, 4, 8, 16, 32)
    - Take last position (causal pooling)
    - Output MLP: hidden_dim -> 32 -> 1 -> Sigmoid
    """

    def __init__(self, input_dim=128, hidden_dim=64, kernel_size=3,
                 dilations=None, dropout=0.1):
        super(DilatedConvProgressHead, self).__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dilations = dilations

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Dilated residual blocks
        self.blocks = nn.ModuleList([
            CausalDilatedResidualBlock(hidden_dim, d, kernel_size, dropout)
            for d in dilations
        ])

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, segment_embeddings):
        """
        Args:
            segment_embeddings: (T, input_dim) - variable length sequence

        Returns:
            progress: scalar [0, 1]
        """
        T = segment_embeddings.shape[0]

        # Handle single frame case
        if T == 1:
            x = self.input_proj(segment_embeddings)  # (1, hidden_dim)
            return self.output_mlp(x.squeeze(0)).squeeze()

        # 1. Input projection
        x = self.input_proj(segment_embeddings)  # (T, hidden_dim)

        # 2. Transpose for Conv1d: (T, C) -> (1, C, T)
        x = x.transpose(0, 1).unsqueeze(0)  # (1, hidden_dim, T)

        # 3. Apply dilated residual blocks
        for block in self.blocks:
            x = block(x)

        # 4. Take last position (causal)
        final = x[0, :, -1]  # (hidden_dim,)

        # 5. Output MLP
        progress = self.output_mlp(final)

        return progress.squeeze()


class CausalDilatedResidualBlock(nn.Module):
    """
    Causal dilated residual block.

    Uses left-only padding to maintain causality.
    """

    def __init__(self, channels, dilation, kernel_size=3, dropout=0.1):
        super(CausalDilatedResidualBlock, self).__init__()

        self.dilation = dilation
        self.kernel_size = kernel_size
        # Causal padding: (kernel_size - 1) * dilation on the left
        self.causal_pad = (kernel_size - 1) * dilation

        # Dilated conv (no padding - we pad manually)
        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=0
        )
        self.bn = nn.BatchNorm1d(channels)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, C, T)

        Returns:
            (B, C, T) - same shape with residual connection
        """
        residual = x

        # Causal padding (left only)
        x_padded = F.pad(x, (self.causal_pad, 0))  # (B, C, T + pad)

        # Dilated conv
        out = self.conv_dilated(x_padded)  # (B, C, T)
        out = self.bn(out)
        out = F.relu(out)

        # 1x1 conv for channel mixing
        out = self.conv_1x1(out)
        out = self.dropout(out)

        # Residual connection
        return residual + out


class MultiProngAttDropoutModel(nn.Module):
    def __init__(
        self,
        base_model_class,
        base_model_params,
        output_dimensionality,
        num_heads,
        dropping=False,
        attn_layers = [512, 256],
        drop_layers = [512, 128, 256],
        use_progress_head=False,
        progress_head_config=None,
    ):
        super(MultiProngAttDropoutModel, self).__init__()
        self.num_classes = num_heads
        self.dropping = dropping
        self.output_dimensionality = output_dimensionality
        # shared base model
        self.base_model = base_model_class(**base_model_params)
        # all prongs
        self.head_models = nn.ModuleList({
            HeadModel(output_dimensionality, class_name=head_id) for head_id in range(num_heads)
        })
        # Attention Mechanism
        self.attention_layer = nn.Sequential(
            nn.Linear(output_dimensionality * self.num_classes, attn_layers[0]),  # Adjust the architecture as needed
            # nn.Dropout(p=0.2),
            *get_linear_layers_w_activations(attn_layers, activation_at_end=True, activation=nn.Tanh()),
            nn.Linear(attn_layers[-1], self.num_classes),
            nn.Softmax(dim=1)  # Apply softmax to get attention weights
        )
        if self.dropping:
            self.dropout = nn.Sequential(
                nn.Linear(output_dimensionality, drop_layers[0]),  # Adjust the architecture as needed
                *get_linear_layers_w_activations(drop_layers, activation_at_end=True, activation=nn.ReLU()),
                nn.Linear(drop_layers[-1], output_dimensionality + 1)
            )

        # Progress head (for learnable progress loss)
        self.use_progress_head = use_progress_head
        if use_progress_head and progress_head_config is not None:
            # Use factory function to create the appropriate ProgressHead
            self.progress_head = create_progress_head(
                input_dim=output_dimensionality,
                config=progress_head_config
            )

    def forward(self, videos):
        general_features = self.base_model(videos)['outputs']
        prong_outputs = [prong(general_features) for prong in self.head_models]
        prong_outputs = list(map(list, zip(*prong_outputs))) # list transpose to get (batch, video_heads)
        outputs = []
        attentions = []
        dropouts = []
        for prong_output in prong_outputs:
            T = prong_output[0].shape[0]
            prong_output_t = torch.stack(prong_output, dim=0)
            concatenated_prongs = torch.stack(prong_output, dim=0).view(T, -1)
            attention_weights = self.attention_layer(concatenated_prongs)
            weighted_combination = prong_output_t.permute(2,1,0) * attention_weights
            combined_embedding = weighted_combination.sum(dim=2).T
            outputs.append(combined_embedding)
            attentions.append(attention_weights)
            if self.dropping:
                dout = self.dropout(combined_embedding).mean(dim=0)
                dropouts.append(dout)

        result = {'outputs': outputs, 'attentions': attentions}
        if self.dropping:
            result['dropouts'] = dropouts
        if self.use_progress_head:
            result['progress_head'] = self.progress_head
        return result


class HeadModel(nn.Module):
    def __init__(self, output_dimensionality, class_name, layers=[512, 128, 256]):
        super(HeadModel, self).__init__()
        self.class_name = class_name
        self.fc_layers = nn.Sequential(
                nn.Linear(output_dimensionality, layers[0]),  # Adjust the architecture as needed
                *get_linear_layers_w_activations(layers, activation_at_end=True, activation=nn.ReLU()),
                nn.Linear(layers[-1], output_dimensionality)
            )
        
    def forward(self, x):
        outputs = []
        for i, sequence in enumerate(x):
            this_video = self.fc_layers(sequence)
            # print(this_video.shape, sequence.shape)
            outputs.append(this_video)
        return outputs
