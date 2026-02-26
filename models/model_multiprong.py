
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


import torch
import torch.nn as nn

from utils.model_util import get_linear_layers_w_activations
from utils.logging import configure_logging_format

logger = configure_logging_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProgressHead(nn.Module):
    """
    Learnable progress prediction head with position encoding and learnable h0.

    Improvements over original (see ProgressIssues2.md):
    - Position encoding: Concatenates normalized frame index to embeddings
      This tells the GRU "where" it is in the sequence (frame 1/T vs frame T/T)
    - Learnable h0: Learns initial hidden state instead of zero-init
      This reduces the Sigmoid bias toward 0.5 at early frames

    Takes segment embeddings and predicts progress at the final frame.
    """
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True):
        super(ProgressHead, self).__init__()
        self.use_gru = use_gru
        self.input_dim = input_dim

        if use_gru:
            # +1 for position encoding
            self.gru = nn.GRU(input_dim + 1, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=False)
            # Learnable initial hidden state
            self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            # +1 for position encoding (mean-pooled position)
            self.fc = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, segment_embeddings):
        T = segment_embeddings.shape[0]
        device = segment_embeddings.device

        # Position encoding: normalized frame index (0 to ~1)
        # For T frames: positions = [0, 1/T, 2/T, ..., (T-1)/T]
        positions = torch.arange(T, device=device, dtype=torch.float32) / max(T, 1)
        positions = positions.unsqueeze(1)  # (T, 1)
        x = torch.cat([segment_embeddings, positions], dim=1)  # (T, D+1)

        if self.use_gru:
            x = x.unsqueeze(0)  # (1, T, D+1)
            _, h_n = self.gru(x, self.h0)  # Use learnable h0 instead of zeros
            progress = self.fc(h_n.squeeze())
        else:
            x = x.mean(dim=0)  # Mean pool: (D+1,)
            progress = self.fc(x)
        return progress.squeeze()


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
            self.progress_head = ProgressHead(
                input_dim=output_dimensionality,
                hidden_dim=progress_head_config.get('hidden_dim', 64),
                use_gru=progress_head_config.get('use_gru', True)
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
