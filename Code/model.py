import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqCNNBiLSTM(nn.Module):

    def __init__(
        self,
        seq_feat_dim: int,
        filter_sizes=(1, 2, 3, 4, 5, 6),
        num_filters: int = 32,
        lstm_hidden: int = 128,
        out_dim: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.filter_sizes = list(filter_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=seq_feat_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in self.filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=num_filters * len(self.filter_sizes),
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        conv_outs = []
        out_lens = []
        for conv in self.convs:
            o = conv(x)
            o = F.relu(o)
            conv_outs.append(o)
            out_lens.append(o.size(-1))

        min_len = min(out_lens)
        conv_outs = [o[..., :min_len] for o in conv_outs]

        cnn_feat = torch.cat(conv_outs, dim=1)
        cnn_feat = self.dropout(cnn_feat)
        cnn_feat = cnn_feat.transpose(1, 2)

        lstm_out, _ = self.lstm(cnn_feat)
        feat, _ = torch.max(lstm_out, dim=1)
        feat = self.dropout(feat)
        return self.proj(feat)


class FeatTransformerBranch(nn.Module):

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_embed = self.input_proj(x)
        x_embed = x_embed.unsqueeze(1)

        enc_out = self.transformer_encoder(x_embed)
        feat_token = enc_out[:, 0, :]

        return self.out_proj(feat_token)



class PepContrastNet(nn.Module):

    def __init__(
        self,
        seq_feat_dim: int,
        mlp_in_dim: int,
        seq_out_dim: int = 64,
        feat_out_dim: int = 64,
    ):
        super().__init__()
        self.seq_encoder = SeqCNNBiLSTM(
            seq_feat_dim=seq_feat_dim,
            out_dim=seq_out_dim,
        )
        self.feat_encoder = FeatTransformerBranch(
            in_dim=mlp_in_dim,
            out_dim=feat_out_dim,
        )
        fused_dim = seq_out_dim + feat_out_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.2),
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, enc: torch.Tensor, feat: torch.Tensor):
        z_seq = self.seq_encoder(enc)
        z_feat = self.feat_encoder(feat)
        fused = torch.cat([z_seq, z_feat], dim=-1)
        logits = self.classifier(fused)
        return logits, z_seq, z_feat, fused


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    device = features.device
    features = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)
    batch_size = features.size(0)

    mask = torch.eq(labels, labels.T).float().to(device)
    logits = torch.div(torch.matmul(features, features.T), temperature)

    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

    loss = -mean_log_prob_pos.mean()
    return loss

