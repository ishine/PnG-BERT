import torch
import torch.nn as nn
from .modules import Transformer, gelu, LayerNorm

class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.dim, cfg.dim)
        self.activ2 = gelu
        self.norm = LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.dim, 2)
        # decoder is shared with embedding layer
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf
        