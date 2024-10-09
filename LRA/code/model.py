import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from attention import Attention


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["transformer_dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(
            config["vocab_size"], config["embedding_dim"]
        )
        torch.nn.init.normal_(self.word_embeddings.weight, std=0.02)

        self.position_embeddings = nn.Embedding(
            config["max_seq_len"], config["embedding_dim"]
        )
        torch.nn.init.normal_(self.position_embeddings.weight, std=0.02)

        self.dropout = torch.nn.Dropout(p=config["embedding_dropout"])

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device=device)[:, np.newaxis]
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device)
            * -(math.log(10000.0) / self.dim)
        )
        pos_embed = torch.stack(
            [torch.sin(position * div_term), torch.cos(position * div_term)], -1
        ).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)[
            None, :
        ].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p=config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p=config["mlp_dropout"]),
            nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
            torch.nn.Dropout(p=config["dropout_prob"]),
        )

    def forward(self, X, mask):
        X = self.dropout1(self.mha(self.norm1(X), mask)) + X
        X = self.mlpblock(self.norm2(X)) + X
        return X


class ButterflyTransformer(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.layer_index = config["layer_index"]
        self.butterfly_radix = config["butterfly_radix"]
        self.max_seq_len = config["max_seq_len"]

        assert (
            self.max_seq_len % self.butterfly_radix == 0
        ), "Sequence Length must be divisible by attention block length"

        def log_base(a, base):
            return np.log(a) / np.log(base)

        ### total number of layers to complete mixing
        num_layers = int(np.ceil(log_base(self.max_seq_len, base=self.butterfly_radix)))
        butterfly_layer_index = (
            self.layer_index % num_layers
        )  ## repeated index in blocks (for layers)
        stride = self.butterfly_radix**butterfly_layer_index
        if stride * self.butterfly_radix > self.max_seq_len:
            stride = int(np.ceil(self.max_seq_len / self.butterfly_radix))
        self.butterfly_stride = stride

    def forward(self, X, mask):
        ## X has shape -> [Batch_size, Seq_len, Model_dim]
        bs, seq_len, model_dim = X.shape
        X = (
            X.view(bs, -1, self.butterfly_radix, self.butterfly_stride, model_dim)
            .transpose(2, 3)
            .contiguous()
            .view(-1, self.butterfly_radix, model_dim)
        )
        mask = (
            mask.view(bs, -1, self.butterfly_radix, self.butterfly_stride)
            .transpose(2, 3)
            .contiguous()
            .view(-1, self.butterfly_radix)
        )

        X = super().forward(X, mask)
        X = (
            X.view(bs, -1, self.butterfly_stride, self.butterfly_radix, model_dim)
            .transpose(2, 3)
            .contiguous()
            .view(bs, seq_len, model_dim)
        )
        return X


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]

        self.embeddings = Embeddings(config)

        if self.tied_weights:
            self.transformer = Transformer(config)
        elif config["attn_type"].startswith("butterfly"):
            __backup_attn_type = config["attn_type"]
            config["attn_type"] = config["sub_attn_type"]
            for idx in range(self.num_layers):
                config["layer_index"] = idx
                setattr(self, f"transformer_{idx}", ButterflyTransformer(config))
            config["attn_type"] = __backup_attn_type
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", Transformer(config))

        self.norm = nn.LayerNorm(config["transformer_dim"])

    def forward(self, input_ids, mask=None):
        X = self.embeddings(input_ids)
        ## X has shape -> [Batch_size, Seq_len, Model_dim]

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            for idx in range(self.num_layers):
                X = self.transformer(X, mask)
        else:
            for idx in range(self.num_layers):
                X = getattr(self, f"transformer_{idx}")(X, mask)

        X = self.norm(X) * mask[:, :, None]

        return X
