
import torch
import torch.nn as nn
import math
from model import Model
import numpy as np

def get_resized_pos_emb(pe, config):
    '''
    pe is positional embedding of size: [Seq_Len, Model_Dim]
    '''
    
    # if mode is None:
    #     return pe
    
    seqlen_s, dim_s = pe.shape
    hw_s = int(np.sqrt(seqlen_s))
    assert hw_s*hw_s == seqlen_s, "Source positional embedding is not square"
    
    seqlen_t, dim_t = config["max_seq_len"], config["transformer_dim"]
    hw_t = int(np.sqrt(seqlen_t))
    assert hw_t*hw_t == seqlen_t, "Target positional embedding is not square"
    
    assert dim_s == dim_t, "Model Dimension of source and target does not match"
    
    repeats = hw_t//hw_s
    assert hw_s*repeats == hw_t, "Target positional embedding is not Integer Multiple of Source"

    pe_ = pe.view(hw_s, 1, hw_s, 1, dim_t)
    pe_ = torch.cat([pe_]*repeats, dim=3)
    pe_ = torch.cat([pe_]*repeats, dim=1)
    return pe_.view(-1, dim_t)

def pooling(inp, mode):
    if mode == "CLS":
        pooled = inp[:, 0, :]
    elif mode == "MEAN":
        pooled = inp.mean(dim = 1)
    else:
        raise Exception()
    return pooled

def append_cls(inp, mask, vocab_size):
    batch_size = inp.size(0)
    cls_id = ((vocab_size - 1) * torch.ones(batch_size, dtype = torch.long, device = inp.device)).long()
    cls_mask = torch.ones(batch_size, dtype = torch.float, device = mask.device)
    inp = torch.cat([cls_id[:, None], inp[:, :-1]], dim = -1)
    mask = torch.cat([cls_mask[:, None], mask[:, :-1]], dim = -1)
    return inp, mask

class SCHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["transformer_hidden_dim"], config["num_classes"])
        )

    def forward(self, inp):
        seq_score = self.mlpblock(pooling(inp, self.pooling_mode))
        return seq_score

class ModelForSC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]
        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]

        if config["seed_model"] >= 0:
            print(f"Model Seed !!! {config['seed_model']}")
            torch.manual_seed(config["seed_model"])
            
        self.model = Model(config)

        self.seq_classifer = SCHead(config)

        if config["pretrained_init"]:
            ckpt = torch.load(config["pretrained_init"], map_location="cpu")["model_state_dict"]
            
            #### This makes torch.compile(model) weights compatible with non-compiled weights 
            if list(ckpt.keys())[0].startswith('_orig_mod.'):
                new_ckpt = {}
                for k, v in ckpt.items():
                    new_ckpt[k[10:]] = v
                ckpt = new_ckpt
            
            ### Resize positional embedding
            pe = ckpt["model.embeddings.position_embeddings.weight"]
            pe = get_resized_pos_emb(pe, config)
            ckpt["model.embeddings.position_embeddings.weight"] = pe

            self.load_state_dict(ckpt)
            print("Pretrained Loaded !!!")



    def forward(self, input_ids_0, mask_0, label):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            if self.pooling_mode == "CLS":
                input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)

            token_out = self.model(input_ids_0, mask_0)
            seq_scores = self.seq_classifer(token_out)

            seq_loss = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
            seq_accu = (seq_scores.argmax(dim = -1) == label).to(torch.float32)
            outputs = {}
            outputs["loss"] = seq_loss
            outputs["accu"] = seq_accu

        return outputs

class SCHeadDual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"] * 4, config["transformer_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["transformer_hidden_dim"], config["num_classes"])
        )

    def forward(self, inp_0, inp_1):
        X_0 = pooling(inp_0, self.pooling_mode)
        X_1 = pooling(inp_1, self.pooling_mode)
        seq_score = self.mlpblock(torch.cat([X_0, X_1, X_0 * X_1, X_0 - X_1], dim = -1))
        return seq_score

class ModelForSCDual(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]
        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]
        
        self.model = Model(config)

        self.seq_classifer = SCHeadDual(config)

    def forward(self, input_ids_0, input_ids_1, mask_0, mask_1, label):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            if self.pooling_mode == "CLS":
                input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)
                input_ids_1, mask_1 = append_cls(input_ids_1, mask_1, self.vocab_size)

            token_out_0 = self.model(input_ids_0, mask_0)
            token_out_1 = self.model(input_ids_1, mask_1)
            seq_scores = self.seq_classifer(token_out_0, token_out_1)

            seq_loss = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
            seq_accu = (seq_scores.argmax(dim = -1) == label).to(torch.float32)
            outputs = {}
            outputs["loss"] = seq_loss
            outputs["accu"] = seq_accu

        return outputs
