import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import json

from utils import get_model, print_rank
from transformers import AutoModel, AutoConfig


class DataScorerModel(nn.Module):
    def __init__(self, args, device, base_model_path, bias=False, encoding="mean", head_type="linear"):
        super().__init__()
        self.args = args
        self.base_model_path = base_model_path
        self.config = AutoConfig.from_pretrained(base_model_path)
        self.base_model = get_model(args, device, base_model_path, self.config, model_cls=AutoModel)
        # self.head = nn.Linear(self.config.hidden_size, 1, bias=bias)
        self.score_head = nn.Linear(self.config.hidden_size, 1, bias=bias, device=device, dtype=next(self.base_model.parameters()).dtype)

        self.bias = bias
        self.head_type = head_type
        self.encoding = encoding
        
        print_rank(f"Data Scorer | Bias: {bias}, Encoding: {encoding}, Head type: {head_type}")

    def _forward(self, input_ids, attention_mask, pos):
        h = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)["last_hidden_state"]
        if self.encoding == "mean":
            mask = (torch.arange(h.size(1), device=h.device)[None, :] <= pos[:, None]).to(h.dtype)
            origin_dtype = h.dtype
            h = h.float()
            h = torch.sum(h * mask[:, :, None], dim=1) / mask.sum(dim=1)[:, None]
            h = h.to(origin_dtype)            
        elif self.encoding == "last":
            h = torch.gather(h, 1, pos[:, None, None].expand(-1, -1, h.size(-1))).squeeze()
        elif self.encoding == "first":
            h = h[:, 0]
        else:
            raise ValueError("encoding should be mean or last")
        
        if self.head_type == "linear":
            # s = self.head(h).squeeze()
            s = self.score_head(h).squeeze()
            
        elif self.head_type == "sigmoid":
            s = torch.sigmoid(self.head(h).squeeze())
        else:
            raise ValueError("score_head should be linear or sigmoid")

        if s.dim() == 0:
            s = s.unsqueeze(0)

        return s.float()

    def forward(self, input_ids, attention_mask, pos, labels):
        s = self._forward(input_ids, attention_mask, pos)
        labels = labels.to(s.dtype)  # (b, s), B
        loss = torch.nn.functional.mse_loss(s, labels)
        return loss

    def save_pretrained(self, save_dir, **kawrgs):
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump({
                "base_model_path": self.base_model_path.replace(os.getcwd(), ""),
                "bias": self.bias,
                "encoding": self.encoding
            }, f, indent=4)
        torch.save(self.state_dict(), os.path.join(
            save_dir, "data_scorer_model.pt"))
    
    def inference(self, input_ids, attention_mask, pos, labels=None):
        return self._forward(input_ids, attention_mask, pos)
