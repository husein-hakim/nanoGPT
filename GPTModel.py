import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class ScaleAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        #scaling for fp16
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)

        self.s_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.s_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.s_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.s_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()

        q = self.s_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.s_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.s_v(x).view(B, T, self.n_head, self.head_dim)

        q = q * self.scale_factor

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.s_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.s_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.s_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.s_fc(x)
        x = self.gelu(x)
        x = self.s_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = ScaleAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.ln = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln(x))
        x = x + self.mlp(self.ln(x))

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tok_embd = nn.Embedding(config.vocab_size, config.n_embd),
            pos_embd = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            norm = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def get_parameters(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.transformer.pos_embd.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok = self.transformer.tok_embd(x)
        pos = self.transformer.pos_embd(pos)
        x = tok + pos
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm(x)

        softcap = 15
        if targets is not None:
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.ndim >=2]
        nondecay_params = [p for n, p in param_dict.items() if p.ndim<2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondeay_params = sum(p.numel() for p in nondecay_params)
        print("number of decay parameters: %.2fM" % (num_decay_params / 1e6))
        print("number of non-decay parameters: %.2fM" % (num_nondeay_params / 1e6))

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat(idx, idx_next, dim=1)
        return idx
    