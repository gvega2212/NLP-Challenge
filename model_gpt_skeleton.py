# ============================================================
# 04_model_gpt_skeleton.py
# TinyGPT scaffold
# Students should complete the forward pass and generation
# functions by mirroring TinyBERT and adapting to
# autoregressive language modeling.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_modules import DecoderBlock


class TinyGPT(nn.Module):
    """
    Decoder-only transformer with a language-modeling head.

    Main ideas:
    1. Use token embeddings and positional embeddings
    2. Pass the sequence through decoder blocks
    3. Project hidden states to vocabulary logits
    4. If targets are provided, compute next-token loss
    """
    def __init__(self, vocab_size, d_model, context_length, n_layers):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)

        self.blocks = nn.Sequential(*[
            DecoderBlock(d_model, context_length) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        """
        Parameters
        ----------
        idx : torch.Tensor
            Input token ids of shape [B, T]

        targets : torch.Tensor or None
            Target token ids of shape [B, T]

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits of shape [B, T, vocab_size]

        loss : torch.Tensor or None
            Cross-entropy loss for next-token prediction
        """
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate_greedy(self, idx, max_new_tokens=100):
        """
        Greedy decoding:
        always choose the most probable next token.

        Parameters
        ----------
        idx : torch.Tensor
            Starting sequence of shape [B, T]

        max_new_tokens : int
            Number of tokens to generate

        Returns
        -------
        idx : torch.Tensor
            Extended sequence
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :]
            next_token = logits_last.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def generate_temperature(self, idx, max_new_tokens=100, temperature=1.0):
        """
        Sampling with temperature.

        temperature < 1.0 makes output more conservative.
        temperature > 1.0 makes output more random.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def generate_top_k(self, idx, max_new_tokens=100, temperature=1.0, k=5):
        """
        Top-k sampling:
        keep only the k most likely tokens and sample from them.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :] / temperature
            topk_vals, topk_idx = torch.topk(logits_last, k, dim=-1)
            filtered_logits = torch.full_like(logits_last, float("-inf"))
            filtered_logits.scatter_(1, topk_idx, topk_vals)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx