# ============================================================
# 05_training_utils_and_demos.py
# Full helpers and demos for BERT and BART.
# GPT demo intentionally left empty.
# ============================================================

import torch

from batching import get_classification_batch, get_seq2seq_batch, get_lm_batch
from setup import (
    context_length,
    d_model,
    decode,
    device,
    encode,
    learning_rate,
    n_layers,
    vocab_size,
)
from models_bert_bart import TinyBERT, TinyBART
from model_gpt_skeleton import TinyGPT


@torch.no_grad()
def estimate_bert_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        accs = []
        for _ in range(eval_iters):
            x, y = get_classification_batch(split)
            logits, loss = model(x, y)
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)
        out[split] = {
            "loss": sum(losses) / len(losses),
            "acc": sum(accs) / len(accs)
        }
    model.train()
    return out


@torch.no_grad()
def estimate_bart_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            src, tgt_in, tgt_out = get_seq2seq_batch(split)
            _, loss = model(src, tgt_in, tgt_out)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def estimate_gpt_loss(model, eval_iters=20):
    """
    Students may use this function once TinyGPT is complete.
    """
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_lm_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def run_bert_demo(steps=101):
    print("\n" + "=" * 60)
    print("BERT-LIKE DEMO")
    print("=" * 60)

    bert_model = TinyBERT(
        vocab_size=vocab_size,
        d_model=d_model,
        context_length=context_length,
        n_layers=n_layers,
        n_classes=2
    ).to(device)

    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=learning_rate)

    for step in range(steps):
        if step % 50 == 0:
            stats = estimate_bert_loss(bert_model, eval_iters=10)
            print(
                f"Step {step:3d} | "
                f"train loss: {stats['train']['loss']:.4f} | "
                f"train acc: {stats['train']['acc']:.4f} | "
                f"val loss: {stats['val']['loss']:.4f} | "
                f"val acc: {stats['val']['acc']:.4f}"
            )

        xb, yb = get_classification_batch("train")
        logits, loss = bert_model(xb, yb)

        bert_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        bert_optimizer.step()


def run_bart_demo(steps=101):
    print("\n" + "=" * 60)
    print("BART-LIKE DEMO")
    print("=" * 60)

    bart_model = TinyBART(
        vocab_size=vocab_size,
        d_model=d_model,
        context_length=context_length,
        n_layers=n_layers
    ).to(device)

    bart_optimizer = torch.optim.Adam(bart_model.parameters(), lr=learning_rate)

    for step in range(steps):
        if step % 50 == 0:
            losses = estimate_bart_loss(bart_model, eval_iters=10)
            print(
                f"Step {step:3d} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f}"
            )

        src, tgt_in, tgt_out = get_seq2seq_batch("train")
        logits, loss = bart_model(src, tgt_in, tgt_out)

        bart_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        bart_optimizer.step()


def run_gpt_demo(steps=101):
    print("\n" + "=" * 60)
    print("GPT-LIKE DEMO")
    print("=" * 60)

    gpt_model = TinyGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        context_length=context_length,
        n_layers=n_layers
    ).to(device)

    gpt_optimizer = torch.optim.Adam(gpt_model.parameters(), lr=learning_rate)

    for step in range(steps):
        x, y = get_lm_batch("train")
        logits, loss = gpt_model(x, y)

        gpt_optimizer.zero_grad(set_to_none=True)
        if loss is not None:
            loss.backward()
            gpt_optimizer.step()

        if step % 50 == 0:
            stats = estimate_gpt_loss(gpt_model, eval_iters=10)
            print(
                f"Step {step:3d} | "
                f"train loss: {stats['train']:.4f} | "
                f"val loss: {stats['val']:.4f}"
            )

    stats = estimate_gpt_loss(gpt_model, eval_iters=20)
    print(
        f"Final GPT loss | train: {stats['train']:.4f} | "
        f"val: {stats['val']:.4f}"
    )

    prompt = "Natural language "
    idx0 = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    gpt_model.eval()
    with torch.no_grad():
        greedy_ids = gpt_model.generate_greedy(idx0.clone(), max_new_tokens=180)
        temp_07_ids = gpt_model.generate_temperature(
            idx0.clone(),
            max_new_tokens=180,
            temperature=0.7,
        )
        temp_13_ids = gpt_model.generate_temperature(
            idx0.clone(),
            max_new_tokens=180,
            temperature=1.3,
        )
        topk_5_ids = gpt_model.generate_top_k(
            idx0.clone(),
            max_new_tokens=180,
            temperature=1.0,
            k=5,
        )
        topk_10_ids = gpt_model.generate_top_k(
            idx0.clone(),
            max_new_tokens=180,
            temperature=1.0,
            k=10,
        )

    print("\nPrompt:")
    print(prompt)

    print("\nGreedy:")
    print(decode(greedy_ids[0].tolist()))

    print("\nTemperature (0.7):")
    print(decode(temp_07_ids[0].tolist()))

    print("\nTemperature (1.3):")
    print(decode(temp_13_ids[0].tolist()))

    print("\nTop-k (k=5):")
    print(decode(topk_5_ids[0].tolist()))

    print("\nTop-k (k=10):")
    print(decode(topk_10_ids[0].tolist()))

    gpt_model.train()


def run_all_demos(steps=101):
    run_bert_demo(steps=steps)
    run_bart_demo(steps=steps)
    run_gpt_demo(steps=steps)

