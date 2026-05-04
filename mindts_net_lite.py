# -*- coding: utf-8 -*-
"""
MindTS-Net minimal runnable training/inference code.

This lightweight version does NOT download Chinese-BERT or ResNet weights.
It is for pipeline verification using simulated .npz data.

For paper-level reproduction, replace:
- SimpleTextEncoder -> Chinese-BERT-WWM
- LightVisualEncoder -> ResNet-50
- simulated data -> real synchronized MindTS-MMD round-level data
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CFG:
    vocab_size = 21128
    max_len = 64
    target_len = 32
    hidden = 128
    n_emotions = 7
    n_strategies = 6
    tic_dim = 8
    batch_size = 16
    lr = 1e-3
    epochs = 10
    patience = 4
    num_workers = 0

    lambda_emo = 1.0
    lambda_va = 0.5
    lambda_strategy = 1.0
    lambda_gen = 0.5
    lambda_dst = 0.1


class MindTSMMDNPZDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No npz files found in: {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx], allow_pickle=True)
        return {
            "input_ids": torch.tensor(x["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(x["attention_mask"], dtype=torch.long),
            "mel": torch.tensor(x["mel"], dtype=torch.float32),
            "image": torch.tensor(x["image"], dtype=torch.float32),
            "emotion_label": torch.tensor(int(x["emotion_label"]), dtype=torch.long),
            "va": torch.tensor(x["va"], dtype=torch.float32),
            "strategy_label": torch.tensor(int(x["strategy_label"]), dtype=torch.long),
            "target_ids": torch.tensor(x["target_ids"], dtype=torch.long),
            "tic_state": torch.tensor(x["tic_state"], dtype=torch.float32),
        }


def pad_1d(x, length, pad_value=0):
    if x.numel() >= length:
        return x[:length]
    return F.pad(x, (0, length - x.numel()), value=pad_value)


def collate_fn(batch):
    return {
        "input_ids": torch.stack([pad_1d(b["input_ids"], CFG.max_len, 0) for b in batch]),
        "attention_mask": torch.stack([pad_1d(b["attention_mask"], CFG.max_len, 0) for b in batch]),
        "mel": torch.stack([b["mel"] for b in batch]),
        "image": torch.stack([b["image"] for b in batch]),
        "emotion_label": torch.stack([b["emotion_label"] for b in batch]),
        "va": torch.stack([b["va"] for b in batch]),
        "strategy_label": torch.stack([b["strategy_label"] for b in batch]),
        "target_ids": torch.stack([pad_1d(b["target_ids"], CFG.target_len, 0) for b in batch]),
        "tic_state": torch.stack([b["tic_state"] for b in batch]),
    }


class SimpleTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(CFG.vocab_size, CFG.hidden, padding_idx=0)
        self.gru = nn.GRU(CFG.hidden, CFG.hidden // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(CFG.hidden, CFG.hidden)

    def forward(self, ids, mask):
        x = self.emb(ids)
        y, _ = self.gru(x)
        mask = mask.unsqueeze(-1).float()
        pooled = (y * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(pooled)


class LightAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, CFG.hidden)

    def forward(self, mel):
        return self.proj(self.net(mel).flatten(1))


class LightVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=4, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, CFG.hidden)

    def forward(self, image):
        return self.proj(self.net(image).flatten(1))


class CrossModalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(CFG.hidden, 4, batch_first=True)
        self.ln = nn.LayerNorm(CFG.hidden)

    def forward(self, ht, ha, hv):
        q = ht.unsqueeze(1)
        kv = torch.stack([ha, hv], dim=1)
        z, _ = self.attn(q, kv, kv)
        return self.ln(q + z).squeeze(1)


class EPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = SimpleTextEncoder()
        self.audio = LightAudioEncoder()
        self.visual = LightVisualEncoder()
        self.fusion = CrossModalFusion()
        self.emotion_head = nn.Linear(CFG.hidden, CFG.n_emotions)
        self.va_head = nn.Sequential(nn.Linear(CFG.hidden, CFG.hidden), nn.ReLU(), nn.Linear(CFG.hidden, 2))

    def forward(self, ids, mask, mel, image):
        ht = self.text(ids, mask)
        ha = self.audio(mel)
        hv = self.visual(image)
        z = self.fusion(ht, ha, hv)
        return z, self.emotion_head(z), self.va_head(z)


class MIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Sequential(nn.Linear(CFG.hidden, CFG.hidden), nn.ReLU(), nn.Linear(CFG.hidden, CFG.n_strategies))
        self.emb = nn.Embedding(CFG.n_strategies, CFG.hidden)

    def forward(self, emo_vec, strategy_label=None):
        logits = self.cls(emo_vec)
        pred = logits.argmax(-1)
        sid = strategy_label if strategy_label is not None else pred
        svec = self.emb(sid)
        control = torch.cat([emo_vec, svec], dim=-1)
        return logits, pred, svec, control


class MTG(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(CFG.vocab_size, CFG.hidden, padding_idx=0)
        self.pos = nn.Embedding(CFG.target_len, CFG.hidden)
        self.ctrl = nn.Linear(CFG.hidden * 2, CFG.hidden)
        layer = nn.TransformerDecoderLayer(CFG.hidden, 4, CFG.hidden * 4, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=2)
        self.lm = nn.Linear(CFG.hidden, CFG.vocab_size)

    def forward(self, target_ids, control):
        b, t = target_ids.shape
        pos = torch.arange(t, device=target_ids.device).unsqueeze(0).expand(b, t)
        tgt = self.tok(target_ids) + self.pos(pos)
        memory = self.ctrl(control).unsqueeze(1)
        mask = torch.triu(torch.ones(t, t, device=target_ids.device), diagonal=1).bool()
        y = self.dec(tgt=tgt, memory=memory, tgt_mask=mask)
        return self.lm(y)

    @torch.no_grad()
    def generate(self, control, max_new_tokens=24, bos_id=101, eos_id=102):
        ids = torch.full((control.size(0), 1), bos_id, dtype=torch.long, device=control.device)
        for _ in range(max_new_tokens):
            logits = self.forward(ids, control)
            nxt = logits[:, -1:].argmax(-1)
            ids = torch.cat([ids, nxt], dim=1)
            if (nxt == eos_id).all():
                break
        return ids


class DST(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(CFG.hidden * 2 + CFG.tic_dim, CFG.hidden, batch_first=True)
        self.effect = nn.Linear(CFG.hidden, 1)

    def forward(self, emo_vec, svec, tic):
        x = torch.cat([emo_vec, svec, tic], -1).unsqueeze(1)
        h, _ = self.gru(x)
        return torch.sigmoid(self.effect(h[:, -1])).squeeze(-1)


class MindTSNetLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.epm = EPM()
        self.mip = MIP()
        self.mtg = MTG()
        self.dst = DST()

    def forward(self, batch):
        emo_vec, emo_logits, va = self.epm(batch["input_ids"], batch["attention_mask"], batch["mel"], batch["image"])
        st_logits, st_pred, svec, control = self.mip(emo_vec, batch["strategy_label"])
        lm_logits = self.mtg(batch["target_ids"][:, :-1], control)
        effect = self.dst(emo_vec, svec, batch["tic_state"])
        return {
            "emo_logits": emo_logits,
            "va": va,
            "strategy_logits": st_logits,
            "strategy_pred": st_pred,
            "lm_logits": lm_logits,
            "effect": effect,
            "control": control,
        }

    @torch.no_grad()
    def infer(self, batch):
        emo_vec, emo_logits, va = self.epm(batch["input_ids"], batch["attention_mask"], batch["mel"], batch["image"])
        st_logits, st_pred, svec, control = self.mip(emo_vec, None)
        gen = self.mtg.generate(control)
        effect = self.dst(emo_vec, svec, batch["tic_state"])
        return emo_logits.argmax(-1), va, st_pred, gen, effect


def loss_fn(batch, out):
    emo = F.cross_entropy(out["emo_logits"], batch["emotion_label"])
    va = F.mse_loss(out["va"], batch["va"])
    st = F.cross_entropy(out["strategy_logits"], batch["strategy_label"])
    gen = F.cross_entropy(
        out["lm_logits"].reshape(-1, CFG.vocab_size),
        batch["target_ids"][:, 1:].reshape(-1),
        ignore_index=0,
    )
    # synthetic effect target: higher tic state should predict higher intervention need
    effect_target = batch["tic_state"].mean(-1)
    dst = F.mse_loss(out["effect"], effect_target)
    total = CFG.lambda_emo * emo + CFG.lambda_va * va + CFG.lambda_strategy * st + CFG.lambda_gen * gen + CFG.lambda_dst * dst
    return total, {
        "loss": total.item(),
        "emo_loss": emo.item(),
        "va_loss": va.item(),
        "strategy_loss": st.item(),
        "gen_loss": gen.item(),
        "dst_loss": dst.item(),
    }


def acc(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_epoch(model, loader, opt, device, train=True):
    model.train(train)
    logs = []
    for batch in loader:
        batch = move(batch, device)
        if train:
            opt.zero_grad()
        out = model(batch)
        loss, log = loss_fn(batch, out)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        log["emo_acc"] = acc(out["emo_logits"], batch["emotion_label"])
        log["strategy_acc"] = acc(out["strategy_logits"], batch["strategy_label"])
        logs.append(log)
    return {k: float(np.mean([d[k] for d in logs])) for k in logs[0]}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(MindTSMMDNPZDataset(args.train_dir), batch_size=CFG.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MindTSMMDNPZDataset(args.val_dir), batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MindTSNetLite().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = math.inf
    bad = 0
    history = []

    for ep in range(1, CFG.epochs + 1):
        tr = run_epoch(model, train_loader, opt, device, train=True)
        with torch.no_grad():
            va = run_epoch(model, val_loader, opt, device, train=False)
        history.append({"epoch": ep, "train": tr, "val": va})
        print(f"Epoch {ep:02d} | train loss={tr['loss']:.4f}, emo_acc={tr['emo_acc']:.3f}, strategy_acc={tr['strategy_acc']:.3f} | val loss={va['loss']:.4f}, emo_acc={va['emo_acc']:.3f}, strategy_acc={va['strategy_acc']:.3f}")

        if va["loss"] < best:
            best = va["loss"]
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val": va}, out_dir / "mindts_lite_best.pt")
        else:
            bad += 1
            if bad >= CFG.patience:
                print("Early stopping.")
                break

    (out_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(MindTSMMDNPZDataset(args.test_dir), batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = MindTSNetLite().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = move(batch, device)
            emo, va, strategy, gen, effect = model.infer(batch)
            results.append({
                "index": i,
                "emotion_id": int(emo[0].cpu()),
                "valence": float(va[0, 0].cpu()),
                "arousal": float(va[0, 1].cpu()),
                "strategy_id": int(strategy[0].cpu()),
                "generated_token_ids": gen[0].cpu().tolist(),
                "effect_score": float(effect[0].cpu()),
            })

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved inference results to {args.out_json}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "infer"], required=True)
    p.add_argument("--train_dir", default="data/train")
    p.add_argument("--val_dir", default="data/val")
    p.add_argument("--test_dir", default="data/test")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--ckpt", default="outputs/mindts_lite_best.pt")
    p.add_argument("--out_json", default="outputs/inference_results.json")
    args = p.parse_args()

    if args.mode == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
