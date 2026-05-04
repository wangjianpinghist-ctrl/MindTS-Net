# -*- coding: utf-8 -*-
"""
Generate simulated MindTS-Net data in .npz format.

This creates:
data/train/*.npz
data/val/*.npz
data/test/*.npz

Each sample simulates one round-level multimodal interaction:
text token ids + mel spectrogram + image tensor + emotion label + VA score
+ strategy label + generated guidance token ids + tic state.
"""

from pathlib import Path
import argparse
import numpy as np

EMOTIONS = ["calm", "relaxed", "focused", "shy", "anxious", "tense", "irritable"]
STRATEGIES = [
    "breathing regulation",
    "body scan",
    "emotion labeling",
    "mindfulness attention",
    "relaxation training",
    "awareness guidance",
]

# rough emotion -> strategy mapping for simulated labels
EMO_TO_STRATEGY = {
    0: 5,  # calm -> awareness guidance
    1: 4,  # relaxed -> relaxation training
    2: 3,  # focused -> mindfulness attention
    3: 2,  # shy -> emotion labeling
    4: 0,  # anxious -> breathing regulation
    5: 0,  # tense -> breathing regulation
    6: 2,  # irritable -> emotion labeling
}

# valence/arousal centers, scale roughly [-4, 4]
VA_CENTER = {
    0: (2.4, -1.5),
    1: (2.1, -1.2),
    2: (1.5, 0.1),
    3: (-0.4, 0.8),
    4: (-1.8, 2.5),
    5: (-1.4, 2.8),
    6: (-2.1, 2.3),
}


def make_one_sample(rng, vocab_size=21128, max_len=64, target_len=32, n_mels=80, frames=64):
    emotion = int(rng.integers(0, 7))
    strategy = EMO_TO_STRATEGY[emotion]
    if rng.random() < 0.15:
        strategy = int(rng.integers(0, 6))  # controlled noise

    # Text ids: use random ids but inject emotion-specific repeated tokens.
    text_len = int(rng.integers(12, max_len))
    input_ids = rng.integers(100, vocab_size, size=(text_len,), dtype=np.int64)
    input_ids[0] = 101
    input_ids[-1] = 102
    input_ids[1:4] = 500 + emotion
    attention_mask = np.ones_like(input_ids, dtype=np.int64)

    # Audio mel: emotion-specific offset + random noise.
    mel = rng.normal(0, 0.8, size=(1, n_mels, frames)).astype(np.float32)
    mel += (emotion - 3) * 0.06
    if emotion in [4, 5, 6]:
        mel[:, :, frames//3:frames//2] += 0.5

    # Image: random image tensor with emotion-specific brightness.
    image = rng.normal(0, 0.5, size=(3, 224, 224)).astype(np.float32)
    image += (emotion - 3) * 0.03

    # VA scores.
    vc, ac = VA_CENTER[emotion]
    va = np.array([
        np.clip(rng.normal(vc, 0.35), -4, 4),
        np.clip(rng.normal(ac, 0.35), -4, 4)
    ], dtype=np.float32)

    # Target generation ids: simple strategy-conditioned sequence.
    target_ids = rng.integers(100, vocab_size, size=(target_len,), dtype=np.int64)
    target_ids[0] = 101
    target_ids[1:4] = 900 + strategy
    target_ids[-1] = 102

    # Tic state: higher for anxious/tense/irritable.
    tic_base = 0.2 if emotion in [0, 1, 2] else 0.8
    tic_state = rng.normal(tic_base, 0.15, size=(8,)).astype(np.float32)
    tic_state = np.clip(tic_state, 0, 1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mel": mel,
        "image": image,
        "emotion_label": np.array(emotion, dtype=np.int64),
        "va": va,
        "strategy_label": np.array(strategy, dtype=np.int64),
        "target_ids": target_ids,
        "tic_state": tic_state,
    }


def generate_split(out_dir, split, n, seed):
    rng = np.random.default_rng(seed)
    split_dir = Path(out_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        sample = make_one_sample(rng)
        np.savez_compressed(split_dir / f"{split}_{i:05d}.npz", **sample)

    print(f"Saved {n} samples to {split_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--n_train", type=int, default=120)
    parser.add_argument("--n_val", type=int, default=30)
    parser.add_argument("--n_test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    generate_split(args.out_dir, "train", args.n_train, args.seed)
    generate_split(args.out_dir, "val", args.n_val, args.seed + 1)
    generate_split(args.out_dir, "test", args.n_test, args.seed + 2)


if __name__ == "__main__":
    main()
