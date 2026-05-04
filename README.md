# MindTS-Net Minimal Runnable Pipeline

This folder contains a minimal simulation-based pipeline for MindTS-Net.

## 1. Generate simulated data

```bash
python generate_simulated_mindts_data.py --out_dir data --n_train 120 --n_val 30 --n_test 20
```

Generated structure:

```text
data/
  train/*.npz
  val/*.npz
  test/*.npz
```

Each `.npz` sample contains:

```text
input_ids
attention_mask
mel
image
emotion_label
va
strategy_label
target_ids
tic_state
```

## 2. Train the lightweight model

```bash
python mindts_net_lite.py --mode train --train_dir data/train --val_dir data/val --out_dir outputs
```

Main outputs:

```text
outputs/mindts_lite_best.pt
outputs/training_history.json
```

## 3. Run inference

```bash
python mindts_net_lite.py --mode infer --test_dir data/test --ckpt outputs/mindts_lite_best.pt --out_json outputs/inference_results.json
```

## 4. What this version is for

This is not the full clinical model. It is a runnable engineering scaffold used to verify:

- data loading
- multimodal input format
- EPM -> MIP -> MTG -> DST pipeline
- multitask loss
- checkpoint saving
- inference output

For paper-level reproduction, replace the lightweight encoders with:

- Chinese-BERT-WWM for text
- CNN + BiGRU for audio
- ResNet-50 for image
- Cross-Modal Transformer fusion
- Prefix-injected Transformer generator
- GRU + Attention DST
