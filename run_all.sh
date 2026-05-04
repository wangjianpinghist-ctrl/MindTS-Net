#!/usr/bin/env bash
set -e
python generate_simulated_mindts_data.py --out_dir data --n_train 120 --n_val 30 --n_test 20
python mindts_net_lite.py --mode train --train_dir data/train --val_dir data/val --out_dir outputs
python mindts_net_lite.py --mode infer --test_dir data/test --ckpt outputs/mindts_lite_best.pt --out_json outputs/inference_results.json
