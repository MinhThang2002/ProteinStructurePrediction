# Protein Angle Prediction (Attention-based)

Concise guide for setup, training, inference, and evaluation with SidechainNet.

## Setup

### Create environment and install library:

- python3 -m venv .venv
- source .venv/bin/activate
- Python 3.10+, install deps: `pip install -r requirements.txt`
- Use local data to avoid downloads: `export SCN_DATA_PATH=./sidechainnet_data`
- On macOS keep `--num_workers 0` (collate pickling).

### Cài torch phù hợp máy bạn, ví dụ CPU:

pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu

## Train

**Light baseline (PSSM input, CPU-friendly):**

```bash
python main.py --train True --mode pssms --integer_sequence False \
  --epoch 10 --batch 4 \
  --d_in 49 --d_hidden 512 --dim 256 --d_embedding 32 \
  --n_heads 8 --head_dim 64 \
  --learning_rate 1e-3 --weight_decay 0 \
  --num_workers 0 --model_save_path ./models
```

**Current best config:** HuberLoss + projection dropout, EMA, cosine per-batch, grad accumulation.

```bash
python main.py --train True --mode pssms --integer_sequence False \
  --epoch 10 --batch 8 \
  --d_in 49 --d_hidden 768 --dim 384 --d_embedding 48 \
  --n_heads 12 --head_dim 64 --attn_dropout 0.1 \
  --learning_rate 1e-4 --min_lr 1e-5 \
  --scheduler cosine --scheduler_step batch --warmup_epochs 2 \
  --grad_accum_steps 2 --ema_decay 0.999 \
  --weight_decay 1e-4 \
  --num_workers 0 --model_save_path ./models
```

To continue from the latest checkpoint: add `--model_load_path ./models/model_weights.pth`.

**Sequence-only (reference):**

```bash
python main.py --train True --mode seqs --integer_sequence True \
  --epoch 10 --batch 4 \
  --d_in 21 --d_hidden 512 --dim 256 --d_embedding 32 \
  --n_heads 8 --head_dim 64 \
  --learning_rate 1e-3 --num_workers 0 --model_save_path ./models
```

## Inference

- Single sample: set `--idx N` (N ≥ 0).
- Auto loop idx 0–19: set `--idx -1` (skips any sample that fails due to missing residues).

```bash
python main.py --train False --mode pssms --integer_sequence False \
  --model_load_path ./models/model_weights.pth --idx -1 --complete_structures_only True \
  --d_in 49 --d_hidden 768 --dim 384 --d_embedding 48 \
  --n_heads 12 --head_dim 64 --attn_dropout 0.1 \
  --batch 8 --num_workers 0
```

Outputs: `plots/{idx}_pred.pdb`, `plots/{idx}_true.pdb`, `plots/{idx}_compare.html`.

## Evaluation

Print RMSE for splits and append to `models/metrics_report.txt`:

```bash
python evaluate.py --mode pssms --integer_sequence False \
  --model_load_path ./models/model_weights.pth \
  --d_in 49 --d_hidden 768 --dim 384 --d_embedding 48 \
  --n_heads 12 --head_dim 64 --attn_dropout 0.1 \
  --batch 8 --num_workers 0
```

If memory is tight, lower `--batch` and/or use `--complete_structures_only True`.

## Logs

- Best metric: `models/best_metric.txt`
- Training runs: `models/train_log.txt`
- Eval summary: `models/metrics_report.txt`

## Notes

- Loss is SmoothL1 (Huber) on sin/cos targets; RMSE is reported by `validation`.
- Best checkpoint uses EMA weights if `--ema_decay > 0`.
