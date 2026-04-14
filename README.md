# UTEF

This repository contains the code, experiment scripts, and paper-oriented result summaries for a temporal signed trust evaluation workflow built on the `EFGNN-main` codebase.

## Environment

```text
conda create -n utef python=3.8
conda activate utef
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.6.1
```

## Main Entry Points

- `main_link.py`: temporal signed trust prediction and baseline evaluation
- `scripts/tune_ours_bitcoin.py`: tune the proposed model on Bitcoin-OTC and Bitcoin-Alpha
- `scripts/build_bitcoin_paper_tables.py`: build paper-ready aggregate tables

## Paper-Facing Outputs

The repository keeps only compact paper-facing summaries under `outputs/paper_bitcoin/`:

- `bitcoin_compare_tuned_agg.csv`
- `bitcoin_ablation_tuned_agg.csv`
- `paper_summary.md`

Raw datasets, caches, and large intermediate outputs are excluded via `.gitignore`.

