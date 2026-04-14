# Reproduce the "paper_best" run with the exact command used in this workspace.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\reproduce_paper_best.ps1

$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."

Write-Host "[1/4] Clean Node2Vec cache for seeds 1..5 to avoid stale artifacts..."
$cachePattern = "data\processed\bitcoin_otc_node2vec_dim64_walk20_ctx10_steps1000_ep30_lr0.01_seed*.pt"
Get-ChildItem $cachePattern -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host "[2/4] Run unified 5-seed experiment (compare + ablation in one run)..."
python main_link.py `
  --seeds 5 `
  --feature-type node2vec `
  --classification-loss bce `
  --models ours,gatrust,trustguard,guardian,ours_nounc,ours_1hop,ours_notemp `
  --epochs 130 `
  --patience 55 `
  --num-hops 3 `
  --lambda-uncertainty 0.08 `
  --lambda-ranking 0.06 `
  --u-alpha 0.6 `
  --out-csv outputs\reproduce_paper_best.csv

Write-Host "[3/4] Validate metric logical consistency..."
python scripts\validate_experiment_metrics.py outputs\reproduce_paper_best.csv

Write-Host "[4/4] Compare aggregated result against paper_best_agg.csv (max absolute diff)..."
@'
import csv

def read(path):
    with open(path, encoding='utf-8') as f:
        return {r['model']: r for r in csv.DictReader(f)}

a = read('outputs/paper_best_agg.csv')
b = read('outputs/reproduce_paper_best_agg.csv')
keys = [k for k in a['ours'].keys() if k != 'model']
max_diff = 0.0
max_model = None
max_key = None
for m in a:
    for k in keys:
        d = abs(float(a[m][k]) - float(b[m][k]))
        if d > max_diff:
            max_diff = d
            max_model = m
            max_key = k
print(f"max_abs_diff={max_diff:.10f} at model={max_model}, metric={max_key}")
print("reference: outputs/paper_best_agg.csv")
print("current:   outputs/reproduce_paper_best_agg.csv")
'@ | python -

Write-Host "Done."
