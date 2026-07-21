#!/bin/bash
set -x
source .venv/bin/activate
for f in llama33_70b_sparql_heldout_full qwen25_72b_sparql_heldout_full qwen25_7b_prompted_sparql_heldout_full qwen25_7b_titan_sparql_heldout phi_titan_sparql_heldout phi_titan_sparql_cot_heldout; do
  echo "=== $f ==="
  python baselines/score_sparql_predictions.py \
    --pred baselines/${f}.json \
    --data datasets/TEMPLATE_DISJOINT/CoT/test_heldout.annotated.csv \
    --workers 24 \
    --out baselines/${f}_eval
done
echo ALL_RESCORE_DONE
