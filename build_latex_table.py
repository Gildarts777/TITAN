#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the paper's main results LaTeX table directly from the
*_eval_report.json files produced by evaluate_titan.py -- no numbers
retyped by hand, so the table can't silently drift from what was actually
measured.

Columns: Path EM and Entity F1 (with 95% bootstrap CIs) on test_heldout
(the leakage-free split, the paper's real claim) and test_iid (seen
templates, in-distribution sanity check) where available. Baselines were
only run on test_heldout (test_iid has no meaning for a non-fine-tuned
prompted model), so their test_iid cell is left blank.

Run: python3 build_latex_table.py > results_table.tex
"""

from __future__ import annotations

import json
from typing import Optional

# (row label, group, heldout report path, iid report path or None)
ROWS = [
    ("Llama-3.3-70B (prompted)", "baseline",
     "baselines/llama33_70b_heldout_full_v3_eval_report.json", None),
    ("Qwen2.5-72B (prompted)", "baseline",
     "baselines/qwen25_72b_heldout_full_v3_eval_report.json", None),
    ("GPT-OSS-120B (prompted)", "baseline",
     "baselines/gptoss_120b_heldout_full_v3_eval_report.json", None),
    ("Phi-3.5-mini NoCoT (fine-tuned, 3.8B)", "finetuned",
     "baselines/phi_titan_nocot_v2_heldout_eval_report.json",
     "baselines/phi_titan_nocot_v2_iid_eval_report.json"),
    ("Phi-3.5-mini CoT (fine-tuned, 3.8B)", "finetuned",
     "baselines/phi_titan_cot_v2_heldout_eval_report.json",
     "baselines/phi_titan_cot_v2_iid_eval_report.json"),
    ("Qwen2.5-7B NoCoT (fine-tuned)", "finetuned",
     "baselines/qwen25_7b_titan_nocot_heldout_eval_report.json",
     "baselines/qwen25_7b_titan_nocot_iid_eval_report.json"),
    ("Qwen2.5-7B CoT (fine-tuned)", "finetuned",
     "baselines/qwen25_7b_titan_cot_heldout_eval_report.json",
     "baselines/qwen25_7b_titan_cot_iid_eval_report.json"),
]


def load(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["overall"]
    except FileNotFoundError:
        return None


def cell(overall: Optional[dict], key: str) -> str:
    if overall is None:
        return "--"
    val = overall[key]
    lo, hi = overall[f"{key}_ci"]
    return f"{val:.3f} \\scriptsize{{[{lo:.3f}, {hi:.3f}]}}"


def main() -> None:
    best_em = best_f1 = -1.0
    resolved = []
    for label, group, held_path, iid_path in ROWS:
        held = load(held_path)
        iid = load(iid_path)
        resolved.append((label, group, held, iid))
        if held:
            best_em = max(best_em, held["path_em"])
            best_f1 = max(best_f1, held["entity_f1"])

    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \multicolumn{2}{c}{\textbf{test\_heldout} (leakage-free)} "
          r"& \multicolumn{2}{c}{\textbf{test\_iid} (seen templates)} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    print(r" & Path EM & Entity F1 & Path EM & Entity F1 \\")
    print(r"\midrule")

    prev_group = None
    for label, group, held, iid in resolved:
        if prev_group == "baseline" and group == "finetuned":
            print(r"\midrule")
        prev_group = group

        em_str = cell(held, "path_em")
        f1_str = cell(held, "entity_f1")
        if held and held["path_em"] == best_em:
            em_str = r"\textbf{" + em_str + "}"
        if held and held["entity_f1"] == best_f1:
            f1_str = r"\textbf{" + f1_str + "}"

        iid_em = cell(iid, "path_em") if iid else "--"
        iid_f1 = cell(iid, "entity_f1") if iid else "--"

        print(f"{label} & {em_str} & {f1_str} & {iid_em} & {iid_f1} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Path exact-match (EM) and entity-level F1 (with 95\% bootstrap CIs) "
          r"on the template-disjoint TITAN split. \emph{test\_heldout} uses templates never "
          r"seen during training (the leakage-free, real generalization test); "
          r"\emph{test\_iid} shares templates with training (in-distribution control, only "
          r"meaningful for fine-tuned models). Bold marks the best result per column on "
          r"test\_heldout. Prompted baselines use the v3 system prompt (typed graph schema "
          r"+ 3 worked examples + 12 executable-verified few-shot examples).}")
    print(r"\label{tab:main-results}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
