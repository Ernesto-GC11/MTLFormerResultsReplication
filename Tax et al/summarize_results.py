"""
Summarise the output of evaluate_next_activity_and_time.py and
evaluate_suffix_and_remaining_time.py for every seed of a dataset.

  python summarize_results.py --dataset helpdesk --prefixes 2 4 6
  python summarize_results.py --dataset bpi_12_w --prefixes 2 10 20

Reads  <results_dir>/<dataset>/seed*/next_activity_and_time_<dataset>.csv
       <results_dir>/<dataset>/seed*/suffix_and_remaining_time_<dataset>.csv
Writes <results_dir>/<dataset>/summary.csv  (metrics x seeds, plus mean and std)
       <results_dir>/<dataset>/remaining_time_mae_by_prefix.csv  (data behind Fig. 3)

Every metric is reported in two ways:
  <metric>       pooled over all evaluated prefixes (how the LSTM paper reports "All")
  <metric>_kavg  computed per prefix length first and then averaged over prefix lengths
                 (how MTLFormer's main.py reports its "Average ... across all prefixes")

Metrics (times in days, as in the paper):
  next activity   accuracy, precision, recall, f1 (weighted average over activity classes, as in MTLFormer)
  next time       nt_mae, nt_mse (days^2), nt_rmse, plus nt_mae_prefix_<k> for the --prefixes
  remaining time  rt_mae, rt_mse (days^2), rt_rmse
  suffix          dls (Damerau-Levenshtein similarity, Table 2 of the LSTM paper)
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn import metrics

DAY = 86400.0

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--results_dir", default="output_files/results")
parser.add_argument("--prefixes", type=int, nargs="*", default=None,
                    help="prefix lengths reported separately (default: all prefix lengths found in the results)")
args = parser.parse_args()

root = os.path.join(args.results_dir, args.dataset)
seed_dirs = sorted(glob.glob(os.path.join(root, "seed*")),
                   key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
assert seed_dirs, "no seed* folders in %s" % root


def load(path):
    return pd.read_csv(path, quotechar="|", encoding="utf-8", keep_default_na=False,
                       dtype={"Groud truth": str, "Predicted": str})


def first_number(col):
    """'123; 456' -> 123.0 (only the first predicted step is compared)."""
    return pd.to_numeric(col.astype(str).str.split(";").str[0], errors="coerce")


def classification(df, out):
    y, p = df["Groud truth"], df["Predicted"]
    def one(y, p):
        pr, rc, f1, _ = metrics.precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
        return pd.Series({"accuracy": metrics.accuracy_score(y, p), "precision": pr, "recall": rc, "f1": f1})
    out.update(one(y, p))
    per_k = df.groupby("Prefix length").apply(lambda g: one(g["Groud truth"], g["Predicted"]))
    out.update({k + "_kavg": v for k, v in per_k.mean().items()})


def regression(err_days, prefix_len, name, out):
    """err_days: ground truth - prediction in days."""
    out[name + "_mae"] = err_days.abs().mean()
    out[name + "_mse"] = (err_days ** 2).mean()
    out[name + "_rmse"] = np.sqrt(out[name + "_mse"])
    g = err_days.groupby(prefix_len)
    out[name + "_mae_kavg"] = g.apply(lambda e: e.abs().mean()).mean()
    out[name + "_mse_kavg"] = g.apply(lambda e: (e ** 2).mean()).mean()
    out[name + "_rmse_kavg"] = g.apply(lambda e: np.sqrt((e ** 2).mean())).mean()


columns, rem_by_prefix = {}, []
for d in seed_dirs:
    seed = os.path.basename(d)[4:]
    out = {}
    p = os.path.join(d, "next_activity_and_time_%s.csv" % args.dataset)
    if os.path.exists(p):
        df = load(p)
        out["n_predictions"] = len(df)
        classification(df, out)
        err = (first_number(df["Ground truth times"]) - first_number(df["Predicted times"])) / DAY
        regression(err, df["Prefix length"], "nt", out)
        prefixes = args.prefixes if args.prefixes else sorted(df["Prefix length"].unique())
        for k in prefixes:
            out["nt_mae_prefix_%d" % k] = err[df["Prefix length"] == k].abs().mean()
    p = os.path.join(d, "suffix_and_remaining_time_%s.csv" % args.dataset)
    if os.path.exists(p):
        df = load(p)
        out["dls"] = pd.to_numeric(df["Damerau"], errors="coerce").mean()
        err = (first_number(df["Ground truth times"]) - first_number(df["Predicted times"])) / DAY
        regression(err, df["Prefix length"], "rt", out)
        rem_by_prefix.append(err.abs().groupby(df["Prefix length"]).mean().rename(seed))
    columns[seed] = pd.Series(out)

res = pd.DataFrame(columns)
res["mean"] = res.mean(axis=1)
res["std"] = res.drop(columns="mean").std(axis=1, ddof=1)
pd.set_option("display.width", 200)
print(res.round(4).to_string())
res.to_csv(os.path.join(root, "summary.csv"))
print("\nwrote", os.path.join(root, "summary.csv"))
if rem_by_prefix:
    by_prefix = pd.concat(rem_by_prefix, axis=1)
    by_prefix["mean"] = by_prefix.mean(axis=1)
    by_prefix.to_csv(os.path.join(root, "remaining_time_mae_by_prefix.csv"))
    print("wrote", os.path.join(root, "remaining_time_mae_by_prefix.csv"))
