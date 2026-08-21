import json
from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


class ExperimentSummarizer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()

    @staticmethod
    def compute_f1(p, r):
        return 0.0 if p + r == 0 else 2 * (p * r) / (p + r)

    @staticmethod
    def extract_n_labels(name):
        m = re.search(r"obj_(\d+)", name)
        return int(m.group(1)) if m else None

    def _collect(self):
        rows = []

        for p in self.root_dir.glob("*/*/obj_*/repeat_*/eval_test/test_metrics.json"):
            parts = p.parts
            dataset, method, obj_dir, repeat = parts[-6], parts[-5], parts[-4], parts[-3]
            n_labels = self.extract_n_labels(obj_dir)

            with open(p) as f:
                m = json.load(f)

            prec = m.get("metrics/precision(B)", np.nan)
            rec = m.get("metrics/recall(B)", np.nan)
            ap50 = m.get("metrics/mAP50(B)", np.nan)
            map5095 = m.get("metrics/mAP50-95(B)", np.nan)

            rows.append({
                "dataset": dataset,
                "method": method,
                "n_labels": n_labels,
                "repeat": repeat,
                "precision": prec,
                "recall": rec,
                "F1": self.compute_f1(prec, rec),
                "AP50": ap50,
                "mAP50-95": map5095,
            })

        for p in self.root_dir.glob("*/*/repeat_*_seed_*/eval_test/test_metrics.json"):
            parts = p.parts
            dataset, method, repeat_dir = parts[-5], parts[-4], parts[-3]

            m_repeat = re.search(r"repeat_(\d+)", repeat_dir)
            repeat = m_repeat.group(1) if m_repeat else repeat_dir

            with open(p) as f:
                m = json.load(f)

            prec = m.get("metrics/precision(B)", np.nan)
            rec = m.get("metrics/recall(B)", np.nan)
            ap50 = m.get("metrics/mAP50(B)", np.nan)
            map5095 = m.get("metrics/mAP50-95(B)", np.nan)

            rows.append({
                "dataset": dataset,
                "method": method,
                "n_labels": "all",
                "repeat": repeat,
                "precision": prec,
                "recall": rec,
                "F1": self.compute_f1(prec, rec),
                "AP50": ap50,
                "mAP50-95": map5095,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _aggregate(df, group_cols, with_std=False):
        g = (
            df.groupby(group_cols)
              .agg(
                  precision_mean=("precision", "mean"),
                  precision_std=("precision", "std"),
                  recall_mean=("recall", "mean"),
                  recall_std=("recall", "std"),
                  F1_mean=("F1", "mean"),
                  F1_std=("F1", "std"),
                  AP50_mean=("AP50", "mean"),
                  AP50_std=("AP50", "std"),
                  mAP50_95_mean=("mAP50-95", "mean"),
                  mAP50_95_std=("mAP50-95", "std"),
                  n_repeats=("repeat", "count"),
            ).reset_index())

        for c in ["precision", "recall", "F1", "AP50", "mAP50_95"]:
            g[c] = g[f"{c}_mean"].map("{:.4f}".format) if not with_std else g[f"{c}_mean"].map("{:.4f}".format) + " ± " + g[f"{c}_std"].map("{:.4f}".format)

        return g

    def summarize(self, average_over_datasets=False):
        df = self._collect()
        if df.empty:
            raise ValueError("Brak wyników.")

        group_cols = ["method", "n_labels"] if average_over_datasets else ["dataset", "method", "n_labels"]

        g = self._aggregate(df, group_cols)

        cols = group_cols + [
            "precision", "recall", "F1",
            "AP50", "mAP50_95", "n_repeats"
        ]

        return g[cols].sort_values(group_cols)

    def summarize_per_run(self):
        df = self._collect()
        if df.empty:
            raise ValueError("Brak wyników.")

        group_cols = ["dataset", "method", "n_labels"]

        g = self._aggregate(df, group_cols, with_std=True)

        cols = group_cols + [
            "precision", "recall", "F1",
            "AP50", "mAP50_95", "n_repeats"
        ]

        return g[cols].sort_values(group_cols)

    def plot_ap50_vs_nlabels(self, df: pd.DataFrame):
        plt.figure()

        for m in ["only_labeled", "yolo_ub", "weak_yolo_kd", "weak_yolo", "sam"]:
            subset = df[df.method == m].copy()
            if subset.empty:
                continue

            subset["n_labels_num"] = pd.to_numeric(subset["n_labels"], errors="coerce")
            subset = subset.dropna(subset=["n_labels_num"]).sort_values("n_labels_num")
            if subset.empty:
                continue

            plt.plot(subset["n_labels_num"], subset["AP50"], marker="o", label=m)

        std = df[df.method == "standard_yolo"]
        if not std.empty:
            ap_std = std["AP50"].iloc[0]
            x_min = df["n_labels"].pipe(pd.to_numeric, errors="coerce").min()
            x_max = df["n_labels"].pipe(pd.to_numeric, errors="coerce").max()
            if pd.notna(x_min) and pd.notna(x_max):
                plt.plot([x_min, x_max], [ap_std, ap_std], marker="o", label="standard_yolo")

        plt.xlabel("n_labels")
        plt.ylabel("AP50")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("chart_results.png", dpi=300)


if __name__ == "__main__":
    ROOT_DIR = "runs/"
    OUTPUT_CSV1 = "results_summary.csv"
    OUTPUT_CSV2 = "results_summary_per_run.csv"
    AVERAGE_OVER_DATASETS = True

    summarizer = ExperimentSummarizer(ROOT_DIR)
    df = summarizer.summarize(average_over_datasets=AVERAGE_OVER_DATASETS)
    df2 = summarizer.summarize_per_run()

    print(df.to_string(index=False))
    print(df2.to_string(index=False))

    if OUTPUT_CSV1 is not None:
        df.to_csv(OUTPUT_CSV1, index=False)
    if OUTPUT_CSV2 is not None:
        df2.to_csv(OUTPUT_CSV2, index=False)

    df["AP50"] = df["AP50"].astype("float16")
    summarizer.plot_ap50_vs_nlabels(df)
