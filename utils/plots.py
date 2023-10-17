# Plotting functions for ROC, PR, and confusion matrix
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_roc(metrics, save_dir=Path("")):
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for metric in metrics:
        fpr, tpr = metric.metrics["fpr"], metric.metrics["tpr"]
        plt.plot(
            tpr,
            fpr,
            color="darkorange",
            lw=2,
            label="{} (area = {:.2f})".format(metric.name, metric.metrics["roc_auc"]),
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(Path(save_dir, "roc.png"))
    plt.close()


def plot_pr(metrics, save_dir=Path("")):
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for metric in metrics:
        recall, precision = metric.metrics["recall"], metric.metrics["precision"]
        plt.plot(
            recall,
            precision,
            color="darkorange",
            lw=2,
            label="{} (area = {:.2f})".format(metric.name, metric.metrics["pr_auc"]),
        )
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall")
    plt.legend(loc="lower right")
    plt.savefig(Path(save_dir, "pr.png"))
    plt.close()
