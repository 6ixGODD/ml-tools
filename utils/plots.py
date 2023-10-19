# Plotting functions
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"


def plot_roc(metrics, save_dir=Path("")):
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(dpi=300)
    for metric in metrics:
        fpr, tpr = metric.metrics["fpr"], metric.metrics["tpr"]
        plt.plot(
            tpr,
            fpr,
            lw=2,
            label="{} (area = {:.2f})".format(metric.name, metric.metrics["roc_auc"]),
            linestyle="-",
        )
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(
        loc="lower right", bbox_to_anchor=(1.5, 0), frameon=False, fontsize="small",
    )
    plt.savefig(Path(save_dir, "roc.png"), bbox_inches="tight")
    plt.close()


def plot_pr(metrics, save_dir=Path("")):
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(dpi=300)
    for metric in metrics:
        recall, precision = metric.metrics["recall"], metric.metrics["precision"]
        plt.plot(
            recall,
            precision,
            lw=2,
            label="{} (area = {:.2f})".format(metric.name, metric.metrics["pr_auc"]),
            linestyle="-",
        )
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall")
    plt.legend(
        loc="lower right", bbox_to_anchor=(1.5, 0), frameon=False, fontsize="small",
    )
    plt.savefig(Path(save_dir, "pr.png"), bbox_inches="tight")
    plt.close()


def plot_lasso_mse(lasso, save_dir=Path("")):
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(dpi=300)
    # Lasso: coordinate descent
    plt.errorbar(
        lasso.alphas_,
        lasso.mse_path_.mean(axis=-1),
        yerr=lasso.mse_path_.std(axis=-1),
        fmt="o",
        ms=3,
        mfc="r",
        mec="r",
        ecolor="lightblue",
        elinewidth=2,
        capsize=3,
        capthick=2,
    )
    plt.semilogx()
    plt.axvline(lasso.alpha_, linestyle="--", color="black")
    plt.xlabel("Lambda")
    plt.ylabel("Mean Square Error")
    plt.title("Mean Square Error: Coordinate Descent ")
    # import MultipleLocator to set the tick interval

    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    plt.savefig(Path(save_dir, "lasso_mse.png"))
    plt.close()


def plot_lasso_path(lasso, X, y, save_dir=Path("")):
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(dpi=300)
    plt.semilogx(
        lasso.alphas_, lasso.path(X, y, alphas=lasso.alphas_, max_iter=100000)[1].T, "-"
    )
    plt.axvline(lasso.alpha_, linestyle="--", color="k")
    plt.xlabel("Lambda")
    plt.ylabel("Coefficients")
    plt.title("Lasso Path")
    plt.savefig(Path(save_dir, "lasso_path.png"))
    plt.close()
