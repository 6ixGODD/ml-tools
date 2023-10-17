# ML-EnsembleHub :)

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.utils import shuffle
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils import Config, increment_path, get_logger
from utils.check import check_data, check_cfg
from utils.plots import plot_roc, plot_pr
from models.common import (
    get_preprocessing,
    get_feature_selection,
    get_classifiers,
    get_model_selection,
    Metrics,
)

LOGGER = None


def run(data, cfg, save_dir, plot, save):
    LOGGER.info(
        "=" * 160
        + "\n>> ML-EnsembleHub :)\n"
        + "=" * 160
        + "\nML-EnsembleHub is a Python tool designed for ensemble machine learning experiments. "
        "It provides an easy-to-use interface for building and evaluating ensemble \n"
        "models using various classifiers, feature selection techniques, and model selection methods.\n"
        + "-" * 160
        + "\n> Data Summary:\n\n{}\n".format(str(data.describe()).replace("\n", "\n\t"))
        + "-" * 160
        + "\n> Config Summary:\n\n{}\n".format(
            str(yaml.dump(cfg)).replace("\n", "\n\t")
        )
        + "-" * 160
        + "\n- start running..."
    )
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    # data preprocessing
    if cfg.shuffle:
        X, y = shuffle(X, y, random_state=cfg.random_state)
    if cfg.preprocessing["method"] is not None:
        LOGGER.info(
            "-" * 160 + "\n> Preprocessing: {}".format(cfg.preprocessing["method"])
        )
        pre = get_preprocessing(cfg.preprocessing["method"])
        pre = pre(**cfg.preprocessing[cfg.preprocessing["method"]])
        X = pre.fit_transform(X)

    # feature selection
    if cfg.feature_selection["method"] is not None:
        LOGGER.info(
            "-" * 160
            + "\n> Feature selection: {}".format(cfg.feature_selection["method"])
        )
        # Lasso and LassoCV
        if (
            cfg.feature_selection["method"] == "Lasso"
            or cfg.feature_selection["method"] == "LassoCV"
        ):
            fs = get_feature_selection(cfg.feature_selection["method"])
            fs = fs(**cfg.feature_selection[cfg.feature_selection["method"]])
            fs.fit(X, y)
            X = X[:, fs.coef_ != 0]
            LOGGER.info("\tSelected {:d} variables".format(len(X[0])))
            LOGGER.info("\talpha: {}".format(fs.alpha_))
        # other methods
        else:
            fs = get_feature_selection(cfg.feature_selection["method"])
            score_func = get_feature_selection(
                cfg.feature_selection[cfg.feature_selection["method"]].pop("score_func")
            )
            fs = fs(
                **cfg.feature_selection[cfg.feature_selection["method"]],
                score_func=score_func,
            )
            fs.fit(X, y)
            X = fs.transform(X)
            LOGGER.info(
                "- Selected features: {}"
                "\n\t Feature scores: {}"
                "\n\t Feature pvalues: {}".format(
                    str(fs.get_support(indices=True)).replace("\n", "\n\t"),
                    str(fs.scores_).replace("\n", "\n\t"),
                    str(fs.pvalues_).replace("\n", "\n\t"),
                )
                + "\n"
                + "-" * 160
            )

    # train
    LOGGER.info("- start training...")
    metrics = []
    for clf_name in cfg.classifiers["methods"]:
        clf = get_classifiers(clf_name)
        clf = clf(**cfg.classifiers[clf_name])
        LOGGER.info(
            "-" * 160
            + "\n> Classifier: {}".format(clf_name)
            + "\n- Hyperparameters: \n\t{}\n".format(
                str(yaml.dump(clf.get_params())).replace("\n", "\n\t")
            )
            + "-" * 160
        )

        # model selection
        ms = get_model_selection(cfg.model_selection["method"])
        # train_test_split
        if cfg.model_selection["method"] == "train_test_split":
            X_train, X_test, y_train, y_test = ms(
                X, y, **cfg.model_selection[cfg.model_selection["method"]]
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            tpr, fpr, thresholds = roc_curve(y_test, y_pred_proba)
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            LOGGER.info(
                "- Metrics:\n"
                "\tAccuracy:\t{}\n"
                "\tPrecision:\t{}\n"
                "\tRecall:\t\t{}\n"
                "\tF1:\t\t\t{}\n"
                "\tAUC:\t\t{}\n"
                "\tConfusion matrix: \n\t{}\n".format(
                    accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred),
                    recall_score(y_test, y_pred),
                    f1_score(y_test, y_pred),
                    roc_auc_score(y_test, y_pred),
                    str(confusion_matrix(y_test, y_pred)).replace("\n", "\n\t"),
                )
            )
            metrics.append(
                Metrics(
                    name=clf_name,
                    clf=clf,
                    metrics={
                        "tpr": tpr,
                        "fpr": fpr,
                        "precision": precision,
                        "recall": recall,
                        "roc_auc": roc_auc_score(y_test, y_pred),
                        "pr_auc": roc_auc_score(y_test, y_pred),
                        "cm": confusion_matrix(y_test, y_pred),
                    },
                )
            )

        # KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
        elif cfg.model_selection["method"] in [
            "KFold",
            "StratifiedKFold",
            "RepeatedKFold",
            "RepeatedStratifiedKFold",
        ]:
            tpr_list, precision_list, roc_auc_list, pr_auc_list, cm_list = (
                [] for _ in range(5)
            )
            ms = ms(**cfg.model_selection[cfg.model_selection["method"]])
            fpr_, recall_ = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
            for i, (train_index, test_index) in tqdm(
                enumerate(ms.split(X, y)),
                desc=clf_name,
                total=ms.get_n_splits(),
                ncols=60,
                unit="fold",
                mininterval=0,
                position=0,
            ):
                # use iloc to avoid SettingWithCopyWarning
                X_train, X_test = (
                    pd.DataFrame(X).iloc[train_index],
                    pd.DataFrame(X).iloc[test_index],
                )
                y_train, y_test = (
                    pd.DataFrame(y).iloc[train_index],
                    pd.DataFrame(y).iloc[test_index],
                )
                clf.fit(X_train, y_train.values.ravel())
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                tpr, fpr, thresholds = roc_curve(y_test, y_pred_proba)
                precision, recall, thresholds = precision_recall_curve(
                    y_test, y_pred_proba
                )
                tpr_list.append(np.interp(fpr_, fpr, tpr))
                tpr_list[-1][0] = 0.0
                precision_list.append(np.interp(recall_, recall[::-1], precision[::-1]))
                precision_list[-1][0] = 1.0
                roc_auc_list.append(roc_auc_score(y_test, y_pred))
                pr_auc_list.append(roc_auc_score(y_test, y_pred))
                cm_list.append(confusion_matrix(y_test, y_pred))
                LOGGER.info(
                    "\tNo.{:d} fold:"
                    "\tAccuracy: {:.5f}  "
                    "Precision: {:.5f} "
                    " Recall: {:.5f} "
                    " F1: {:.5f}  "
                    "AUC: {:.5f}".format(
                        i,
                        accuracy_score(y_test, y_pred),
                        precision_score(y_test, y_pred),
                        recall_score(y_test, y_pred),
                        f1_score(y_test, y_pred),
                        roc_auc_score(y_test, y_pred),
                    )
                )
            tpr = np.mean(tpr_list, axis=0)
            fpr = fpr_
            precision = np.mean(precision_list, axis=0)
            recall = recall_
            roc_auc = np.mean(roc_auc_list)
            pr_auc = np.mean(pr_auc_list)
            cm = np.mean(cm_list, axis=0)

            metrics.append(
                Metrics(
                    name=clf_name,
                    clf=clf,
                    metrics={
                        "tpr": tpr,
                        "fpr": fpr,
                        "precision": precision,
                        "recall": recall,
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "cm": cm,
                    },
                )
            )
        else:
            raise NotImplementedError

        # save model
        if save:
            if not os.path.exists(Path(save_dir, "models")):
                os.makedirs(Path(save_dir, "models"))
            clf.fit(X, y)
            with open(Path(save_dir, "models", f"{clf_name}.pkl"), "wb") as f:
                pickle.dump(clf, f)  # add other serialization methods
            LOGGER.info(
                "* Model saved to `{}`".format(
                    Path(save_dir, "models", f"{clf_name}.pkl")
                )
            )
    metrics_df = pd.DataFrame()
    for m in metrics:
        metrics_df = metrics_df.append(
            pd.DataFrame(
                {
                    "Method": [m.name],
                    "TPR": [m.metrics["tpr"]],
                    "FPR": [m.metrics["fpr"]],
                    "Precision": [m.metrics["precision"]],
                    "Recall": [m.metrics["recall"]],
                    "ROC AUC": [m.metrics["roc_auc"]],
                    "PR AUC": [m.metrics["pr_auc"]],
                    "Confusion Matrix": [m.metrics["cm"]],
                }
            )
        )

    # save metrics as csv
    if not Path(save_dir, "metrics").exists():
        Path(save_dir, "metrics").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(Path(save_dir, "metrics", "metrics.csv"), index=False)
    LOGGER.info("* Metrics saved to `{}`".format(Path(save_dir, "metrics")))

    # plot metrics
    if plot:
        plot_pr(metrics, save_dir=Path(save_dir, "plot"))
        plot_roc(metrics, save_dir=Path(save_dir, "plot"))
        LOGGER.info("* Plots saved to `{}`".format(Path(save_dir, "plot")))
    LOGGER.info("-" * 160 + "\n> Done!" + "\n" + "-" * 160)  # end of run
    LOGGER.info(
        "> Metrics Summary:\n\n{}".format(
            metrics_df.to_string(
                columns=["Method", "ROC AUC", "PR AUC", "Confusion Matrix"],
                index=False,
                col_space=20,
            ),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="data file path",
        default=Path(ROOT, "data", "data.csv"),
        type=Path,
    )
    parser.add_argument(
        "--cfg",
        help="config file path",
        default=Path(ROOT, "configs", "_base_.yml"),
        type=Path,
    )
    parser.add_argument(
        "--save-dir",
        help="dir to save logs, models and plots(if enabled)",
        default=Path(ROOT, "output"),
        type=Path,
    )
    parser.add_argument(
        "--name", help="name of the experiment", default="experiment", type=str
    )
    parser.add_argument(
        "--plot", help="enable plot", action="store_true", default=False
    )
    parser.add_argument(
        "--save", help="enable model saving", action="store_true", default=False
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_file, cfg_file, save_dir, name, plot, save = (
        args.data,
        args.cfg,
        args.save_dir,
        args.name,
        args.plot,
        args.save,
    )

    # init save_dir
    save_dir = increment_path(save_dir / name)

    # init logger
    global LOGGER
    LOGGER = get_logger(name, save_dir)

    # config
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = Config(cfg)

    # load Data
    data = pd.read_csv(data_file, header=0)

    # check data and config
    check_data(data)
    check_cfg(cfg)

    # noinspection PyTypeChecker
    run(data, cfg, save_dir, plot, save)


if __name__ == "__main__":
    main()
