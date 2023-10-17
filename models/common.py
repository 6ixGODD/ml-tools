# Common functions for models
import dataclasses

from sklearn import svm
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
)
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    SelectFpr,
    SelectFdr,
    SelectFwe,
    f_classif,
)
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
)
from sklearn.tree import DecisionTreeClassifier


def get_preprocessing(name):
    if name == "StandardScaler":
        return StandardScaler
    elif name == "MinMaxScaler":
        return MinMaxScaler
    elif name == "MaxAbsScaler":
        return MaxAbsScaler
    elif name == "RobustScaler":
        return RobustScaler
    elif name == "Normalizer":
        return Normalizer
    else:
        raise ValueError("Preprocessing method not found.")


def get_classifiers(name):
    if name == "SVC":
        return svm.SVC
    elif name == "AdaBoost":
        return AdaBoostClassifier
    elif name == "Bagging":
        return BaggingClassifier
    elif name == "RandomForest":
        return RandomForestClassifier
    elif name == "LogisticRegression":
        return LogisticRegression
    elif name == "GaussianNB":
        return GaussianNB
    elif name == "KNN":
        return KNeighborsClassifier
    elif name == "DecisionTree":
        return DecisionTreeClassifier
    elif name == "ExtraTrees":
        return ExtraTreesClassifier
    else:
        print(name)
        raise ValueError("Algorithm not found.")


def get_feature_selection(name):
    if name == "SelectKBest":
        return SelectKBest
    elif name == "SelectPercentile":
        return SelectPercentile
    elif name == "SelectFpr":
        return SelectFpr
    elif name == "SelectFdr":
        return SelectFdr
    elif name == "SelectFwe":
        return SelectFwe
    elif name == "Lasso":
        return Lasso
    elif name == "LassoCV":
        return LassoCV
    elif name == "f_classif":
        return f_classif
    else:
        raise ValueError("Feature selection method not found.")


def get_model_selection(name):
    if name == "KFold":
        return KFold
    elif name == "StratifiedKFold":
        return StratifiedKFold
    elif name == "RepeatedKFold":
        return RepeatedKFold
    elif name == "RepeatedStratifiedKFold":
        return RepeatedStratifiedKFold
    elif name == "train_test_split":
        return train_test_split
    else:
        raise ValueError("Model selection method not found.")


# Dataclass for metrics
Metrics = dataclasses.make_dataclass("Metric", ["name", "clf", "metrics"])
