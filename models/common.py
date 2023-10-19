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
    GenericUnivariateSelect,
    SequentialFeatureSelector,
    RFE,
    RFECV,
    VarianceThreshold,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    SelectFpr,
    SelectFdr,
    SelectFwe,
    f_classif,
    chi2,
    mutual_info_classif,
    f_oneway,
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
    preprocessing_methods = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "MaxAbsScaler": MaxAbsScaler,
        "RobustScaler": RobustScaler,
        "Normalizer": Normalizer,
    }

    if name not in preprocessing_methods:
        raise ValueError(f"Preprocessing method '{name}' not found.")

    return preprocessing_methods[name]


def get_classifier(name):
    classifiers = {
        "SVC": svm.SVC,
        "AdaBoost": AdaBoostClassifier,
        "Bagging": BaggingClassifier,
        "RandomForest": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "GaussianNB": GaussianNB,
        "KNN": KNeighborsClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "ExtraTrees": ExtraTreesClassifier,
    }

    if name not in classifiers:
        raise ValueError(f"Classifier '{name}' not found.")

    return classifiers[name]


def get_feature_selector(name):
    feature_selectors = {
        "SelectKBest": SelectKBest,
        "SelectPercentile": SelectPercentile,
        "SelectFpr": SelectFpr,
        "SelectFdr": SelectFdr,
        "SelectFwe": SelectFwe,
        "GenericUnivariateSelect": GenericUnivariateSelect,
        "SequentialFeatureSelector": SequentialFeatureSelector,
        "RFE": RFE,
        "RFECV": RFECV,
        "VarianceThreshold": VarianceThreshold,
        "SelectFromModel": SelectFromModel,
    }

    if name not in feature_selectors:
        raise ValueError("feature selectors {} not found.".format(name))

    return feature_selectors[name]


def get_feature_selection_score(name):
    score_func = {
        "f_classif": f_classif,
        "chi2": chi2,
        "mutual_info_classif": mutual_info_classif,
        "f_oneway": f_oneway,
    }

    if name not in score_func:
        raise ValueError(f"Feature selection score '{name}' not found.")

    return score_func[name]


def get_model_selection(name):
    model_selection_methods = {
        "KFold": KFold,
        "StratifiedKFold": StratifiedKFold,
        "RepeatedKFold": RepeatedKFold,
        "RepeatedStratifiedKFold": RepeatedStratifiedKFold,
        "train_test_split": train_test_split,
    }

    if name not in model_selection_methods:
        raise ValueError(f"Model selection method '{name}' not found.")

    return model_selection_methods[name]


# Dataclass for metrics
Metrics = dataclasses.make_dataclass("Metric", ["name", "clf", "metrics"])
