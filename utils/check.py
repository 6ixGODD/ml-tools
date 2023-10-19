# Check the data and config file.
def check_data(data):
    # value check
    if data.isnull().values.any():
        raise ValueError("Data contains null values.")
    if data.isna().values.any():
        raise ValueError("Data contains NA values.")
    if data.empty:
        raise ValueError("Data is empty.")
    if "label" not in data.columns:
        raise ValueError("Data does not contain label.")
    if not data.columns[1:].all():
        raise ValueError("The column names should not be empty.")
    if not set(data["label"].unique()) == {0, 1}:  # 0: negative, 1: positive
        raise ValueError("The label should be 0 or 1.")
    if len(data.columns) != len(set(data.columns)):
        raise ValueError("There are duplicate column names.")


def check_cfg(cfg):
    # existence check
    if cfg.classifiers is None:
        raise ValueError("No classifier specified.")
    if cfg.feature_selection is None:
        raise ValueError("No feature selection method specified.")
    if cfg.model_selection is None:
        raise ValueError("No model selection method specified.")
    if cfg.shuffle is None:
        raise ValueError("No shuffle method specified.")
    if cfg.random_state is None:
        raise ValueError("No random state specified.")
    if cfg.classifiers["methods"] is None:
        raise ValueError("No classifier method specified.")
    for method in cfg.classifiers["methods"]:
        if method not in cfg.classifiers:
            raise ValueError("No hyperparameters for {} specified.".format(method))
    if cfg.model_selection["method"] is None:
        raise ValueError("No model selection method specified.")

    # type check
    if not isinstance(cfg.random_state, int):
        raise ValueError("Random state should be an integer.")
    if not isinstance(cfg.shuffle, bool):
        raise ValueError("Shuffle should be a boolean.")
    if not isinstance(cfg.classifiers["methods"], list):
        raise ValueError("Classifier method should be a list.")
    if not isinstance(cfg.classifiers, dict):
        raise ValueError("Classifiers should be a dict.")
    if not isinstance(cfg.feature_selection, dict):
        raise ValueError("Feature selection should be a dict.")
    if not isinstance(cfg.model_selection, dict):
        raise ValueError("Model selection should be a dict.")
    if not isinstance(cfg.preprocessing, dict):
        raise ValueError("Preprocessing should be a dict.")
    if not isinstance(cfg.model_selection["method"], str):
        raise ValueError("Model selection method should be a string.")
    if not isinstance(cfg.classifiers["methods"], list):
        raise ValueError("Classifier method should be a list.")
