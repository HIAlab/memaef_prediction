
from sklearn.metrics import roc_auc_score, brier_score_loss
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def return_y_true_and_pred(model_dict):
    """
    Extract true labels and predicted probabilities from model dictionary.

    Args:
        model_dict (dict): Dictionary containing model outputs with keys
                           'prob_1_test' and 'y_test_true'.

    Returns:
        tuple:
            y_true (array-like): True labels.
            y_prob (array-like): Predicted probabilities for class 1.
    """
    one_probs = model_dict[list(model_dict.keys())[0]]["prob_1_test"]
    testy = model_dict[list(model_dict.keys())[0]]["y_test_true"]
    return testy, one_probs


def calibration_intercept_slope(y_true, y_prob):
    """
    Compute calibration intercept and slope using logistic recalibration.

    A logistic regression model is fitted with the logit of predicted
    probabilities as predictor and true labels as outcome.

    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities.

    Returns:
        tuple:
            calibration_intercept (float)
            calibration_slope (float)
    """
    # Avoid division by zero (important edge case!)
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    logit_pred = np.log(y_prob / (1 - y_prob))

    if isinstance(logit_pred, pd.Series):
        logit_pred = logit_pred.values.reshape(-1, 1)

    log_reg = LogisticRegression(fit_intercept=True, solver='liblinear')

    log_reg.fit(
        np.array(logit_pred).reshape(-1, 1),
        np.array(y_true).ravel()
    )

    calibration_slope = log_reg.coef_[0][0]
    calibration_intercept = log_reg.intercept_[0]

    return calibration_intercept, calibration_slope


def calc_eval_metrics(models_dict: dict, met_list: list) -> pd.DataFrame:
    """
    Calculate evaluation metrics for a given model dictionary.

    Args:
        models_dict (dict): Dictionary containing model predictions and labels.
        met_list (list): List of tuples (metric_function, metric_name).

    Returns:
        pd.DataFrame: DataFrame containing computed metrics per model.
    """
    metrics = defaultdict(list)

    y_true, y_prob = return_y_true_and_pred(models_dict)

    model_entry = models_dict[list(models_dict.keys())[0]]
    model_name = model_entry["model_name"]

    metrics['Model'].append(model_name)

    for metric_func, metric_name in met_list:
        result = metric_func(y_true, y_prob)

        if "Calibration" not in metric_name:
            metrics[metric_name].append(result)
        else:
            if "Intercept" in metric_name:
                metrics[metric_name].append(result[0])
            else:
                metrics[metric_name].append(result[1])

    metric_df = pd.DataFrame(metrics).set_index('Model')

    return metric_df.round(2)