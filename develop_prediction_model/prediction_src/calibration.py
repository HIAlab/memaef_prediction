import copy

from sklearn.metrics import roc_curve, confusion_matrix
import pandas as pd
import numpy as np


def return_y_true_and_pred(model_dict):
    """
    Extract true labels and predicted probabilities from model dictionary.

    Args:
        model_dict (dict): Dictionary containing model outputs, including
                           'prob_1_test' and 'y_test_true'.

    Returns:
        tuple:
            y_true (array-like): True labels.
            y_prob (array-like): Predicted probabilities for class 1.
    """
    one_probs = model_dict[list(model_dict.keys())[0]]["prob_1_test"]
    testy = model_dict[list(model_dict.keys())[0]]["y_test_true"]
    return testy, one_probs


def compute_and_save_cutoff(model_dict,cutoff_type):
    y_true, y_prob = return_y_true_and_pred(model_dict)

    if cutoff_type.lower() == "fpr":
        cutoff = cutoff_fpr_constraint(y_true, y_prob, max_fpr=0.1)
    elif cutoff_type.lower() == "ppv":
        cutoff = cutoff_max_ppv(y_true, y_prob)
    elif cutoff_type.lower() == "youden":
        cutoff = cutoff_youden(y_true, y_prob)
    elif cutoff_type.lower() == "default":
        cutoff = 0.5
    else:
        print("Cutoff type not implemented.")

    print(f"{cutoff_type} optimal cutoff:", round(cutoff,3))

    metrics_for_cutoff = classification_metrics(model_dict,cutoff)
    print("\nMetrics for this cutoff:")
    print(metrics_for_cutoff.round(3))

    return cutoff, metrics_for_cutoff

def update_with_new_cutoff(model_dict,cutoff):
    
    cutoff_model_dict=copy.deepcopy(model_dict)
    
    cutoff_model_dict[list(cutoff_model_dict.keys())[0]]["model"].set_class_threshold(cutoff)
    
    cutoff_model_dict = update_cutoff_dict(cutoff_model_dict,f"{cutoff}_model")

    return cutoff_model_dict


def update_cutoff_dict(model_dict, dict_key):
    """
    Update model dictionary with prediction outputs and return a new dictionary.

    Args:
        model_dict (dict): Dictionary containing model and test data.
        dict_key (str): Key under which the updated model data is stored.

    Returns:
        dict: Updated dictionary containing predictions and probabilities.
    """
    model_entry = model_dict[list(model_dict.keys())[0]]

    model_entry["prob_1_test"] = model_entry["model"].predict_proba(
        model_entry["X_test_true"]
    )[:, 1]

    model_entry["pred_test"] = model_entry["model"].predict(
        model_entry["X_test_true"]
    )

    model_entry["prob_0_test"] = model_entry["model"].predict_proba(
        model_entry["X_test_true"]
    )[:, 0]

    updated_dict = {dict_key: model_entry}
    return updated_dict


def cutoff_closest_roc(y_true, y_prob):
    """
    Compute threshold closest to the top-left corner of the ROC curve.

    Args:
        y_true (array-like): True labels.
        y_prob (array-like): Predicted probabilities.

    Returns:
        float: Optimal threshold minimizing distance to (0,1).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    distances = np.sqrt((1 - tpr) ** 2 + (fpr) ** 2)
    best_idx = np.argmin(distances)

    return thresholds[best_idx]


def cutoff_max_ppv(y_true, y_prob):
    """
    Compute threshold that maximizes Positive Predictive Value (PPV).

    Args:
        y_true (array-like): True labels.
        y_prob (array-like): Predicted probabilities.

    Returns:
        float: Threshold yielding highest PPV.
    """
    thresholds = np.linspace(0, 1, 500)
    best_ppv = -1
    best_threshold = 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        if tp + fp == 0:
            continue

        ppv = tp / (tp + fp)

        if ppv > best_ppv:
            best_ppv = ppv
            best_threshold = t

    return best_threshold


def cutoff_youden(y_true, y_prob):
    """
    Compute threshold using Youden's J statistic (maximizing sensitivity + specificity - 1).

    Args:
        y_true (array-like): True labels.
        y_prob (array-like): Predicted probabilities.

    Returns:
        float: Optimal threshold based on Youden index.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)

    return thresholds[ix]


def cutoff_fpr_constraint(y_true, y_prob, max_fpr=0.1):
    """
    Compute threshold under a maximum false positive rate constraint.

    Args:
        y_true (array-like): True labels.
        y_prob (array-like): Predicted probabilities.
        max_fpr (float): Maximum allowed false positive rate.

    Returns:
        float: Threshold maximizing TPR under FPR constraint.

    Raises:
        ValueError: If no threshold satisfies the FPR constraint.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    valid = fpr <= max_fpr

    if not np.any(valid):
        raise ValueError("No threshold satisfies the FPR constraint")

    best_idx = np.argmax(tpr[valid])
    return thresholds[valid][best_idx]


def classification_metrics(model_dict, threshold):
    """
    Compute classification metrics based on a given threshold.

    Args:
        model_dict (dict): Dictionary containing model predictions.
        threshold (float): Decision threshold for classification.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics.
    """
    y_true, y_prob = return_y_true_and_pred(model_dict)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    fpr = fp / (fp + tn)
    tpr = sensitivity
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return pd.DataFrame({
        "Sensitivity": [sensitivity],
        "Specificity": [specificity],
        "PPV": [ppv],
        "FPR": [fpr],
        "TPR": [tpr],
        "Accuracy": [accuracy],
        "TP": [tp],
        "FP": [fp],
        "FN": [fn],
        "TN": [tn]
    })
