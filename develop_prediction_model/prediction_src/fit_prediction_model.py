"""Refactored helpers preserving the original CV-final-model workflow.

This module intentionally keeps the original modelling behaviour:

* cross-validation is run on the training split
* the last estimator returned by cross-validation is used as the final model
* threshold changes remain explicit via set_class_threshold(...) or
  update_with_new_cutoff(...)

It exists as a readability-focused counterpart to fit_prediction_model.py.
"""

import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_validate

from fit_prediction_model import (
    SMWrapperClassifier,
    create_stratified_split_data,
    fill_model_dict,
)


def fit_model_kfold(X_train, y_train, n_folds=5, scorer="roc_auc", model_type="logistic"):
    """Run CV and return the last CV estimator, matching the original code."""
    if model_type != "logistic":
        raise ValueError(f"Unsupported model_type: {model_type}")

    X_train = X_train.astype(float)
    k_folds = KFold(n_splits=n_folds)

    results = cross_validate(
        SMWrapperClassifier(sm.GLM),
        X_train,
        y_train,
        cv=k_folds,
        scoring=scorer,
        return_estimator=True,
    )
    return results["estimator"][-1]


def predict_classes_and_probas(model, X, threshold):
    """Return probabilities and predictions using the model's current threshold."""
    probs = model.predict_proba(X)
    thresh_probs = model.predict(X)
    return probs, thresh_probs


def train_model(data, target_column, as_grouped, is_sum, nvf, variables, split_size,
                model_dict, threshold, n_folds, scorer):
    """Train a model while preserving the original final-estimator workflow."""
    X_train, X_test, y_train, y_test = create_stratified_split_data(
        data,
        target_column,
        variables,
    )

    final_model = fit_model_kfold(X_train, y_train, n_folds, scorer)

    prob_train, thresh_probs_train = predict_classes_and_probas(
        final_model,
        X_train,
        threshold,
    )
    prob_test, thresh_probs_test = predict_classes_and_probas(
        final_model,
        X_test,
        threshold,
    )

    return fill_model_dict(
        model_dict,
        data,
        X_train,
        X_test,
        y_train,
        y_test,
        target_column,
        as_grouped,
        is_sum,
        nvf,
        final_model,
        prob_train,
        thresh_probs_train,
        prob_test,
        thresh_probs_test,
    )
