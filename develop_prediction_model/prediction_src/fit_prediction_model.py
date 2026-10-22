"""Refactored helpers preserving the original CV-final-model workflow.

This module intentionally keeps the original modelling behaviour:

* cross-validation is run on the training split
* the last estimator returned by cross-validation is used as the final model
* threshold changes remain explicit via set_class_threshold(...) or
  update_with_new_cutoff(...)

It exists as a readability-focused counterpart to fit_prediction_model.py.
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np
import inspect
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from params import *


def create_binary_predictors(data, group_cols, model_cols, as_sum = False):

    cat_data = pd.get_dummies(data, columns=["category"],prefix="", prefix_sep="")
    model_cols = list(set(model_cols).intersection(cat_data.columns))
    group_cols = list(set(group_cols).intersection(cat_data.columns))
    cat_data = cat_data[model_cols].copy()


    cat_data_res = cat_data.groupby("VersID")[group_cols].sum().reset_index()
    cat_data_res = pd.merge(cat_data[["VersID",TARGET_COLUMN,TARGET_COLUMN_ALL_ENDOS,TARGET_COLUMN_ONLY_ADENO]],cat_data_res,on="VersID")

    if as_sum == False:

        cat_data_bin = cat_data_res.copy()
        
        cat_data_bin = cat_data_bin[group_cols].apply(lambda x:x >0,)
        
        cat_data_res = pd.merge(cat_data_res[["VersID", TARGET_COLUMN,TARGET_COLUMN_ALL_ENDOS,TARGET_COLUMN_ONLY_ADENO]],cat_data_bin,
                               left_index=True,right_index=True)

    return cat_data_res.drop_duplicates(), group_cols, model_cols



def create_target_variables(df):
    """
    Create binary target variables for adenomyosis and endometriosis.

    Args:
        df (pd.DataFrame): Input dataframe containing 'VersID' and 'code'.

    Returns:
        pd.DataFrame: Dataframe with additional target columns.
    """
    df["is_adeno"] = df["code"].astype(str).eq("N80.0")
    df["is_endo_other"] = (
        df["code"].astype(str).str.startswith("N80")
        & ~df["is_adeno"]
    )

    agg = df.groupby("VersID").agg(
        has_adeno=("is_adeno", "any"),
        has_endo_other=("is_endo_other", "any")
    ).reset_index()

    agg["only_adeno"] = agg["has_adeno"] & ~agg["has_endo_other"]
    agg["no_adeno"] = agg["has_endo_other"] & ~agg["has_adeno"]
    agg["all_endos"] = agg["has_adeno"] & agg["has_endo_other"]
    agg["endo"] = agg["no_adeno"] | agg["all_endos"]

    return pd.merge(df, agg, on="VersID")

def get_df_name(df):
    frame = inspect.currentframe().f_back
    names = [name for name, val in frame.f_locals.items() if val is df]
    if names:
        return names[0]
    names = [name for name, val in frame.f_globals.items() if val is df]
    if names:
        return names[0]
    return "unknown_df"
class SMWrapperClassifier(ClassifierMixin,BaseEstimator):
    """
    Scikit-learn compatible wrapper for statsmodels GLM (Binomial).
    """

    def __init__(self, family=None):
        # falls family falsch (z.B. sm.GLM) -> automatisch Binomial
        if family is None or isinstance(family, type):
            self.family = sm.families.Binomial()
        else:
            self.family = family
        self.threshold = 0.5

    def fit(self, X, y):
        """
        Fit GLM model.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            self
        """
        # ✅ Fix: alle Inputs numerisch machen (Bool → float)
        X = pd.DataFrame(X).astype(float)
        y = np.asarray(y).astype(float)
        self.classes_ = np.unique(y)

        X_sm = sm.add_constant(X, has_constant='add')
        self.model_ = sm.GLM(y, X_sm, family=self.family).fit()

        self.feature_names_in_ = np.array(["Intercept"] + list(X.columns))
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Probabilities for both classes.
        """
        # ✅ Fix auch hier: Bool → float + 2D erzwingen
        X = pd.DataFrame(X).astype(float)

        if X.ndim == 1:
            X = X.to_frame().T

        X_sm = sm.add_constant(X, has_constant='add')
        preds = self.model_.predict(X_sm)

        return np.column_stack([1 - preds, preds])

    def predict(self, X):
        """
        Predict binary class labels using threshold.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Predicted labels.
        """
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
    
    def set_class_threshold(self,threshold):
        """Set classification threshold."""
        self.threshold = threshold
        
    @property
    def coef_table_(self):
        """Return coefficient table with odds ratios and confidence intervals."""

        coef = self.model_.params
        pvals = self.model_.pvalues
        conf = self.model_.conf_int()
        conf.columns = ["ci_lower", "ci_upper"]

        df = pd.DataFrame({
            "feature": self.feature_names_in_,
            "coef": coef.values,
            "odds_ratio": np.exp(coef.values),
            "p_value": pvals.values,
            "ci_lower": conf["ci_lower"].values,
            "ci_upper": conf["ci_upper"].values
        })
        return df

    @property
    def coef_(self):
        return self.model_.params[1:] 
    @property
    def intercept_(self):
        return self.model_.params[0]

    @property
    def _coeffs(self):  
        return self.model_.params


def create_stratified_split_data(data, target_column, variables):
    """
    Create stratified train/test split.

    Args:
        data (pd.DataFrame): Input data.
        target_column (str): Target variable.
        variables (list): Feature columns.

    Returns:
        tuple: X_train, y_train, X_test, y_test
    """
    X = data[variables].copy()
    y = data[target_column].copy()

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def fill_model_dict(model_dict, data, X_train, X_test, y_train,y_test,target_column,as_grouped, is_sum,nvf, model,
                   prob_train,thresh_probs_train,prob_test,thresh_probs_test):

    model_name = f"{target_column}_{get_df_name(data)}_as_grouped{as_grouped}_is_sum{is_sum}_nvf{nvf}"
    model_dict[model_name] = {}
    model_dict[model_name]["model_name"] = model_name
    model_dict[model_name]["model"] = model

    # Use column 1 (probability of class 1)
    model_dict[model_name]['prob_0_train'] = prob_train[:, 0]
    model_dict[model_name]['prob_1_train'] = prob_train[:, 1]
    model_dict[model_name]['prob_0_test'] = prob_test[:, 0]
    model_dict[model_name]['prob_1_test'] = prob_test[:, 1]
    
    # Get binary predictions (0/1)
    model_dict[model_name]['pred_train'] = thresh_probs_train
    model_dict[model_name]['pred_test'] = thresh_probs_test

    model_dict[model_name]["X_train_true"] = X_train
    model_dict[model_name]["X_test_true"] = X_test
    model_dict[model_name]["y_train_true"] = y_train
    model_dict[model_name]["y_test_true"] = y_test
    
    return model_dict

def train_model(data, target_column,as_grouped, is_sum,nvf,variables,split_size,
                model_dict,threshold,n_folds,scorer):

    X_train, X_test,y_train, y_test = create_stratified_split_data(data,target_column,variables)


    final_model = fit_model_kfold(X_train, y_train, n_folds, scorer)

    prob_train,thresh_probs_train = predict_classes_and_probas(final_model,X_train, threshold)
    prob_test,thresh_probs_test = predict_classes_and_probas(final_model,X_test, threshold)  

    model_dict = fill_model_dict(model_dict, data, X_train, X_test,
                                 y_train,y_test,target_column, as_grouped, is_sum,nvf,final_model,
                                prob_train, thresh_probs_train, 
                                 prob_test,thresh_probs_test)

    return model_dict


def fit_model_kfold(X_train, y_train, n_folds=5,
                    scorer="roc_auc", model_type="logistic"):

    cv = StratifiedKFold(
    n_splits=n_folds,
    shuffle=True,
    random_state=42
    )

    results = cross_validate(
    SMWrapperClassifier(sm.GLM),
    X_train,
    y_train,
    cv=cv,
    scoring=scorer,
    return_estimator=True,
    error_score="raise",
    )

    cv_auc = results["test_score"].mean()
    print(f"Mean AUC during cross validation: {cv_auc:.3f}")
    
    # Danach finales Modell auf ALLEN Trainingsdaten fitten
    final_model = SMWrapperClassifier(sm.GLM)
    final_model.fit(X_train, y_train)

    return final_model


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

