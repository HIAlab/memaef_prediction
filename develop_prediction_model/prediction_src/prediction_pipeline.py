import pandas as pd
from fit_prediction_model import *


def model_pipeline(filepath, target_column, as_grouped, nvf,
                   code_to_label, group_cols, model_cols,
                   is_sum, split_size, threshold, n_folds, scorer):
    """
    End-to-end pipeline for loading data, preprocessing, and training a model.

    Steps:
        1. Load data from pickle file
        2. Create target variables if not present
        3. Map ICD codes to categories
        4. Create binary predictors
        5. Train model with cross-validation

    Args:
        filepath (str): Path to input pickle file.
        target_column (str): Target variable name.
        as_grouped (bool): Whether grouping is applied.
        nvf (bool): Flag for NVF-specific processing.
        code_to_label (dict): Mapping from ICD codes to category labels.
        group_cols (list): Columns used for grouping features.
        model_cols (list): Candidate feature columns.
        is_sum (bool): Whether to use summed features instead of binary.
        split_size (float): Train/test split ratio.
        threshold (float): Classification threshold.
        n_folds (int): Number of CV folds.
        scorer (str): Scoring metric for cross-validation.

    Returns:
        dict: Dictionary containing trained model and predictions.
    """
    # Load data
    df = pd.read_pickle(filepath)
    df_dag = df.copy()

    # Prepare target column
    if target_column not in df_dag.columns:
        print("creating target_column")
        df_dag = create_target_variables(df_dag)

    # Map codes to categories
    df_dag['category'] = df_dag['code'].map(code_to_label)
    # Create predictors
    cat_data, group_cols, model_cols = create_binary_predictors(
        df_dag, group_cols, model_cols, as_sum=is_sum
    )
    print(cat_data["VersID"].shape)
    # Train model
    grouped_models = dict()

    model_dict = train_model(
        cat_data,
        target_column,
        as_grouped,
        is_sum,
        nvf,
        group_cols,
        split_size,
        grouped_models,
        threshold,
        n_folds,
        scorer
    )

    return model_dict