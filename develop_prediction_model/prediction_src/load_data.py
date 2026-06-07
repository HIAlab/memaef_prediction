# Contains functions for loading csv files from vandage with patient data

import pandas as pd
import logging

DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def load_participant(file_path: str) -> pd.DataFrame:
    """Load and preprocess participant file

    Args:
        file_path (str): path to csv file

    Returns:
        pd.DataFrame: preprocessed data frame
    """
    # read file
    df_participant = pd.read_csv(file_path, encoding="utf-8", sep=";")
    
    # data preprocessing
    # TODO: insert preprocessing here
    df_participant["consent_to_study"] = pd.to_datetime(pd.to_datetime(df_participant["consent_to_study"], utc=True).dt.strftime(DATE_TIME_FORMAT))
    return df_participant


def load_answer(file_path: str) -> pd.DataFrame:
    """Load and preprocess answer file

    Args:
        file_path (str): path to csv file

    Returns:
        pd.DataFrame: preprocessed data frame
    """
    # read file
    df_answer = pd.read_csv(file_path, encoding="utf-8", sep=";")

    # data preprocessing
    df_answer = df_answer.rename(columns = {
        "id": "answer_id"
    })
    
    # transform data to a standardized date format
    df_answer["date_created"] = pd.to_datetime(pd.to_datetime(df_answer["date_created"], utc=True).dt.strftime(DATE_TIME_FORMAT))
    df_answer["date_updated"] = pd.to_datetime(pd.to_datetime(df_answer["date_updated"], utc=True).dt.strftime(DATE_TIME_FORMAT))
    df_answer["journal_date"] = pd.to_datetime(pd.to_datetime(df_answer["journal_date"], utc=True).dt.strftime(DATE_TIME_FORMAT))
    
    # Filter for latest answer_id for multiple answers for the same question. We are using the latest answer for that. 

    # create column with latest change
    df_answer_copy = df_answer.copy()
    df_answer_copy["date_latest"] = df_answer_copy[["date_created", "date_updated"]].max(axis=1, skipna=True).fillna(pd.to_datetime("1990-01-01"))
    # groupby and selected the latest date
    flag_journal_date = ~(df_answer_copy["journal_date"].isna())
    # rows with journal dates
    answer_ids_journal = list(df_answer_copy[flag_journal_date].groupby(['pseudonym', 'questionnaire_id', 'question_id', 'journal_date']).agg({"date_latest":"idxmax"})["date_latest"].unique())
    # rows without journal dates
    answer_ids_other = list(df_answer_copy[~flag_journal_date].groupby(['pseudonym', 'questionnaire_id', 'question_id']).agg({"date_latest":"idxmax"})["date_latest"].unique())

    # match the answer id back
    df_answer_latest = df_answer.loc[answer_ids_journal + answer_ids_other,:]
    logging.info(f"Found {df_answer.shape} entries in original df_answer.")
    logging.info(f"Filter double entries. Found {df_answer_latest.shape} entries in filtered df_answer.")
    return df_answer_latest


def load_answer_question_answer(file_path: str) -> pd.DataFrame:
    """Load and preprocess answer question answer file

    Args:
        file_path (str): path to csv file

    Returns:
        pd.DataFrame: preprocessed data frame
    """
    # read file
    df_answer_question_answer = pd.read_csv(file_path, encoding="utf-8", sep=";")
    
    # data preprocessing
    # TODO: insert preprocessing here

    return df_answer_question_answer


def load_participant_filled_questionnaire(file_path: str) -> pd.DataFrame:
    """Load and preprocess participant filled questionnaire file

    Args:
        file_path (str): path to csv file

    Returns:
        pd.DataFrame: preprocessed data frame
    """
    # read file
    df_participant_filled_questionnaire = pd.read_csv(file_path, encoding="utf-8", sep=";")

    # rename columns
    df_participant_filled_questionnaire = df_participant_filled_questionnaire.rename(columns={"questionnaire": "questionnaire_id"})
    
    # data preprocessing
    df_participant_filled_questionnaire["date_created"] = pd.to_datetime(pd.to_datetime(df_participant_filled_questionnaire["date_created"], utc=True).dt.strftime(DATE_TIME_FORMAT))
    df_participant_filled_questionnaire["date_updated"] = pd.to_datetime(pd.to_datetime(df_participant_filled_questionnaire["date_updated"], utc=True).dt.strftime(DATE_TIME_FORMAT))

    return df_participant_filled_questionnaire


def load_translation_files(path_data_trans:str)-> tuple:
    """Load translation files 

    Args:
        path_data_trans (str): path to files

    Returns:
        tuple: df_questionnaire_trans, df_question_answer_translation, df_question_translation
    """
    # Read translation files
    df_questionnaire_translation = pd.read_csv(path_data_trans + "/questionnaire.csv", index_col="id").rename(columns={"shortname":"questionnaire_text"})
    df_question_answer_translation = pd.read_csv(f"{path_data_trans}/question_answer_translations.csv", index_col="question_answer_id")
    df_question_translation = pd.read_csv(f"{path_data_trans}/question_translations.csv", index_col="question_id")
    df_question_categories = pd.read_csv(f"{path_data_trans}/question_categories.csv", index_col="question_id")
    df_question_translation = df_question_translation.join(df_question_categories, how="left")
    # Add Missing as text as nan is multiple times in the data set
    df_question_answer_translation.loc["MISSING", "text"] = "MISSING"
    return df_questionnaire_translation, df_question_answer_translation, df_question_translation


def load_definition_files(path_data_trans:str)-> tuple:
    """Load files defining the questionnaire, question and question-answer

    Args:
        path_data_trans (str): path to files

    Returns:
        tuple: df_questionnaire, df_question, df_question_answer
    """
    df_question = pd.read_csv(f"{path_data_trans}/question.csv", index_col="id")
    df_questionnaire = pd.read_csv(f"{path_data_trans}/questionnaire.csv", index_col="id")
    df_question_answer = pd.read_csv(f"{path_data_trans}/question_answer.csv", index_col="id")
    return df_questionnaire, df_question, df_question_answer


def load_relation_files(path_data_relation:str) -> tuple:
    """Load files defining the questionnaire_question and question_answer_question file.

    Args:
        path_data_trans (str): path to files

    Returns:
        tuple: df_questionnaire_question, df_question_answer_question
    """
    df_questionnaire_question = pd.read_csv(f"{path_data_relation}/questionnaire_question.csv")
    df_question_answer_question = pd.read_csv(f"{path_data_relation}/question_answer_question.csv")
    return df_questionnaire_question, df_question_answer_question