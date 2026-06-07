import pandas as pd 
import numpy as np
import sys
from pathlib import Path
import pickle

# Add the project directory to sys.path
code_path = Path('/mnt/c/Users/Juliana/dev/memaef_prediction/')
sys.path.insert(0, str(code_path))
sys.path.insert(0, str(code_path / "develop_prediction_model"))

from prediction_src.load_data import load_translation_files 
import prediction.papermodels.ashrafi as ashrafi
import prediction.papermodels.fauconnier as fauconnier
from evaluate_prediction_model import calc_eval_metrics
from calibration import classification_metrics
from numpy.core.multiarray import array as array
from prediction.models import LogRegModel
from prediction.coeff_func import or_to_coeff
import os
from sklearn.base import BaseEstimator, ClassifierMixin

def load_prediction_tables(experiment_name="Vandage_20241023",
                           path_data="/mnt/c/Users/Juliana/dev/memaef/data",
                           questionnaire_numbers=["14_t0b"]):
    """
    Load prediction tables and corresponding question translation data.

    Args:
        experiment_name (str): Name of the experiment folder.
        path_data (str): Base path to the data directory.
        questionnaire_numbers (list): List of questionnaire file identifiers.

    Returns:
        tuple:
            df_t0b (pd.DataFrame): Concatenated questionnaire data.
            df_question_translation (pd.DataFrame): Question translation mapping.
    """
    path_trans_data = f"{path_data}/frageboegen/csv"
    _, _, df_question_translation = load_translation_files(path_trans_data)

    path_processed_data = f"{path_data}/1_processed_data/{experiment_name}"
    df_t0b = []

    for questionnaire_number in questionnaire_numbers:
        df_t0 = pd.read_csv(
            path_processed_data + f"/{questionnaire_number}.csv",
            index_col=[0],
            header=[0, 1],
            dtype=str
        )
        df_t0b.append(df_t0)

    df_t0b = pd.concat(df_t0b).reset_index(drop=False)

    return df_t0b, df_question_translation


def create_translation_map(df_t0b, df_question_translation, table_author):
    """
    Create a mapping from questionnaire columns to model predictors.

    Args:
        df_t0b (pd.DataFrame): Raw questionnaire data.
        df_question_translation (pd.DataFrame): Translation table for questions.
        table_author (pd.DataFrame): Author-specific mapping table.

    Returns:
        dict: Mapping from questionnaire column identifiers to predictors.
    """

    q_translation = {}
    for col in df_t0b.columns.levels[0]:
        c = col.split(".")[-1]
        
        if (col != "pseudonym") and (int(c) in df_question_translation.index):
            q_translate = df_question_translation.loc[int(c), "text"]
            q_translation[col] = q_translate

    q_translation = (
        pd.DataFrame(q_translation, index=[0])
        .T.reset_index()
        .rename(columns={"index": "question_number", 0: "question_text"})
    )



    df_merge = pd.merge(
        q_translation,
        table_author,
        left_on=["question_text"],
        right_on=["equivalent_question"],
        how="right"
    )

    
    value_to_number = {}
    for col in df_t0b.columns:
        col_values = df_t0b[col].dropna().unique()
        for val in col_values:
            value_to_number[val] = col[1]


    q_dict = {}
    for _, data in df_merge.dropna(subset=["question_number"]).iterrows():
        qn = data["question_number"]
        predictor = data["predictor"]
        answer = data.get("equivalent_answer")
        number = value_to_number.get(answer, "")
        if number != "":
            q_dict[f"{qn}_{number}"] = predictor
        else:
            q_dict[f"{qn}_Answer"] = predictor

    return q_dict


def select_and_filter_df_for_prediction(df_t0b, q_dict, author_table, author):
    """
    Select and rename relevant columns for prediction based on mapping.

    Args:
        df_t0b (pd.DataFrame): Raw questionnaire data.
        q_dict (dict): Mapping from columns to predictors.
        author_table (pd.DataFrame): Author-specific mapping table.
        author (str): Author/model identifier.

    Returns:
        pd.DataFrame: Filtered and renamed dataframe ready for prediction.
    """
    cols_to_keep = list(q_dict.keys())

    df_t0b.columns = [f"{lvl0}_{lvl1}" for lvl0, lvl1 in df_t0b.columns]
    pseudonym_col = [col for col in df_t0b.columns if "pseudonym" in col]

    existing_cols = [col for col in cols_to_keep if col in df_t0b.columns]
    df_t0b = df_t0b[existing_cols + pseudonym_col].copy()

    merged = []
    sorted_cols = []

    for col in df_t0b.columns:
        if col in q_dict:
            merged.append((col, q_dict[col]))
            sorted_cols.append(col)

    merged_col_names = [c[-1] for c in merged]
    df_t0b = df_t0b[sorted_cols + pseudonym_col]
    df_t0b.columns = merged_col_names + pseudonym_col

    return df_t0b


def get_translation_map_for_beschwerde_columns(path_data, questionnaire_numbers):
    df_t0a, _ = load_prediction_tables(experiment_name = 'AppData_2026-01-26',
        path_data = path_data,
        questionnaire_numbers = questionnaire_numbers)
    
    lla = df_t0a.columns
    
    l17a = [l for l in lla if l[0] == "T0B.BESCHW.17"]
    
    beschwerde_col_translation_map = df_t0a[l17a].apply(lambda x:x.dropna().unique()).T.reset_index().iloc[:,1:]
    beschwerde_col_translation_map.columns = ["col_number","translation"]
    beschwerde_col_translation_map = beschwerde_col_translation_map[beschwerde_col_translation_map["col_number"]!="Answer_Flag"]
    translation_map = dict(
        zip(beschwerde_col_translation_map["col_number"].astype(str),
            beschwerde_col_translation_map["translation"])
    )


    return translation_map

def translate_beschwerde_columns(df,path_data, questionnaire_numbers):
    translation_map = get_translation_map_for_beschwerde_columns(path_data, questionnaire_numbers)
    llb = df.columns
    
    l17b = [l for l in llb if l[0] == "T0B.BESCHW.17"]
    #l17b2 = [l for l in llb if l[0] in ["T0B.MENST.59", "T0B.MENST.69"]]
    #l17b = l17b + l17b2
    
    for col in df[l17b].columns:
        lvl0, lvl1 = col
        
        # skip non-numeric or special columns
        if lvl1 != "Answer_Flag":
            if lvl1 in translation_map:
                label = translation_map[lvl1]
                df[col] = df[col].where(df[col] == "1", np.nan)
                df[col] = df[col].where(df[col] != "1", label)

    return df

def prep_model_validation(dump_folder_name="Vandage_20241023",
                         path_data="/mnt/c/Users/Juliana/dev/memaef/data",
                         questionnaire_numbers=["14_t0b"],
                         author="fauconnier",
                         translate_beschwerde_cols=False):
    """
    Prepare dataset and model for validation.

    Args:
        dump_folder_name (str): Experiment folder name.
        path_data (str): Base data path.
        questionnaire_numbers (list): Questionnaire identifiers.
        author (str): Model author identifier.

    Returns:
        tuple:
            df_t0b_author (pd.DataFrame): Prepared dataset.
            model: Loaded prediction model.
    """
    df_t0b, df_question_translation = load_prediction_tables(
        experiment_name=dump_folder_name,
        path_data=path_data,
        questionnaire_numbers=questionnaire_numbers
    )


    if translate_beschwerde_cols:
        df_t0b = translate_beschwerde_columns(df_t0b,path_data, questionnaire_numbers)

    if (author != "fauconnier") and (author != "ashrafi"):
        table_author = pd.read_csv(
            f"../{author}_nvf_trans.csv",
            encoding='latin-1',
            sep=";",
            on_bad_lines='skip'
        )
    else:
        table_author = pd.read_csv(
            f"{path_data}/3_prediction/table_{author}.csv",
            encoding='latin-1',
            sep=";",
            on_bad_lines='skip'
        )

    if author == "fauconnier":
        model = fauconnier.FauconnierLogRegModel(False)
    elif author == "ashrafi":
        model = ashrafi.AshrafiLogRegModel(False)
    else:
        with open(f"{author}.pkl", "rb") as f:
            dag_model_dict = pickle.load(f)
        model = dag_model_dict[list(dag_model_dict.keys())[0]]["model"]
    q_dict = create_translation_map(df_t0b, df_question_translation, table_author)
    df_t0b_author = select_and_filter_df_for_prediction(df_t0b, q_dict, table_author, author)

    if author == "fauconnier":
        if "positional_dyspareunia" not in df_t0b_author.columns:
            df_t0b_author["positional_dyspareunia"] = df_t0b_author["dyspareunia"]
        if "disabling_pain" not in df_t0b_author.columns:
            df_t0b_author["disabling_pain"] = df_t0b_author["worsening_pain"]

    return df_t0b_author, model


def fix_pseudonym(df):
    """
    Ensure pseudonym column is filled and consistent.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Updated dataframe with fixed pseudonym column.
    """
    df.reset_index(inplace=True)
    df["pseudonym_"].fillna(df["index"], inplace=True)
    df.drop(columns="index", inplace=True)
    return df


def replace_answers(df):
    """
    Replace categorical survey answers with numeric values.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Transformed dataframe.
    """

    df = df.replace(4,0)
    df = df.replace(7,1)

    replacements1 = {        
        '60':'Alle 1 bis 2 Stunden',
        '61':'Alle 2 bis 4 Stunden',
        '62':'Frühestens alle 4 Stunden',
        '34':"Ich hatte noch keinen Geschlechtsverkehr",
        "4":0,
        "4.0":0,
        "7":1,
        "7.0":0,
        "34":0,
        }
    replacements2 = {
        'Ich hatte noch keinen Geschlechtsverkehr': 0,
        'Ja, mit Einnahmepause (z.B. Pille, Ring)': 1,
        'Ja, durchgehend (z.B. Pille, Hormonspirale)': 1,
        'Durchfall, Übelkeit oder Erbrechen': 1,
        'Hüftschmerzen':1,
        'Schlafprobleme':1,
        'Bauchkrämpfe':1,
        'Schmerzen beim Stuhlgang':1,
        'Schmerzen beim Wasserlassen':1,
        'Kopfschmerzen':1,
        'Schmerzen im unteren Rücken':1,
        'MISSING': 0,        
        'Alle 1 bis 2 Stunden': 1,
        'Alle 2 bis 4 Stunden': 0,
        'Frühestens alle 4 Stunden': 0
    }
    df = df.replace(replacements1)
    df = df.replace(replacements2)
    return df


def prepare_dag_model_df(df, df_eligible,eligible_col, dag_model,split_col_names=True):
    """
    Prepare dataframe for DAG model prediction.

    Args:
        df (pd.DataFrame): Input questionnaire data.
        df_eligible (pd.DataFrame): Eligibility dataframe.
        eligible_col (str): Eligibility column name.
        dag_model: Trained DAG model.

    Returns:
        pd.DataFrame: Model-ready dataframe.
    """
    df = fix_pseudonym(df)
    
    df = replace_answers(df)
    pseudonym_col = df.filter(like="pseudonym").columns
    
    df_work = df.drop(columns=pseudonym_col).copy()
    df_work = df_work.where(df_work != "NaN", np.nan)
    df_work = df_work.fillna(0)
    
    df_work = df_work.replace("0.0",0)
    
    df_work = df_work.replace("Nein",0)
    #df_work = df_work.replace("Ja",1)
        
    for i in range(df_work.shape[1]):
        col = df_work.iloc[:, i]
    
        if col.astype(str).str.contains(r".0", regex=True).any():
           
            df_work.iloc[:,i] = (df_work.iloc[:,i].fillna(0)
                                            .astype(float)
                                            .astype(int))
            threshold = 5 if col.name in ["Menstruationsschmerzen", "dysmenorrhea"] else 7
             
            num = pd.to_numeric(col, errors="coerce")
            
            df_work.iloc[:, i] = np.where(
                num.isna(),
                pd.NA,
                np.where(num > threshold, 1, 0)
            )    

            # Convert to numeric if possible (non-numeric becomes NaN)
        col_numeric = pd.to_numeric(col, errors="coerce")
        
        # Replace everything not 0 or 1 with 1
        df_work.iloc[:, i] = np.where(
            (col_numeric == 0) | (col_numeric == 1),
            col_numeric,
            1
        )

    
    df_work[pseudonym_col] = df[pseudonym_col]
  #  df_work.rename(columns={"dysmenorrhea":"dysmenorrhoea"},inplace=True)
    
    for i in dag_model.coef_.index:

        if i not in df_work.columns:
            df_work[f"{i}_model"] = 0
        else:
            if type(df_work[i]) != pd.Series:
                df_work[f"{i}_model"] = df_work[i].sum(axis=1)
            else:
                df_work[f"{i}_model"] = df_work[i]
        df_work[f"{i}_model"] = pd.Series(df_work[f"{i}_model"] > 0).astype(int)

    df_model = df_work.filter(regex=f"model|pseudonym_").groupby(pseudonym_col[0], as_index=False).apply(lambda x:x.sum() >0)
    df_model = pd.merge(df_model, df_eligible[["pseudonym",eligible_col]], how='left', left_on=pseudonym_col[0], right_on="pseudonym")
    df_model = df_model.drop(columns="pseudonym")
    df_model.columns = [col.split("_model")[0] for col in df_model.columns]
    df_model.rename(columns={"nvf2":"nvf2_eligible"},inplace=True)
    df_model[eligible_col]=df_model[eligible_col].fillna(0)
    return df_model

def return_nvf_for_eval_metrics(model_dict,df_orig,eval_var,met_list):
    df = df_orig.copy()
    model = model_dict[list(model_dict.keys())[0]]['model']
    nvf_probs_1 = model.predict_proba(df[model.coef_.index].values)[:, 1]
    nvf_probs_0 = model.predict_proba(df[model.coef_.index].values)[:, 0]
    nvf_preds = model.predict(df[model.coef_.index].values)
    
    df["risk_preds"] = nvf_preds
    df["risk_probs_1"] = nvf_probs_1
    df["risk_probs_0"] = nvf_probs_0
    
    nvf_dict = model_dict.copy()
    
    nvf_dict[list(nvf_dict.keys())[0]]["prob_1_test"] = nvf_probs_1
    nvf_dict[list(nvf_dict.keys())[0]]["y_test_true"] = df[eval_var]

    metrics = calc_eval_metrics(nvf_dict,met_list).round(3)
    class_metrics = classification_metrics(nvf_dict,model.threshold).round(3)
    return nvf_dict, metrics, class_metrics,df

def prep_nvf_diagnosis_df(df):
    dfe = df.filter(regex="pseudonym|DIAG.80").copy()
    
    dfe.columns = ['_'.join(col).strip() for col in dfe.columns.values]
    
    dfe= dfe.filter(regex="pseudonym|DIAG.80_Answer$").copy()

    
    dfe.fillna(4,inplace=True)
    
    for col in dfe.filter(regex="DIAG.80_Answer$"):
        dfe[col] = dfe[col].astype(float).astype(int)
    
    dfe.replace(4,0,inplace=True)
    dfe.replace(7,1,inplace=True)

    
    dfe["diagnose"] = dfe.drop(columns="pseudonym_").sum(axis=1)
    dfe = dfe.groupby("pseudonym_")["diagnose"].apply(lambda x:x.sum()).reset_index()
    dfe["diagnose"] = pd.Series(dfe["diagnose"] >0).astype(int)    
    dfe.rename(columns={"pseudonym_":"pseudonym"},inplace=True)
    return dfe

def return_literature_nvf_results(basis_dict, model, model_name,eval_var, MET_LIST,df_t0b_literature, df_eval_var,update_dict_name):
    literature_dict = dict()
    literature_dict["literature_dict"] = basis_dict[list(basis_dict.keys())[0]].copy()
    
    literature_dict[list(literature_dict.keys())[0]]["model"] = model #AshrafiLogRegModelNew()
    
    
    literature_dict[list(literature_dict.keys())[0]]["model_name"] = model_name#"ashrafi"
    
    literature_dict[list(literature_dict.keys())[0]]["model"].model.coef_ = pd.Series(literature_dict[list(literature_dict.keys())[0]]["model"]._coeffs)
    
    literature_dict[list(literature_dict.keys())[0]]["model"].coef_ = pd.Series(literature_dict[list(literature_dict.keys())[0]]["model"]._coeffs)
    
    df_literature = prepare_dag_model_df(df_t0b_literature, df_eval_var,eval_var,literature_dict[list(literature_dict.keys())[0]]["model"].model)
    
    literature_dict[list(literature_dict.keys())[0]]["X_test_true"] = df_literature[literature_dict[list(literature_dict.keys())[0]]["model"].coef_.index]
    literature_dict[list(literature_dict.keys())[0]]["y_test_true"] = df_literature[eval_var]
    
    
    df_literature = df_literature.loc[:, ~df_literature.columns.duplicated()]
    
    literature_dict = update_cutoff_dict(literature_dict, update_dict_name)#"ashrafi_nvf_eligible")
    
    
    nvf_dict_literature,nvf_metrics_literature, nvf_class_metrics_literature,df_application_literature = return_nvf_for_eval_metrics(literature_dict,df_literature,eval_var,MET_LIST)
    
    return nvf_dict_literature,nvf_metrics_literature, nvf_class_metrics_literature,df_application_literature

class AshrafiLogRegModelNew(LogRegModel, BaseEstimator, ClassifierMixin):

    def __init__(self, fit_intercept=False):
        coeffs = or_to_coeff("../../evaluate_existing_models/prediction/papermodels/ashrafi_or.csv")
        super().__init__(fit_intercept, coeffs)

        self.threshold = 0.5
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        # nothing to fit — fixed coefficients
        return self

    def predict_proba(self, X):
        """
        Same interface as SMWrapperClassifier
        Returns: [P(class0), P(class1)]
        """
        X = pd.DataFrame(X).astype(float)

        logit = self.model.intercept_ + np.dot(
            X.values,
            np.array(list(self._coeffs.values()))
        )
    
        probs = 1 / (1 + np.exp(-logit))
    
        return np.column_stack([1 - probs, probs])
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def set_class_threshold(self, threshold):
        self.threshold = threshold



class FauconnierLogRegModelNew(LogRegModel, BaseEstimator, ClassifierMixin):

    def __init__(self, fit_intercept=True):

        intercept = -7.11592

        coeffs = {
            "age": 0.07954,
            "dysmenorrhea": 0.47058,
            "pelvic_pain": 1.01086,
            "intense_pain": 0.12454,
            "worsening_pain": 2.06677,
            "pain_before_period": 0.41621,
            "stabbing_pain": -0.50397,
            "lower_back_pain": -0.23761,
            "leg_hip_pain": -1.01338,
            "disabling_pain": 1.19157,
            "pain_affects_mobility": 0.36398,
            "dyspareunia": -0.9589,
            "positional_dyspareunia": -0.76672,
            "interruption_of_sexual_intercourse": 1.5806,
            "painful_bowel_movements": 1.09474,
            "bowel_spasms": 1.0158,
            "diarrhea_constipation": 0.35057,
            "painful_urination": 1.70728,
            "bladder_pain": 0.82828,
            "sciatica": -0.90821,
            "right_shoulder_pain": -0.4595,
            "infertility": 2.25768
        }

        super().__init__(fit_intercept, coeffs, intercept)

        self.threshold = 0.5
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):

        X = pd.DataFrame(X).astype(float)

        logit = self.model.intercept_ + np.dot(
            X.values,
            np.array(list(self._coeffs.values()))
        )

        probs = 1 / (1 + np.exp(-logit))

        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def set_class_threshold(self, threshold):
        self.threshold = threshold
