import pandas as pd


import sys
sys.path.append("..")
sys.path.append("../prediction_src")
from fit_prediction_model import *
from prediction_pipeline import *
from evaluate_prediction_model import *
from params import *

# import sys, os, Path
import os
from pathlib import Path
import re
import difflib
import numpy as np

def read_code_list(code_list_file_path, code_list_file_name, sheet_number, code_variable_name):
    """ Read in all code, e.g. ICD.
    
    Args:
            code_list_file_path (str): path to where list of all codes is stored
            code_list_file_name (str): excel file name of the list of all codes
            sheet_number (str): which excel file sheet number
            code_variable_name (str): The type of code, e.g. ICD

     Returns:
             code_list (List): a list of all codes
    
    """
    
    code_df = pd.read_excel(f"{root_dir}/{code_list_file_path}/{code_list_file_name}",
             sheet_name=sheet_number)

    code_list = code_df[code_variable_name].unique()

    return code_list

def merge_dates_and_codes(dict_data, case_data_name,diag_data_name, source, diag_variable_name):
    """ 
    Merge Krankenkassen data (KK data) from multiple tables per case ID and diagnosis type.
    
    Args:
        dict_data: ata structured as a dictionary
        case_data_name: Name of the Case data within dict_data
        diag_data_name: Name of the Diagnosis data within dict_data
        source (Str): Name of the common source of case_data_name and diag_data_name
        diag_variable_name (str): Diagnosis type, e.g. ICD.
        
        """
    print(f'{case_data_name}:{dict_data[case_data_name]["VersID"].nunique()}, {diag_data_name}:{dict_data[diag_data_name]["VersID"].nunique()}')
    merged_data = pd.merge(dict_data[case_data_name], dict_data[diag_data_name],
                        on = ["VersID","FallID"], how="left")
    merged_data["source"] = source
    merged_data["code"] = merged_data[diag_variable_name] 
    print(merged_data["VersID"].nunique())
    return merged_data

def fast_format(series):
    """ Formats a series of strings """
    # Ensure all values are strings
    codes = series.astype(str)

    # Create a boolean mask: matches one letter + 3-4 digits, no dot
    pattern = r'^[A-Za-z]\d{3,4}$'
    mask = codes.str.match(pattern)

    # Apply transformation only to matching values
    formatted = codes[mask].str.replace(
        r'^([A-Za-z])(\d{2})(\d+)$', r'\1\2.\3', regex=True
    )

    # Replace only matching rows with formatted version
    codes[mask] = formatted
    return codes