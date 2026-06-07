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
    
    code_df = pd.read_excel(f"{root_dir}/{code_list_file_path}/{code_list_file_name}",
             sheet_name=sheet_number)

    code_list = code_df[code_variable_name].unique()

    return code_list

def merge_dates_and_codes(dict_data, case_data_name,diag_data_name, source, diag_variable_name):
    print(f'{case_data_name}:{dict_data[case_data_name]["VersID"].nunique()}, {diag_data_name}:{dict_data[diag_data_name]["VersID"].nunique()}')
    merged_data = pd.merge(dict_data[case_data_name], dict_data[diag_data_name],
                        on = ["VersID","FallID"], how="left")
    merged_data["source"] = source
    merged_data["code"] = merged_data[diag_variable_name] 
    print(merged_data["VersID"].nunique())
    return merged_data

def fast_format(series):
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