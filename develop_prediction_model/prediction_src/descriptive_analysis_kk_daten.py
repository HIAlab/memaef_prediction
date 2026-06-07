import sys
from pathlib import Path
import pandas as pd
import re

from scipy.stats import chi2_contingency
from dateutil.relativedelta import relativedelta

def find_project_root(marker=".git"):
    path = Path.cwd()
    for parent in [path] + list(path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Marker '{marker}' not found in any parent directories.")
root_dir = find_project_root()

sys.path.append("..")
sys.path.append("../prediction_src")

from prediction_src.fit_prediction_model import *
from params import *


def compute_age_at_endo(group):
    """
    Compute age at first endometriosis diagnosis (ICD code N80) per group.

    Args:
        group (pd.DataFrame): Grouped dataframe per individual.

    Returns:
        pd.Series: Age at diagnosis repeated for each row in the group,
                   or NA if no diagnosis is present.
    """
    if group["endo"].iloc[0] != 1:
        return pd.Series([pd.NA] * len(group), index=group.index)

    n80_rows = group[group["code"].str.contains("N80", na=False)]
    if n80_rows.empty:
        return pd.Series([pd.NA] * len(group), index=group.index)

    age_at_endo = (
        n80_rows["Beginn"].dt.year.min() - group["Geburtsjahr"].iloc[0]
    )

    return pd.Series([age_at_endo] * len(group), index=group.index)


def return_age_in_years(df):
    """
    Compute age in years based on earliest event date and birth year.

    Args:
        df (pd.DataFrame): Dataframe containing 'Beginn' and 'Geburtsjahr'.

    Returns:
        int: Age in years.
    """
    x = df["Beginn"].min()
    y = df["Geburtsjahr"].min()
    return relativedelta(x, y).years


def return_current_age(df):
    """
    Compute current age based on latest available year and birth year.

    Args:
        df (pd.DataFrame): Dataframe containing 'AktuellesJahr' and 'Geburtsjahr'.

    Returns:
        int: Current age in years.

    """
    x = df["AktuellesJahr"].min()
    y = df["Geburtsjahr"].min()
    return relativedelta(x, y).years


def adenomyosis_analysis(df):
    """
    Analyze co-occurrence of adenomyosis (N80.0) and other endometriosis codes.

    Computes:
    - Presence of N80.0 vs other N80 codes
    - Contingency table
    - Conditional probabilities
    - Chi-square test of independence

    Args:
        df (pd.DataFrame): Dataframe with columns 'VersID' and 'code'.

    Returns:
        None: Prints results and displays contingency table.
    """
    table = df.groupby("VersID")["code"].apply(list)

    has_n80_0 = table.apply(lambda codes: any(c == "N80.0" for c in codes))

    pattern = re.compile(r"^N80\.[1-9]\d*$", re.IGNORECASE)
    has_n80_other = table.apply(
        lambda codes: any(bool(pattern.match(str(c))) for c in codes)
    )

    print(has_n80_other.value_counts())

    has_n80_0.name = "N80.0"
    has_n80_other.name = "N80 other"

    contingency = pd.crosstab(has_n80_0, has_n80_other)
    print(contingency)

    # Extract counts safely
    a = contingency.loc[False, False] if (False, False) in contingency.stack().index else 0
    b = contingency.loc[False, True] if (False, True) in contingency.stack().index else 0
    c = contingency.loc[True, False] if (True, False) in contingency.stack().index else 0
    d = contingency.loc[True, True] if (True, True) in contingency.stack().index else 0

    n80_0_total = c + d
    n80_other_total = b + d
    any_n80_total = b + c + d

    p_other_given_0 = d / n80_0_total if n80_0_total > 0 else 0
    p_0_given_other = d / n80_other_total if n80_other_total > 0 else 0
    p_any = any_n80_total / contingency.values.sum()
    p_only_nother = b / contingency.values.sum()

    print(
        f"Prob(Endometriosis | Adenomyosis): {round(p_other_given_0,3)},\n"
        f"Prob(Adenomyosis | Endometriosis): {round(p_0_given_other,3)},\n"
        f"Prob(any of both): {round(p_any,3)},\n"
        f"Prob(only endometriosis): {round(p_only_nother,3)}"
    )

    chi2, p, dof, expected = chi2_contingency(contingency)
    print("Chi2:", chi2, "p-value:", p)
