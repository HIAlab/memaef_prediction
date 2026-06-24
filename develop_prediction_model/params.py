import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from evaluate_prediction_model import calibration_intercept_slope

TARGET_COLUMN = "endo"#"no_adeno"
TARGET_COLUMN_ALL_ENDOS = "all_endos"
TARGET_COLUMN_ONLY_ADENO = "only_adeno"
THRESHOLD = 0.5
SPLIT_SIZE = 0.7
N_FOLDS = 5
SCORER = "roc_auc"

MENS_MAP = {"Menstruationsschmerzen": ["N94","N94.1","N94.2","N94.3","N94.4","N94.5","N94.6","N94.8"],}
MENS_TO_LABEL = {code: label for label, codes in MENS_MAP.items() for code in codes}
MENS_COLS = list(pd.Series(MENS_TO_LABEL.values()).unique())

CODE_MAP = {
    "Dyspareunie": ["F52", "F52.6"],
    "Verstopfung": ["K59.0", "K59.01", "K59.09"],
    "Durchfall": ["K59.1", "R19.7"],
    "Blutung": ["N92", "N92.0", "N92.1"],
    "Menstruationsschmerzen": ["N94","N94.1","N94.2","N94.3","N94.4","N94.5","N94.6","N94.8"],
    "Bauchschmerzen": ["R10","R10.0","R10.1","R10.2","R10.3","R10.4"],
    "Verdauungsstörungen": ["K59","R19","R19.5", "R19.8","R19.88"],
    "Urinieren": ["R30",  "R30.0",  "R30.1",  "R30.9"],
#    "METAMIZOL" : ["N02BB02"],
#    "NSAR":["G02CC01", "G02CC02"],
#     "Orale_Kontrazeptiva":
#     ["G03AC03",
# "G03AC08",
# "G03CA01",
# "G03DA06",
# "G03DB08",
# "L02AA03",
# "G03AA",
# "G03AA01",
# "G03AA02",
# "G03AA03",
# "G03AA04",
# "G03AA05",
# "G03AA06",
# "G03AA07",
# "G03AA08",
# "G03AA09",
# "G03AA10",
# "G03AA11",
# "G03AA12",
# "G03AA13",
# "G03AA14",
# "G03AA15",
# "G03AA16",
# "G03AA17",
# "G03AA18",
# "G03AB",
# "G03AB01",
# "G03AB02",
# "G03AB03",
# "G03AB04",
# "G03AB05",
# "G03AB06",
# "G03AB07",
# "G03AB08",
# "G03AB09",
# "G03AC",
# "G03AC01",
# "G03AC02",
# "G03AC03",
# "G03AC04",
# "G03AC05",
# "G03AC06",
# "G03AC07",
# "G03AC08",
# "G03AC09",
# "G03AC10"],

}

NVF_MAP = {
"Infertilitaet": ["N96", "N97"],
"NSAR": ["C01EB03",
"D11AX18",
"G02CC01",
"G02CC02",
"L01XX33",
"M01AB01",
"M01AB05",
"M01AE01",
"M01AE02",
"M01AE09",
"M01AE14",
"M01AE51",
"M01AE52",
"M01AE56",
"M01AE57",
"M01AH01",
"M02AA15",
"M02AA19",
"M02AA23",
"R02AX01",
"S01BC01",
"S01BC03",
"S01BC04"], 
"Orale_Kontrazeptiva":
    ["G03AC03",
"G03AC08",
"G03CA01",
"G03DA06",
"G03DB08",
"L02AA03",
"G03AA",
"G03AA01",
"G03AA02",
"G03AA03",
"G03AA04",
"G03AA05",
"G03AA06",
"G03AA07",
"G03AA08",
"G03AA09",
"G03AA10",
"G03AA11",
"G03AA12",
"G03AA13",
"G03AA14",
"G03AA15",
"G03AA16",
"G03AA17",
"G03AA18",
"G03AB",
"G03AB01",
"G03AB02",
"G03AB03",
"G03AB04",
"G03AB05",
"G03AB06",
"G03AB07",
"G03AB08",
"G03AB09",
"G03AC",
"G03AC01",
"G03AC02",
"G03AC03",
"G03AC04",
"G03AC05",
"G03AC06",
"G03AC07",
"G03AC08",
"G03AC09",
"G03AC10"],

"Kopfschmerzen": ["G44"],
"Rückenschmerzen": ["M54"],
"Praemenstruelle_Beschwerden": ["N94.3"],
"Bauch_und_Beckenschmerzen": ["R10"]
}

CODE_TO_LABEL = {code: label for label, codes in CODE_MAP.items() for code in codes}
GROUP_COLS = list(pd.Series(CODE_TO_LABEL.values()).unique())


CODE_TO_CODE = dict(zip(CODE_TO_LABEL.keys(),CODE_TO_LABEL.keys()))
IND_COLS = list(pd.Series(CODE_TO_CODE.values()).unique())

NVF_CODE_TO_LABEL = {code: label for label, codes in NVF_MAP.items() for code in codes}
NVF_CODE_TO_LABEL = CODE_TO_LABEL | NVF_CODE_TO_LABEL
NVF_GROUP_COLS = list(pd.Series(NVF_CODE_TO_LABEL.values()).unique())
NVF_GROUP_COLS = NVF_GROUP_COLS + GROUP_COLS

MENS_MODEL_COLS = ["VersID","code",TARGET_COLUMN,TARGET_COLUMN_ALL_ENDOS,TARGET_COLUMN_ONLY_ADENO] + MENS_COLS
GROUP_MODEL_COLS = ["VersID","code",TARGET_COLUMN,TARGET_COLUMN_ALL_ENDOS,TARGET_COLUMN_ONLY_ADENO] + GROUP_COLS
IND_MODEL_COLS = ["VersID","code",TARGET_COLUMN,TARGET_COLUMN_ALL_ENDOS,TARGET_COLUMN_ONLY_ADENO] + IND_COLS
NVF_MODEL_COLS = ["VersID","code",TARGET_COLUMN,TARGET_COLUMN_ALL_ENDOS,TARGET_COLUMN_ONLY_ADENO] + NVF_GROUP_COLS

MET_LIST = [
    (roc_auc_score, "AUC"),
    (brier_score_loss, "Brier Score Loss"),
    (calibration_intercept_slope, "Calibration Intercept"),
    (calibration_intercept_slope, "Calibration Slope")
]

CALIBRATION_MET_LIST = [
    (cutoff_fpr_constraint, "Constrained FPR Cutoff"),
    (cutoff_max_ppv, "Max PPV Cutoff"),
    (cutoff_youden, "Youden Cutoff"),
]