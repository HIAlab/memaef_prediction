# memaef_prediction

## Overview
This repository contains code for the development and evaluation of a prediction model for endometriosis in girls and young women with menstrual pain. This repository is part of the MeMäF Project ([visit website](https://innovationsfonds.g-ba.de/projekte/memaef.504)). 


## Structure
The repository contains one code area:

- `develop_prediction_model`

In it, there are several subdirectories:  
- `prediction_src` for functionalities
- `data` for local data storage
- `notebooks` for data preprocessing, descriptive analysis, model development, evaluation and application to the data of the new care form within MeMäF.

In the `params.py` file, several mapping dictionaries between EBM and ICD-10 codes and predictor names for different models and different predictor sets are stored, as well as lists of computable calibration and evaluation metrics.

memaf_prediction  
└── README.md  
└── .gitignore  
└── params 
└── develop_prediction_model  
    ├── prediction_src  
    │   ├── application_prediction  
    │   ├── calibration  
    │   ├── descriptive_analysis_kk_daten  
    │   ├── evaluate_prediction_model  
    │   ├── fit_prediction_model  
    │   ├── join_kk_data  
    │   ├── load_data  
    │   ├── prediction_pipeline  
    │   └── utils  
    ├── notebooks  
    │   ├── 00_Join_KK_Daten  
    │   ├── 01_Descriptive_Analysis  
    │   ├── 02_Fit_DAG  
    │   ├── 03_Calibration  
    │   ├── 04_Evaluation  
    │   └── 05_Application_to_NVF  
    ├── models  
    └── data  
        ├── raw  
        └── processed  
        └── results  



## How to run
The  `data` subdirectory is empty - make sure to store your data here in tabular format. Also, the `models` folder is empty - here the models should be saved.

The notebooks are numbered and can be run in sequential order. The final model from our work will be "DAG_model_youden_final".
