# Premier Personalized Service Package Recommendation Model

This bachelor's thesis focuses on the development of a personalized sales scenario system for the premium service package "Sber Premier" for Sberbank clients.
The repository contains a pipeline for data preparation, feature generation, model training, and customer scoring. The work is under NDA, so not all data and code fragments will be visible.

## Project description

Goal of the work - to develop a model of personalized sales scenarios for the premium service package that forms for each client an individual offer, considering their potential benefit, relevant products and optimal communication channel.

The project consists of several parts:
1) preparation and processing of initial data
2) formation of the target variable
3) generation and cleaning of features
4) training of machine learning models
5) evaluation of the quality of models
6) scoring of clients for subsequent generation of recommendations

## Repository structure

```text

Premier-Personalized-Service-Package-Recommendation-Model/

├── Data_pipeline/

│   └── final_script_optim.ipynb

│

├── MultiClass_model/

│   ├── 1_Target_version_1.ipynb

│   ├── 1_Target_version_2.ipynb

│   ├── 2_Features_version_1.ipynb

│   ├── 2_Feaures_version_2.ipynb

│   ├── 3_Model_version_1.ipynb

│   ├── 3_Model_version_2.ipynb

│   ├── 3_Model_version_4_final.ipynb

│   ├── 3_Model_version_5.ipynb

│   ├── 4_Scoring_version_1.ipynb

│   ├── 4_Scoring_version_2.ipynb

│   └── bpm_features.py

│

├── Response_model/

│   ├── 1_Target_version_1.ipynb

│   ├── 1_Target_version_2.ipynb

│   ├── 2_Features_version_2.ipynb

│   ├── 3_CatBoost_version_2.ipynb

│   ├── 3_Forest_version_2.ipynb

│   ├── 3_LightGBM_version_2.ipynb

│   ├── 3_XGBoost_version_2.ipynb

│   ├── 4_Scoring_version_1.ipynb

│   ├── 4_Scoring_version_2.ipynb

│   └── bpm_features.py

│

└── README.md
```

## Main modules

### Data_pipeline
The folder contains a notebook for data preparation and optimization of the overall processing pipeline.

### MultiClass_model
A module for solving multi-class classification problems.
Includes the following steps:
1) Preparing the target variable;
2) Generating and processing features;
3) Training the model in several versions;
4) Final customer scoring.

The module also contains the bpm_features.py file, which contains functions for calculating features.

### Response_model
A module for building response models that estimate the likelihood of a customer's response.
This folder contains experiments with various machine learning algorithms:
1) CatBoost
2) Random Forest
3) LightGBM
4) XGBoost

Также присутствуют ноутбуки для подготовки таргета, признаков и скоринга.

## Technologies used
The work was written primarily in Python. Spark was used for data collection.

Main libraries and tools:
1) pandas
2) numpy
3) scikit-learn
4) catboost
5) lightgbm
6) xgboost
7) optuna
8) shap
9) matplotlib
10) seaborn
11) pyspark
12) tqdm.

## Installation and launch
Clone the repository:
```bash
git clone https://github.com/ART1X1/Premier-Personalized-Service-Package-Recommendation-Model.git
cd Premier-Personalized-Service-Package-Recommendation-Model
```

You won't be able to launch it because there is no data.

## Model quality assessment
Standard classification metrics are used to evaluate models:
1) ROC-AUC
2) F1-score
3) precision
4) recall
5) classification report.

The project also uses SHAP to interpret the impact of features on model predictions.

## Plans for developing personalized sales scenarios
1) Improving the quality of the response model
2) Developing a multiclass model for identifying life situations
3) Improving the selection of personalized products

## Author
Bolotin Platon
