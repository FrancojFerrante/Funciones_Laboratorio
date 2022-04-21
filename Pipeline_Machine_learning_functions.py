# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:30:04 2022

@author: fferrante
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:43:44 2022

@author: fferrante
"""

# Machine_learning_functions.py>

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score
from sklearn import metrics
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
plt.ioff()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from sklearn.datasets import load_iris, load_boston
from sklearn.metrics import accuracy_score, classification_report

# import libraries for charting and manipulations with datasets
import matplotlib.pyplot as plt
import random

from statsmodels.stats.outliers_influence import variance_inflation_factor

import Machine_learning_functions as mlf

import Graficar as graficar

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_score
import matplotlib.pyplot as plt
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold

def pipeline_cross_validation(df,ml_classifier,group_labels,group_column,features,k_fold,random_seed = 123,normalization=True,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance="",data_input=False,feature_selection=False):
    

    pipeline_list = []

    if data_input:
        pipeline_list.append(('imputer',KNNImputer()))
        
    if normalization:
        pipeline_list.append(('scaler',MinMaxScaler()))
        
    if feature_selection:
        pipeline_list.append(('feat_sel',RFECV(estimator=ml_classifier,n_jobs=-1,step=2,)))
        
    pipeline_list.append(('model',ml_classifier))
    kf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=1, random_state=random_seed)
    
    scoring = {'acc': 'accuracy',
               'prec_micro': 'precision_micro',
               'rec_micro': 'recall_micro',
               'auc':'roc_auc',
               'f1_score':'f1_micro'}
    
    pipeline = Pipeline(pipeline_list)

    # scores["estimator"] devuelve tantos Pipelines como n_splits en cross-validation
    scores = cross_validate(pipeline, df[features], df[group_column], scoring=scoring,
                         cv=kf, return_train_score=False,return_estimator=True)
    
    df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"])
    df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),"Regresión logística",str(k_fold),str(normalization),np.mean(scores['test_acc']),np.mean(scores['test_prec_micro']),np.mean(scores['test_rec_micro']),np.mean(scores['test_auc']),np.mean(scores['test_f1_score'])]
   
    return (scores,df_resultados)


cwd = 'D:\Franco\Doctorado\Laboratorio\BasesFluidezFonoYSeman\Scripts\ProyectoPablo'
# cwd = 'C:\Franco\Doctorado\Laboratorio\BasesFluidezFonoYSeman\Scripts\ProyectoPablo'

data = pd.read_csv(cwd + '//features_nlp_acustica.csv')

# columnas_valores = ["promedio","errores","cfv","sustantivo_distancia","cantidad_palabras_correctas"]

columnas_valores = ["promedio","minimo","maximo","std","mediana","curtosis","skewness","cfv","sustantivo_distancia","cantidad_palabras_correctas"]
# columnas_valores = ["cantidad","porcentaje","distancia"]

columnas_estadisticas = []
for columna in data.columns:
    if (any(columna_valores in columna for columna_valores in columnas_valores) & ("todo_distancia_concepto" not in columna) & \
        ("_errores_francos" not in columna) & ("_entero_correctas_" not in columna) & ("_correctas_indivi" not in columna) & \
            ("_num_phon" not in columna) & ("_sustantivo_" not in columna) & ("_p_" not in columna) & \
            ("_f_" not in columna) & ("_a_" not in columna) & ("_s_" not in columna)):
        columnas_estadisticas.append(columna)
            
lista_aux = columnas_estadisticas + ["Codigo"]
lista_aux = lista_aux + ["Grupo"]
data = data[lista_aux]



df_controles = data[data["Grupo"] == "CTR"]
df_alzheimer = data[data["Grupo"] == "AD"]

df_combinado_ctr_ad = df_controles.append(df_alzheimer)
df_combinado_ctr_ad["Grupo"].replace({"CTR": 0, "AD": 1}, inplace=True)

df_combinado_labels_list = ["AD"]

# Borro las columnas donde todas las filas tienen el mismo valor
nunique = df_combinado_ctr_ad.nunique()
cols_to_drop = nunique[nunique == 1].index
df_combinado_ctr_ad = df_combinado_ctr_ad.drop(cols_to_drop, axis=1)

for col in cols_to_drop:
    while col in columnas_estadisticas: columnas_estadisticas.remove(col)

# df_combinado_ctr_ad[columnas_estadisticas]=(df_combinado_ctr_ad[columnas_estadisticas]-df_combinado_ctr_ad[columnas_estadisticas].min())/(df_combinado_ctr_ad[columnas_estadisticas].max()-df_combinado_ctr_ad[columnas_estadisticas].min())
# df_combinado_list = [df_combinado_ctr_ad]

k_folds=[3]

logisticRegr = LogisticRegression(max_iter=100000,)

df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"])

  
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
accuracy = []
precision = []
recall = []
# auc = []
f1 = []
# cm_total = np.full((len(group_labels), len(group_labels)),0)




features = [[x for x in df_combinado_ctr_ad.columns if (("granularidad" in x) | ("cfv" in x))],
            [x for x in df_combinado_ctr_ad.columns if (("log_frq" in x) | ("num_syll" in x) | ("sa_NP" in x) | ("familiarity" in x)\
                                         | ("imageability" in x) | ("concreteness" in x) | ("cantidad_palabras_correctas" in x))],
            [x for x in df_combinado_ctr_ad.columns if (("log_frq" in x) | ("num_syll" in x) | ("sa_NP" in x) | ("familiarity" in x)\
                                         | ("imageability" in x) | ("concreteness" in x) | ("cfv" in x)\
                                         | ("granularidad" in x) | ("cantidad_palabras_correctas" in x))],
           [x for x in df_combinado_ctr_ad.columns if (("speech_rate" in x) | ("silence_durations" in x) | ("log_silence_durations" in x) | ("#syl-#intervals" in x)\
                                                    | ("#fp-#syl" in x) | ("duration_syl-duration_segmment" in x)\
                                                    | ("duration_fp-duration_syl" in x) | ("total_speech" in x) | ("total_duration" in x))],
           [x for x in df_combinado_ctr_ad.columns if (("log_frq" in x) | ("num_syll" in x) | ("sa_NP" in x) | ("familiarity" in x)\
                                                    | ("imageability" in x) | ("concreteness" in x) | ("cfv" in x)\
                                                    | ("granularidad" in x) | ("cantidad_palabras_correctas" in x)\
                                                    | ("speech_rate" in x) | ("silence_durations" in x) | ("log_silence_durations" in x) | ("#syl-#intervals" in x)\
                                                    | ("#fp-#syl" in x) | ("duration_syl-duration_segmment" in x)\
                                                    | ("duration_fp-duration_syl" in x) | ("total_speech" in x) | ("total_duration" in x))]]

feature_file = ["NLP","Psycholinguistics","Psycholinguistics-NLP","Acoustics","All"]
titulo = ["NLP","Psycholinguistics","Psycholinguistics-NLP","Acoustics","All"]

df_combinado_ctr_ad_shuffle = df_combinado_ctr_ad.copy(deep=True)
df_combinado_ctr_ad_shuffle["Grupo"] = df_combinado_ctr_ad["Grupo"].sample(frac=1).values


bases = [df_combinado_ctr_ad,df_combinado_ctr_ad_shuffle]
label_grupos = ["AD_CTR","AD_CTR_shuffled"]



# %% Pruebo el clasificador
(scores,df_resultados) = logistic_regression_cross_validation(df_combinado_ctr_ad,LogisticRegression(max_iter=100000,),["CN","AD"], "Grupo", features[4],3,random_seed = 123,normalization=True,data_input=True,feature_selection=True)

# %% Obtengo las features elegidas en el primer Pipeline

# scores["estimator"] devuelve tantos Pipelines como n_splits en cross-validation
# Obtengo el paso "feat_sel" que hace referencia al RFECV.
features_selected = scores["estimator"][0].named_steps['feat_sel']

# Obtengo las features elegidas en el primer Pipeline.
print(df_combinado_ctr_ad[features[4]].columns[features_selected.get_support()])








