# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:43:44 2022

@author: fferrante
"""

# Machine_learning_functions.py>

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn import metrics
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
import random
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


def pipeline_cross_validation(df,ml_classifier,classifier_name,group_labels,group_column,features,k_fold, random_seed = None,normalization=True,data_input=False,feature_selection=False,multi=False,n_repeats = 1):
    
    print("Classifier: " + classifier_name)
    print("Groups: " + group_labels)
    print("Fold: " + str(k_fold))

    if (random_seed==None):
        random_seed = random.randint(0,np.iinfo(np.int32).max)
    
    transformers = []
    pipeline_list = []
    
    # Si hay imputación o normalización
    if data_input | normalization:
        if data_input:
            transformers.append(('imputer',KNNImputer()))
            
        if normalization:
            transformers.append(('scaler',MinMaxScaler()))
        pipe_transformer = Pipeline(steps=transformers)
        preprocessor = ColumnTransformer(transformers= [('preprocessor',pipe_transformer,features)])
        pipeline_list.append(('preprocessor',preprocessor))
        
    # Si hay feature selection
    if feature_selection:
        svm_model = svm.SVC(kernel = "linear",max_iter=100000)
        if not multi:
            pipeline_list.append(('feat_sel',SFS(estimator=svm_model,n_features_to_select="auto",tol=0.01,
                                                 direction="forward",n_jobs=-1,scoring='roc_auc',cv=5)))
        else:
            pipeline_list.append(('feat_sel',SFS(estimator=svm_model,n_features_to_select="auto",tol=0.01,
                                                 direction="forward",n_jobs=-1,scoring='accuracy',cv=5)))
        
        #if not multi:
        #    pipeline_list.append(('feat_sel',RFECV(estimator=svm_model,step=1,scoring='roc_auc'))) # Ver si reemplazar por forward
        #else:
        #    pipeline_list.append(('feat_sel',RFECV(estimator=svm_model,step=1,scoring='accuracy')))
                        
    # genero el pipeline
    pipeline_list.append(('model',ml_classifier))
    pipeline = Pipeline(pipeline_list)

    # Instancio para generar los folds de cv
    kf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=random_seed)
    
    # Funciones propias para contar con la información para armar las matrices de confusión
    def confusion_matrix_tn(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm[0, 0]

    def confusion_matrix_fp(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm[0, 1]

    def confusion_matrix_fn(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm[1, 0]
    
    def confusion_matrix_tp(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm[1, 1]
    
    def return_prob(y_true,y_pred):
        return y_pred
    
    # Armo las métricas
    if multi:
        scoring = {'acc': 'accuracy'}
    else:
        scoring = {
                     'acc': 'accuracy',
                     'prec_micro': make_scorer(precision_score, zero_division=0),
                     'rec_micro': make_scorer(recall_score, zero_division=0),
                     'auc': make_scorer(roc_auc_score),
                     'f1_score': make_scorer(f1_score, zero_division=0),
                     'true_neg':make_scorer(confusion_matrix_tn),
                     'false_pos':make_scorer(confusion_matrix_fp),
                     'false_neg':make_scorer(confusion_matrix_fn),
                     'true_pos':make_scorer(confusion_matrix_tp)
            #'prob':make_scorer(return_prob,needs_proba=True)   
                   }
    
    # scores["estimator"] devuelve tantos Pipelines como n_splits en cross-validation
    scores = cross_validate(pipeline, df[features], df[group_column].values.ravel(), scoring=scoring,
                         cv=kf, return_train_score=False,return_estimator=True,verbose=1,n_jobs=-1)

    if not multi:    
        # Armo el DataFrame con los resultados
        df_prob = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","df_index","target","probability"])
        
        # La cantidad de filas es igual a todas las predicciones que se hicieron (para cada sujeto, en distintas combinaciones de folds)
        cant_filas_test = np.sum(scores["test_true_neg"])+np.sum(scores["test_true_pos"])+np.sum(scores["test_false_neg"])+np.sum(scores["test_false_pos"])
        df_prob["Random-Seed"] = np.full((cant_filas_test,),random_seed)
        df_prob["Feature"] = "Multi-features"
        df_prob["Grupo"] = group_labels
        df_prob["Clasificador"] = classifier_name
        df_prob["k-fold"] = str(k_fold)
        df_prob["Normalization"] = str(normalization)
        df_prob["i_iteration"] = 0
    
        
        
        # Armo el DataFrame de feature importance para devolverlo
        #df_features_importances = pd.DataFrame(columns=[["estimator"]+features])
        #for idx,estimator in enumerate(scores['estimator']):
            
        #    list_feature_importance = [idx] + list(np.abs(estimator["model"].coef_[0]))
        #    df_features_importances.loc[len(df_features_importances)] = list_feature_importance        
              
    
        # Genero la misma estructura de folds para obtener las probabilidades de predicción de los modelos ajustados anteriormente.
        # Esto lo hice así porque, en el momento en que lo codifiqué, sklearn no tenía forma de utilizar cross_validate y a la vez
        # obtener los scores de probabilidad. (Quise usar cross_validate para poder correrlo 1000 iteraciones paralelizando)
        kf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=random_seed)
        i = 0
        llenado=0
        for train_index, test_index in kf.split(df[features], df[group_column].values.ravel()):
            if hasattr(scores["estimator"][i], "decision_function"):
                resultado = scores["estimator"][i].decision_function(df[features].iloc[test_index])
                target = df.iloc[test_index][group_column]
                df_prob.loc[llenado:llenado+len(resultado)-1,"df_index"] = test_index
                df_prob.loc[llenado:llenado+len(resultado)-1,"probability"] = resultado
                df_prob.loc[llenado:llenado+len(resultado)-1,"target"] = target.values
                df_prob.loc[llenado:llenado+len(resultado)-1,"i_iteration"] = i
    
                llenado = llenado+len(resultado)
            else:
                resultado = scores["estimator"][i].predict_proba(df[features].iloc[test_index])
                target = df.iloc[test_index][group_column]
                df_prob.loc[llenado:llenado+len(resultado)-1,"df_index"] = test_index
                df_prob.loc[llenado:llenado+len(resultado)-1,"probability"] = resultado[:,1]
                df_prob.loc[llenado:llenado+len(resultado)-1,"target"] = target.values
                df_prob.loc[llenado:llenado+len(resultado)-1,"i_iteration"] = i
    
                llenado = llenado+len(resultado)
                
            i=i+1    
        
        scores["classifier"]=classifier_name
        scores["group"]=group_labels
        scores["k_fold"]=k_fold

        df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1","true_neg","false_pos","false_neg","true_pos"])
        df_resultados["Accuracy"] = scores['test_acc']
        df_resultados["Precision"] = scores['test_prec_micro']
        df_resultados["Recall"] = scores['test_rec_micro']
        df_resultados["AUC"] = scores['test_auc']
        df_resultados["F1"] = scores['test_f1_score']
        df_resultados["true_neg"] = scores['test_true_neg']
        df_resultados["false_pos"] = scores['test_false_pos']
        df_resultados["false_neg"] = scores['test_false_neg']
        df_resultados["true_pos"] = scores['test_true_pos']
        df_resultados["Random-Seed"] = random_seed
        df_resultados["Feature"] = "Multi-features"
        df_resultados["Grupo"] = group_labels
        df_resultados["Clasificador"] = classifier_name
        df_resultados["k-fold"] = str(k_fold)
        df_resultados["Normalization"] = str(normalization)
       
        return scores,df_resultados,df_prob
    
    # multi
    else:
        scores["classifier"]=classifier_name
        scores["group"]=group_labels
        scores["k_fold"]=k_fold
        df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy"])
        df_resultados["Accuracy"] = scores['test_acc']
        df_resultados["Random-Seed"] = random_seed
        df_resultados["Feature"] = "Multi-features"
        df_resultados["Grupo"] = group_labels
        df_resultados["Clasificador"] = classifier_name
        df_resultados["k-fold"] = str(k_fold)
        df_resultados["Normalization"] = str(normalization)
        
    

       
        return scores,df_resultados,None

def pipeline_personalizado_cross_validation(df,pipeline,classifier_name,group_labels,group_column,features,k_fold, random_seed = None,normalization=True,data_input=False,feature_selection=False,multi=False,n_repeats = 1):
    

    print("Classifier: " + classifier_name)
    print("Groups: " + "-".join(group_labels))
    print("Fold: " + str(k_fold))

    kf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=random_seed)
    
    def confusion_matrix_tn(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm[0, 0]

    def confusion_matrix_fp(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm[0, 1]

    def confusion_matrix_fn(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm[1, 0]
    
    def confusion_matrix_tp(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm[1, 1]
    
    if multi:
        scoring = {'acc': 'accuracy'}
    else:
        scoring = {'acc': 'accuracy',
                    'prec_micro': 'precision_micro',
                    'rec_micro': 'recall_micro',
                    'auc':'roc_auc',
                    'f1_score':'f1_micro',
                    'true_neg':make_scorer(confusion_matrix_tn),
                    'false_pos':make_scorer(confusion_matrix_fp),
                    'false_neg':make_scorer(confusion_matrix_fn),
                    'true_pos':make_scorer(confusion_matrix_tp)
                   }
    

    # scores["estimator"] devuelve tantos Pipelines como n_splits en cross-validation
    scores = cross_validate(pipeline, df[features], df[group_column].values.ravel(), scoring=scoring,
                         cv=kf, return_train_score=False,return_estimator=True,verbose=1,n_jobs=-1,error_score='raise')
    
    scores["classifier"]=classifier_name
    scores["group"]=group_labels
    scores["k_fold"]=k_fold
    if multi:
        df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy"])
        df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),classifier_name,str(k_fold),str(normalization),scores['test_acc']]
    else:
        df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1","true_neg","false_pos","false_neg","true_pos"])
        df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),classifier_name,str(k_fold),str(normalization),\
                                                 scores['test_acc'],scores['test_prec_micro'],scores['test_rec_micro'],scores['test_auc'],scores['test_f1_score'],scores['test_true_neg'],\
                                                 scores['test_false_pos'],scores['test_false_neg'],scores['test_true_pos']]
   
    return scores,df_resultados

def pipeline_cross_validation_hyper_opt(df,group_column,features,k_fold=5,pipe=None,search_space=None,random_seed = None,normalization=True,data_input=False,feature_selection=False,multi=False,n_repeats = 1):
    
    print("Hiperparameter optimization")
    if search_space == None:
        
        transformers = []
        pipeline_list = []
        search_space = []

        if data_input | normalization:
            if data_input:
                transformers.append(('imputer',KNNImputer(n_neighbors=5)))
            if normalization:
                transformers.append(('scaler',MinMaxScaler()))
            pipe_transformer = Pipeline(steps=transformers)
            preprocessor = ColumnTransformer(transformers= [('num',pipe_transformer,features)])
            pipeline_list.append(('preprocessor',preprocessor))
            
        if feature_selection:
            svm_model = svm.SVC(kernel = "linear",max_iter=100000)
            if not multi:
                pipeline_list.append(('feat_sel',SFS(estimator=svm_model,n_features_to_select="auto",tol=0.01,
                                                     direction="forward",n_jobs=-1,scoring='roc_auc')))
            else:
                pipeline_list.append(('feat_sel',SFS(estimator=svm_model,n_features_to_select="auto",tol=0.01,
                                                     direction="forward",n_jobs=-1,scoring='accuracy')))
    
        pipeline_list.append(('model', LogisticRegression(C=0.01)))
        pipe = Pipeline(pipeline_list)
        # print(check_params_exist(pipe, 'imputer'))

        search_space.append({'model': [LogisticRegression(multi_class='ovr',max_iter=100000)],
          'model__C': [0.01, 0.1, 1.0,2.0],
          'model__penalty': ["None", "l2", "l1"]})
        
        search_space.append({'model': [XGBClassifier(n_estimators=5000,learning_rate=0.01)],
          'model__learning_rate': [0.001,0.01,0.1,1,2],
          'model__n_estimators': [100,1000,3000,5000],
          'model__max_depth': [1,3,5,10,20]})
        
        search_space.append({'model': [svm.SVC(max_iter=100000)],
          'model__kernel': ["linear", "poly", "sigmoid"],
          'model__degree': [2,3,4,5],
          'model__gamma': ["scale", "auto"]})
    
    # Specify cross-validation generator, in this case (10 x 5CV)
    cv = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats)

    clf = GridSearchCV(estimator=pipe, param_grid=search_space, scoring="roc_auc", cv=cv,n_jobs=-1)
    
    clf = clf.fit(df[features], df[group_column])
    
    print(clf.best_estimator_)
    
    print(clf.best_params_)
   
    return clf.best_params_["model"]
    
def menu_clasificador(clasificador, dict_df,columna_features,columnas_grupo,k_folds,path,data_input=False,feature_selection=False,multi=False, random_seed = None,n_repeats=1):
    
    df_clasificador_multi = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1","true_neg","false_pos","false_neg","true_pos"])


    scores_dict = {}
    prob_dict = {}
    model = None
    
    if clasificador == "regresion_logistica":
        model = LogisticRegression(max_iter=100000,)
    elif clasificador == "svm":
        model = svm.SVC(kernel = "linear",max_iter=100000,probability=True)
    elif clasificador == "xgboost":
        model = XGBClassifier(eval_metric = "logloss")
        
    for key, value in dict_df.items():
        scores_dict[key] = []
        prob_dict[key] = {}

        for folds_counter,k_fold in enumerate(k_folds):
            (scores_clasi, df_clasif,df_prob) = pipeline_cross_validation(value,model, clasificador,\
                                      key, columnas_grupo, list(columna_features.values())[0], k_fold,random_seed = random_seed,\
                                      normalization=True, data_input = data_input,feature_selection=feature_selection,multi=multi,n_repeats=n_repeats)
            df_clasificador_multi = pd.concat([df_clasificador_multi, df_clasif])
            scores_dict[key].append(scores_clasi)
            prob_dict[key][k_fold] = df_prob

            if not multi:
                df_prob.to_excel(path+"/probs_"+key+"_"+str(k_fold)+"_"+clasificador+"_"+\
                                                          list(columna_features.keys())[0]+"_"+str(n_repeats)+".xlsx") 
                

    return (scores_dict,df_clasificador_multi,prob_dict)

def clasificador_personalizado(ml_classifier,ml_classifier_name, dict_df,columna_features,columnas_grupo,tipo_columnas,k_folds,path,data_input=False,feature_selection=0,multi=False, random_seed = None,n_repeats=1):
    
    print("Base: " + tipo_columnas)
    print("------------------------")
    
    df_clasificador_multi = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1","true_neg","false_pos","false_neg","true_pos"])

    scores_dict = {}
    prob_dict = {}

    model = ml_classifier
        
    for key, value in dict_df.items():
        scores_dict[key] = []
        prob_dict[key] = {}

        for folds_counter,k_fold in enumerate(k_folds):
            (scores_clasi, df_clasif,df_prob) = pipeline_cross_validation(value,model, ml_classifier_name,\
                                      key, columnas_grupo, columna_features, k_fold,random_seed = random_seed,\
                                      normalization=True, data_input = data_input,feature_selection=feature_selection,multi=multi,n_repeats=n_repeats)
            df_clasificador_multi = pd.concat([df_clasificador_multi, df_clasif])

            scores_dict[key].append(scores_clasi)
            prob_dict[key][k_fold] = df_prob

            df_prob.to_excel(path+"/resultados_machine_learning/probs_"+key+"_"+str(k_fold)+"_"+ml_classifier_name+"_"+\
                                                      tipo_columnas+".xlsx")    
    df_clasificador_multi.to_excel(path+"/resultados_machine_learning/resultados_"+ml_classifier_name+"_"+\
                                              tipo_columnas+".xlsx")    
    

    return (scores_dict,df_clasificador_multi,prob_dict)

def pipeline_personalizado(pipeline,ml_classifier_name, df,df_labels,columna_features,columnas_grupo,tipo_columnas,k_folds,path,data_input=False,feature_selection=0,multi=False, random_seed = None,n_repeats=1):
    
    print("Base: " + tipo_columnas)
    print("------------------------")
    
    df_clasificador_multi = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1","true_neg","false_pos","false_neg","true_pos"])

    scores_list = []
        
    for i_label, df_combinado in enumerate(df):
        scores_list.append([])
        for folds_counter,k_fold in enumerate(k_folds):
            (scores_clasi, df_clasif) = pipeline_personalizado_cross_validation(df_combinado,pipeline, ml_classifier_name,\
                                      df_labels[i_label], columnas_grupo, columna_features, k_fold,random_seed = random_seed,\
                                      normalization=True, data_input = data_input,feature_selection=feature_selection,multi=multi,n_repeats=n_repeats)
            df_clasificador_multi = pd.concat([df_clasificador_multi, df_clasif])

            scores_list[i_label].append(scores_clasi)
    
    df_clasificador_multi.to_excel(path+"/resultados_machine_learning/resultados_"+ml_classifier_name+"_"+\
                                              tipo_columnas+".xlsx")           

    return (scores_list,df_clasificador_multi)

def tres_clasificadores(clasificadores,dict_df,columnas_features,columnas_grupo,tipo_columnas,k_folds,path,data_input=False,feature_selection=False,multi=False,random_seed=None,n_repeats=1):
    
    dict_scores_list = {}
    dict_resultados = {}
    dict_prob = {}

    print("Base: " + tipo_columnas)
    print("------------------------")

    for clasificador in clasificadores:
        (scores,df_clasificador_multi,df_prob) = menu_clasificador(clasificador,dict_df,columnas_features,columnas_grupo,k_folds,path,\
                                                                   data_input = data_input,feature_selection=feature_selection,multi=multi,random_seed=random_seed,n_repeats=n_repeats)
    
        df_clasificador_multi.to_excel(path+"/resultados_"+clasificador+"_"+\
                                       list(columnas_features.keys())[0]+"_" + tipo_columnas +".xlsx") 
        dict_resultados[clasificador] = df_clasificador_multi
        dict_scores_list[clasificador] = scores
        dict_prob[clasificador] = df_prob

    return (dict_scores_list,dict_resultados,dict_prob)

# Ploteo feature importance entregado por el clasificador. Promedio la importanced de todos y después me quedo con los primeros n_feature_importance
def feature_importance_not_feat_selection(dict_df_scores,n_feature_importance,k_folds,n_repeats,cwd,show_figures=False, replacements=None,colors=None):
        
    for key_features, value_features in dict_df_scores.items():
        if "feat" not in key_features:
            for key_algorithm, value_algorithm in value_features.items():
                for key_group,value_group in value_algorithm.items():
                    for i_fold,fold in enumerate(value_group):
                        # creating the dataset
                        importances_matrix = []

                        for i_pipe,pipe in enumerate(fold["estimator"]):
                            if hasattr(pipe["model"], "coef_"):
                                importances_matrix.append([abs(ele) for ele in pipe["model"].coef_])
                            elif hasattr(pipe["model"], "feature_importances_"):
                                importances_matrix.append(pipe["model"].feature_importances_)
                            else:
                                importances_matrix.append(np.zeros(len(pipe.feature_names_in_)))
                                print(key_algorithm+"_"+key_features+"_"+str(k_folds[i_fold])+"_folds_"+key_group+"_"+str(n_repeats) + " No tiene feature importance")

                        y_values = np.mean(np.vstack(importances_matrix),axis=0)
                        y_std = np.std(np.vstack(importances_matrix),axis=0)[0:n_feature_importance]
                        x_values = fold["estimator"][0]["preprocessor"]._columns[0]

                        if replacements!=None:
                            for i in range(len(x_values)):
                                if x_values[i] in replacements:
                                    x_values[i] = replacements[x_values[i]]
                            
                        data = {'x_values':x_values,'y_values':y_values}
                        
                        df_feature_importance = pd.DataFrame(data).sort_values('y_values', ascending=False)
                        df_feature_importance.to_excel(cwd+"_"+key_algorithm+"/"+key_features+"_"+str(k_folds[i_fold])+"_folds_"+key_group+"_"+str(n_repeats)+".xlsx")
                        

                        df_feature_importance = df_feature_importance.head(n_feature_importance) # Me quedo con las n_feature_importance filas para el gráfico
                        
                        # formato para los labels
                        font_axis_labels = {'family': 'arial',
                            'color':  'black',
                            'weight': 'bold',
                            'size': 32,
                            }
                        
                        plt.figure(figsize = (10, 6))
                        ax1 = plt.gca()
                        
                        color = "blue"
                        
                        if colors!=None:
                            for color_key, color_value in colors.items():
                                if key_group == color_key:
                                    color = color_value
                            
                        data = {'x_values':x_values,'y_values':y_values}
                        
                        plt.barh(df_feature_importance["x_values"],df_feature_importance["y_values"], color=color, xerr=y_std, align='center', ecolor='black', capsize=10)
                        plt.xlabel("Coefficient score",fontsize=32, fontdict=font_axis_labels)
                        plt.ylabel("Feature",fontsize=32, fontdict=font_axis_labels)
                        plt.xticks(fontsize=20,fontname = "arial")
                        plt.yticks(fontsize=20,fontname = "arial")
                        #ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        # Pongo en negrita los ticks
                        for element in ax1.get_xticklabels():
                            element.set_fontweight("bold")
                        for element in ax1.get_yticklabels():
                            element.set_fontweight("bold")
                        plt.title("")

                        plt.savefig(cwd+"_"+key_algorithm+"/"+key_features+"_"+str(k_folds[i_fold])+"_folds_"+key_group+"_"+str(n_repeats)+".png",\
                                    bbox_inches='tight')
                        plt.show()
                        if show_figures:
                            plt.show()
                        else:
                            plt.close('all')

def feature_importance_feat_selection(dict_df_scores,algorithm_label,n_repeats,cwd,show_figures=False):
    # ploteo la feature importance por cantidad de veces que fue elegida la feature.

    for key_features, value_features in dict_df_scores.items():
       if "feat_sel" in key_features:
            for key_algorithm, value_algorithm in value_features.items():
                for key_group,value_group in value_algorithm.items():
                    for i_fold,fold in enumerate(value_group):
                        # creating the dataset
                        num_features_dict = dict()
            
                        for i_pipe,pipe in enumerate(fold["estimator"]):
                            features = [f for f,s in zip(pipe.feature_names_in_, pipe["feat_sel"].support_) if s]
                            for feature in features:    
                                if feature in num_features_dict:
                                    num_features_dict[feature] = num_features_dict[feature]+1
                                else:
                                    num_features_dict[feature] = 1
            
                        courses = list(num_features_dict.keys())
                        values = list(num_features_dict.values())
                        
                        plt.figure(figsize = (10, 5))
            
                        # creating the bar plot
                        plt.bar(courses, values, color ='blue',
                                width = 0.4)
                         
                        plt.xlabel("Base")
                        plt.ylabel("N° times feature selected")
                        plt.xticks(rotation=90)
                        plt.title(key_features + " with " + key_algorithm + " " + key_group)
                        
                        plt.savefig(cwd+"_"+key_algorithm+"/"+key_features+"_"+key_group+"_"+str(n_repeats)+".png",\
                                    bbox_inches='tight')
                        if show_figures:
                            plt.show()
                        else:
                            plt.close('all')
           

def logistic_regression_cross_validation(df,group_labels,group_column,features,k_fold,random_seed = None,normalization=True,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance="",data_input=False,feature_selection=0):
    
    logisticRegr = LogisticRegression(max_iter=100000,)

    df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"])
    
    if not data_input:
        df_actual = df.dropna(subset = features)
    else:
        df_actual = df.copy(deep=True)
      
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
    accuracy = []
    precision = []
    recall = []
    auc = []
    f1 = []
    cm_total = np.full((len(group_labels), len(group_labels)),0)
    
    importances_matrix = []

    for train_index, test_index in kf.split(df_actual, df_actual[group_column]):
        # df_actual_train = df_actual.iloc[train_index]
        # df_actual_train_control = df_actual_train[df_actual_train[group_column] == 0]
        # df_actual_train_no_control = df_actual_train[df_actual_train[group_column] == 1]
        
        # imputer_control = KNNImputer()
        # # fit on the dataset
        # imputer_control.fit(df_actual_train_control)
        # # transform the dataset
        # df_actual_train_control = imputer_control.transform(df_actual_train_control)
        
        # imputer_no_control = KNNImputer()
        # # fit on the dataset
        # imputer_no_control.fit(df_actual_train_no_control)
        # # transform the dataset
        # df_actual_train_no_control = imputer_no_control.transform(df_actual_train_no_control)
        
        # X_test = imputer.transform(X_test)
        
        X_train = df_actual.iloc[train_index].loc[:, features]
        X_test = df_actual.iloc[test_index].loc[:,features]
        y_train = df_actual.iloc[train_index].loc[:,group_column]
        y_test = df_actual.iloc[test_index].loc[:,group_column]
        
        # Hago inputación con knn
        if data_input:
            imputer = KNNImputer()
            # fit on the dataset
            imputer.fit(X_train)
            # transform the dataset
            X_train = imputer.transform(X_train)
            X_test = imputer.transform(X_test)

        
        # Normalizo
        if normalization:
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
        if feature_selection>0:
            select_k_best = SelectKBest(f_classif, k=5)
            X_train = select_k_best.fit_transform(X_train, y_train)
            X_test = select_k_best.transform(X_test)
            # Para obtener la feature importance
            # cols = select_k_best.get_support(indices=True)
            # df.iloc[:,cols].columns
        #Train the model
        logisticRegr.fit(X_train, y_train) #Training the model
        predictions = logisticRegr.predict(X_test)
        if (len(group_labels)>2):
            #Train the model
            accuracy.append(metrics.accuracy_score(y_test, predictions))
            precision.append(metrics.precision_score(y_test, predictions,average="micro"))
            recall.append(metrics.recall_score(y_test, predictions,average="micro"))
            # macro_roc_auc_ovo = roc_auc_score(np.array(y_test)[np.newaxis].T, np.array(predictions)[np.newaxis].T, multi_class="ovo", average="macro")
            # weighted_roc_auc_ovo = roc_auc_score(
            #     y_test, predictions, multi_class="ovo", average="weighted"
            # )
            # auc.append(macro_roc_auc_ovo)
            f1.append(metrics.f1_score(y_test, predictions,average="micro"))
        else:
            accuracy.append(metrics.accuracy_score(y_test, predictions))
            precision.append(metrics.precision_score(y_test, predictions))
            recall.append(metrics.recall_score(y_test, predictions))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
            auc.append(metrics.auc(fpr, tpr))
            f1.append(metrics.f1_score(y_test, predictions))
        
        cm_actual = metrics.confusion_matrix(y_test.values, predictions)
        for m in range(0,len(group_labels)):
            for n in range (0,len(group_labels)):
                cm_total[m][n] += cm_actual[m][n]
    
        importances_matrix.append(logisticRegr.coef_[0])

    df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),"Regresión logística",str(k_fold),str(normalization),np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(auc),np.mean(f1)]
    
    # if n_importances>0:
    #     df_importances = pd.DataFrame(data={
    #         'attribute': features,
    #         'importance': 0
    #     })
        
    #     df_importances["importance"] = np.mean(np.vstack(importances_matrix),axis=0)
    #     df_importances["importance_abs"] = np.abs(df_importances["importance"])
    #     df_importances = df_importances.sort_values(by='importance_abs', ascending=False)
    
    #     n = 20
    #     df_importances = df_importances.nlargest(n, 'importance_abs')
    
    #     df_importances = df_importances.sort_values(by='importance', ascending=False)
    
    #     fig=plt.figure();
    #     plt.bar(x=df_importances['attribute'], height=df_importances['importance'], color='#087E8B')
    #     plt.title('Feature importances obtained from coefficients')
    #     plt.xticks(rotation='vertical')
    #     plt.savefig(path_feature_importance+".png",bbox_inches='tight')
    #     plt.close(fig)
    
    
    if path_confusion_matrix!= "":
        fig=plt.figure();
        sns.heatmap(cm_total, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', yticklabels=group_labels, xticklabels=group_labels);
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = "Multi-features. k-folds: "+str(k_fold)
        plt.title(all_sample_title);
        plt.savefig(path_confusion_matrix+'.png')
        plt.close(fig)
    
    if path_excel!= "":
        df_resultados.to_excel(path_excel+".xlsx")
    
    return df_resultados

def svm_cross_validation(df,group_labels,group_column,features,k_fold,random_seed = None,normalization=False,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance="",data_input=False,feature_selection=0):
    
    svm_model = svm.SVC(kernel = "linear",max_iter=100000)

    df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"])
        
    if not data_input:
        df_actual = df.dropna(subset = features)
    else:
        df_actual = df.copy(deep=True)
    
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
    accuracy = []
    precision = []
    recall = []
    auc = []
    f1 = []
    cm_total = np.full((len(group_labels), len(group_labels)),0)
    
    importances_matrix = []

    for train_index, test_index in kf.split(df_actual, df_actual[group_column]):
        X_train = df_actual.iloc[train_index].loc[:, features]
        X_test = df_actual.iloc[test_index].loc[:,features]
        y_train = df_actual.iloc[train_index].loc[:,group_column]
        y_test = df_actual.iloc[test_index].loc[:,group_column]
    
        # Hago inputación con knn
        if data_input:
            imputer = KNNImputer()
            # fit on the dataset
            imputer.fit(X_train)
            # transform the dataset
            X_train = imputer.transform(X_train)
            X_test = imputer.transform(X_test)
            
        # Normalizo
        if normalization:
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
        if feature_selection>0:
            select_k_best = SelectKBest(f_classif, k=5)
            X_train = select_k_best.fit_transform(X_train, y_train)
            X_test = select_k_best.transform(X_test)
            
        #Train the model
        svm_model.fit(X_train, y_train) #Training the model
        predictions = svm_model.predict(X_test)
        if (len(group_labels)>2):
            #Train the model
            accuracy.append(metrics.accuracy_score(y_test, predictions))
            precision.append(metrics.precision_score(y_test, predictions,average="micro"))
            recall.append(metrics.recall_score(y_test, predictions,average="micro"))
            # macro_roc_auc_ovo = roc_auc_score(np.array(y_test)[np.newaxis].T, np.array(predictions)[np.newaxis].T, multi_class="ovo", average="macro")
            # weighted_roc_auc_ovo = roc_auc_score(
            #     y_test, predictions, multi_class="ovo", average="weighted"
            # )
            # auc.append(macro_roc_auc_ovo)
            f1.append(metrics.f1_score(y_test, predictions,average="micro"))
        else:
            accuracy.append(metrics.accuracy_score(y_test, predictions))
            precision.append(metrics.precision_score(y_test, predictions))
            recall.append(metrics.recall_score(y_test, predictions))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
            auc.append(metrics.auc(fpr, tpr))
            f1.append(metrics.f1_score(y_test, predictions))
        
        cm_actual = metrics.confusion_matrix(y_test.values, predictions)
        for m in range(0,len(group_labels)):
            for n in range (0,len(group_labels)):
                cm_total[m][n] += cm_actual[m][n]
        
        importances_matrix.append(svm_model.coef_[0])

    df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),"SVM",str(k_fold),str(normalization),np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(auc),np.mean(f1)]
    
    # if n_importances>0:
    #     df_importances = pd.DataFrame(data={
    #         'attribute': features,
    #         'importance': 0
    #     })
        
    #     df_importances["importance"] = np.mean(np.vstack(importances_matrix),axis=0)
    #     df_importances["importance_abs"] = np.abs(df_importances["importance"])
    #     df_importances = df_importances.sort_values(by='importance_abs', ascending=False)
    
    #     n = 20
    #     df_importances = df_importances.nlargest(n, 'importance_abs')
    
    #     df_importances = df_importances.sort_values(by='importance', ascending=False)
    
    #     fig=plt.figure();
    #     plt.bar(x=df_importances['attribute'], height=df_importances['importance'], color='#087E8B')
    #     plt.title('Feature importances obtained from coefficients')
    #     plt.xticks(rotation='vertical')
    #     plt.savefig(path_feature_importance+".png",bbox_inches='tight')
    #     plt.close(fig)
    
    
    if path_confusion_matrix!= "":
        fig=plt.figure();
        sns.heatmap(cm_total, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', yticklabels=group_labels, xticklabels=group_labels);
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = "Multi-features. k-folds: "+str(k_fold)
        plt.title(all_sample_title);
        plt.savefig(path_confusion_matrix+'.png')
        plt.close(fig)
    
    if path_excel!= "":
        df_resultados.to_excel(path_excel+".xlsx")
    
    return df_resultados


def xgboost_cross_validation(df,group_labels,group_column,features,k_fold,random_seed = None,normalization=False,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance="",data_input=False,feature_selection=0):
    
    xg_model = XGBClassifier(max_iter=100000)

    df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"])
        
    if not data_input:
        df_actual = df.dropna(subset = features)
    else:
        df_actual = df.copy(deep=True)

    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
    accuracy = []
    precision = []
    recall = []
    auc = []
    f1 = []
    cm_total = np.full((len(group_labels), len(group_labels)),0)
    
    importances_matrix = []

    for train_index, test_index in kf.split(df_actual, df_actual[group_column]):
        X_train = df_actual.iloc[train_index].loc[:, features]
        X_test = df_actual.iloc[test_index].loc[:,features]
        y_train = df_actual.iloc[train_index].loc[:,group_column]
        y_test = df_actual.iloc[test_index].loc[:,group_column]
    
        # Hago inputación con knn
        if data_input:
            imputer = KNNImputer()
            # fit on the dataset
            imputer.fit(X_train)
            # transform the dataset
            X_train = imputer.transform(X_train)
            X_test = imputer.transform(X_test)
            
        # Normalizo
        if normalization:
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
        if feature_selection>0:
            select_k_best = SelectKBest(f_classif, k=5)
            X_train = select_k_best.fit_transform(X_train, y_train)
            X_test = select_k_best.transform(X_test)
            
        #Train the model
        xg_model.fit(X_train, y_train) #Training the model
        predictions = xg_model.predict(X_test)
        if (len(group_labels)>2):
            #Train the model
            accuracy.append(metrics.accuracy_score(y_test, predictions))
            precision.append(metrics.precision_score(y_test, predictions,average="micro"))
            recall.append(metrics.recall_score(y_test, predictions,average="micro"))
            # macro_roc_auc_ovo = roc_auc_score(np.array(y_test)[np.newaxis].T, np.array(predictions)[np.newaxis].T, multi_class="ovo", average="macro")
            # weighted_roc_auc_ovo = roc_auc_score(
            #     y_test, predictions, multi_class="ovo", average="weighted"
            # )
            # auc.append(macro_roc_auc_ovo)
            f1.append(metrics.f1_score(y_test, predictions,average="micro"))
        else:
            accuracy.append(metrics.accuracy_score(y_test, predictions))
            precision.append(metrics.precision_score(y_test, predictions))
            recall.append(metrics.recall_score(y_test, predictions))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
            auc.append(metrics.auc(fpr, tpr))
            f1.append(metrics.f1_score(y_test, predictions))
        
        cm_actual = metrics.confusion_matrix(y_test.values, predictions)
        for m in range(0,len(group_labels)):
            for n in range (0,len(group_labels)):
                cm_total[m][n] += cm_actual[m][n]
    
        importances_matrix.append(xg_model.feature_importances_)

    df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),"xgboost",str(k_fold),str(normalization),np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(auc),np.mean(f1)]
    
    # if n_importances>0:
    #     df_importances = pd.DataFrame(data={
    #         'attribute': features,
    #         'importance': 0
    #     })
        
    #     df_importances["importance"] = np.mean(np.vstack(importances_matrix),axis=0)
    #     df_importances["importance_abs"] = np.abs(df_importances["importance"])
    #     df_importances = df_importances.sort_values(by='importance_abs', ascending=False)
    
    #     n = 20
    #     df_importances = df_importances.nlargest(n, 'importance_abs')
    
    #     df_importances = df_importances.sort_values(by='importance', ascending=False)
    
    #     fig=plt.figure();
    #     plt.bar(x=df_importances['attribute'], height=df_importances['importance'], color='#087E8B')
    #     plt.title('Feature importances obtained from coefficients')
    #     plt.xticks(rotation='vertical')
    #     plt.savefig(path_feature_importance+".png",bbox_inches='tight')
    #     plt.close(fig)
    
    
    if path_confusion_matrix!= "":
        fig=plt.figure();
        sns.heatmap(cm_total, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', yticklabels=group_labels, xticklabels=group_labels);
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = "Multi-features. k-folds: "+str(k_fold)
        plt.title(all_sample_title);
        plt.savefig(path_confusion_matrix+'.png')
        plt.close(fig)
    
    if path_excel!= "":
        df_resultados.to_excel(path_excel+".xlsx")
    
    return df_resultados


def clasificacion_hyp_opt():




    df_data = pd.read_csv("D:/Franco/Doctorado/Laboratorio/17_aphasia_verbal_fluency_ucsf/mayor_a_2/nueva/psicolinguisticas/psico_4_grupos.csv")
    # selected_columns = df_data.filter(regex="promedio|osv|PIDN|Zac's DX")

    df_data = df_data.loc[:, ~df_data.columns.str.contains("words_lemma_Nsyll")]
    # df_data = df_data.drop(["Unnamed: 0"],axis=1)

    df_nc_lvppa = df_data[(df_data["Zac's DX"] == "NC") | (df_data["Zac's DX"] == "lvPPA")].reset_index(drop=True)
    df_nc_nfvppa = df_data[(df_data["Zac's DX"] == "NC") | (df_data["Zac's DX"] == "nfvPPA")].reset_index(drop=True)
    df_nc_svppa = df_data[(df_data["Zac's DX"] == "NC") | (df_data["Zac's DX"] == "svPPA")].reset_index(drop=True)
    df_lvppa_nfvppa = df_data[(df_data["Zac's DX"] == "lvPPA") | (df_data["Zac's DX"] == "nfvPPA")].reset_index(drop=True)
    df_lvppa_svppa = df_data[(df_data["Zac's DX"] == "lvPPA") | (df_data["Zac's DX"] == "svPPA")].reset_index(drop=True)
    df_nfvppa_svppa = df_data[(df_data["Zac's DX"] == "nfvPPA") | (df_data["Zac's DX"] == "svPPA")].reset_index(drop=True)

    # df_nc_lvppa["Zac's DX"] = df_nc_lvppa["Zac's DX"].sample(frac=1).reset_index(drop=True)
    # df_nc_nfvppa["Zac's DX"] = df_nc_nfvppa["Zac's DX"].sample(frac=1).reset_index(drop=True)
    # df_nc_svppa["Zac's DX"] = df_nc_svppa["Zac's DX"].sample(frac=1).reset_index(drop=True)
    # df_lvppa_nfvppa["Zac's DX"] = df_lvppa_nfvppa["Zac's DX"].sample(frac=1).reset_index(drop=True)
    # df_lvppa_svppa["Zac's DX"] = df_lvppa_svppa["Zac's DX"].sample(frac=1).reset_index(drop=True)
    # df_nfvppa_svppa["Zac's DX"] = df_nfvppa_svppa["Zac's DX"].sample(frac=1).reset_index(drop=True)

    group_column = "Zac's DX"
    id_column = "PIDN"
    dict_dfs= {
        "nc_lvppa_sin_nsyll":df_nc_lvppa,
               "nc_nfvppa_sin_nsyll":df_nc_nfvppa,
                "nc_svppa_sin_nsyll":df_nc_svppa,
                "lvppa_nfvppa_sin_nsyll":df_lvppa_nfvppa,
                "lvppa_svppa_sin_nsyll":df_lvppa_svppa,
                "nfvppa_svppa_sin_nsyll":df_nfvppa_svppa}


    results = []
    models = [
        ('Logistic Regression', LogisticRegression(max_iter = 100000), {'C': [0.00001,0.0001, 0.001,0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}),
        ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 200]}),
        ('SVM', SVC(max_iter = 100000), {'kernel': ['linear'], 'C': [0.00001,0.0001, 0.001,0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}),
        # ('XGBoost', XGBClassifier(), {'max_depth': [3, 5, 7], 'learning_rate': [0.00001,0.0001, 0.001,0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'n_estimators': [10, 100, 1000, 3000, 5000]})
        
        # ('Logistic Regression', LogisticRegression(), {'C': [0.00001,0.0001, ]}),
        # ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100]}),
        # ('SVM', SVC(), {'kernel': ['linear'], 'C': [0.00001,0.0001]})
        # ('XGBoost', XGBClassifier(), {'max_depth': [3, 5], 'learning_rate': [0.00001,0.0001]})
    ]

    # %% Grafico de barras

    def calcular_percentil_25(columna):
        return np.percentile(columna, 25)

    def calcular_percentil_75(columna):
        return np.percentile(columna, 75)


    def grafico_barras_horizontales(df, x_label='Columnas', y_label='Promedio', title='Promedio y Desvío Estándar de Columnas', path=None):
        
        # Identificar las columnas numéricas
        df_columnas_numericas = df.select_dtypes(include=['number'])
        
        # Calcular promedio y desvío estándar de cada columna
        promedios = df_columnas_numericas.abs().mean()
        desvios = df_columnas_numericas.abs().std()
        
        # Aplica la función a cada columna
        q1 = df_columnas_numericas.apply(calcular_percentil_25)    #q3 = np.percentile(datos, 75)  # Tercer cuartil (Q3)
        q3 = df_columnas_numericas.apply(calcular_percentil_75)    #q3 = np.percentile(datos, 75)  # Tercer cuartil (Q3)

        iqr = q3 - q1
        
        # Ordenar por promedio de mayor a menor
        promedios_sorted = promedios.sort_values(ascending=False)
        
        # Crear el gráfico de barras horizontales
        fig, ax = plt.subplots()
        
        # Posiciones de las barras
        y_pos = np.arange(len(promedios_sorted))
        
        # Dibujar las barras horizontales
        bars = ax.barh(y_pos, promedios_sorted, xerr=desvios[promedios_sorted.index], align='center', alpha=0.7)
        
        # Etiquetas de las columnas en el eje Y
        ax.set_yticks(y_pos)
        ax.set_yticklabels(promedios_sorted.index)
        
        # Etiquetas y título
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # Crear un DataFrame con los datos de variable, promedio y std
        df_result = pd.DataFrame({
            'variable': promedios_sorted.index,
            'promedio': promedios_sorted.values,
            'std': desvios[promedios_sorted.index].values,
            'iqr': iqr[promedios_sorted.index]
        })
        
        if path is not None:
            df_result.to_excel(path + ".xlsx",index=False)
            plt.savefig(path + '.png', bbox_inches='tight')
            
        # Mostrar el gráfico
        plt.show()
        
        
    def grafico_barras_horizontales_count(df, x_label='Columnas', y_label='Promedio', title='Promedio y Desvío Estándar de Columnas', path=None):
        
        # Identificar las columnas numéricas
        df_columnas_numericas = df.select_dtypes(include=['number'])
        
        # Calcular promedio y desvío estándar de cada columna
        promedios = df_columnas_numericas.abs().sum()
        # desvios = df_columnas_numericas.abs().std()
        
        # Aplica la función a cada columna
        q1 = df_columnas_numericas.apply(calcular_percentil_25)    #q3 = np.percentile(datos, 75)  # Tercer cuartil (Q3)
        q3 = df_columnas_numericas.apply(calcular_percentil_75)    #q3 = np.percentile(datos, 75)  # Tercer cuartil (Q3)

        iqr = q3 - q1
        
        # Ordenar por promedio de mayor a menor
        promedios_sorted = promedios.sort_values(ascending=False)
        
        # Crear el gráfico de barras horizontales
        fig, ax = plt.subplots()
        
        # Posiciones de las barras
        y_pos = np.arange(len(promedios_sorted))
        
        # Dibujar las barras horizontales
        bars = ax.barh(y_pos, promedios_sorted, align='center', alpha=0.7)
        
        # Etiquetas de las columnas en el eje Y
        ax.set_yticks(y_pos)
        ax.set_yticklabels(promedios_sorted.index)
        
        # Etiquetas y título
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # Crear un DataFrame con los datos de variable, promedio y std
        df_result = pd.DataFrame({
            'variable': promedios_sorted.index,
            'promedio': promedios_sorted.values,
            # 'std': desvios[promedios_sorted.index].values,
            'iqr': iqr[promedios_sorted.index]
        })
        
        if path is not None:
            df_result.to_excel(path + "_count.xlsx",index=False)
            plt.savefig(path + '_count.png', bbox_inches='tight')
            
        # Mostrar el gráfico
        plt.show()
        
    def abs_values_in_dict(input_dict):
        return {key: abs(value) for key, value in input_dict.items()}
    # %%

    label_mapping = [{'NC': 0, 'lvPPA': 1},{'NC': 0, 'nfvPPA': 1},{'NC': 0, 'svPPA': 1}
                      ,{'lvPPA': 0, 'nfvPPA': 1},{'lvPPA': 0, 'svPPA': 1},{'nfvPPA': 0, 'svPPA': 1}]

    # label_mapping = [{'NC': 0, 'nfvPPA': 1},{'NC': 0, 'svPPA': 1},{'nfvPPA': 0, 'svPPA': 1}]
    # %%
    i_model = 0
    for key, value in dict_dfs.items():
        # Divide los datos en características (variables independientes) y la variable objetivo
        X = value.drop([group_column,id_column], axis=1)  # Características
        y = value[group_column]  # Variable objetivo
        # Crea una instancia del codificador de etiquetas
        # label_encoder = LabelEncoder()
        
        # Convierte las etiquetas categóricas a valores numéricos
        # y = label_encoder.fit_transform(y)
        
        y = y.map(label_mapping[i_model])
        i_model+=1
        results = []
        
        # Aplica la normalización Min-Max a los datos de entrenamiento
        scaler = MinMaxScaler()
        imputer = KNNImputer(n_neighbors=3)

        n = 10
        list_all_results=[]
        decision_scores = []
        print(key)
        
        df_feature_importance = pd.DataFrame(columns=["modelo"] + list(X.columns))
        df_feature_importance_count = pd.DataFrame(columns=["modelo"] + list(X.columns))


        for i_loop in range (0,n):
            print(i_loop)
            outer_cv = StratifiedKFold(n_splits=10, shuffle=True)
            for name, model, params in models:
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
                grid_search = GridSearchCV(model, params, cv=inner_cv,n_jobs=-1)
            
                best_params_list = []
                best_model_list = []
                accuracy_list = []
                auc_list = []
                f1_list = []
                recall_list = []
                precision_list = []
                sens_list = []
                spec_list = []
                uar_list = []
                
                i_outer_loop = 0
                for train_index, test_index in outer_cv.split(X, y):
                    i_outer_loop+=1
                    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_val = y[train_index], y[test_index]
            
                    X_train = scaler.fit_transform(X_train)
                
                    # Aplica la misma transformación a los datos de prueba
                    X_val = scaler.transform(X_val)
                    
                    # Ajustar y transformar los datos de entrenamiento
                    X_train = imputer.fit_transform(X_train)
                    
                    # Transformar los datos de prueba utilizando el imputador ya ajustado
                    X_val = imputer.transform(X_val)

                    grid_search.fit(X_train, y_train)
            
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
            
                    best_model_list.append(best_model)
                    best_params_list.append(best_params)
            
                    best_model.fit(X_train, y_train)
                    y_pred = best_model.predict(X_val)
                    
                    if (name == "Random Forest") or (name == "XGBoost"):
                        decision_scores_actual = best_model.predict_proba(X_val)[:,1]
                    else:
                        decision_scores_actual = best_model.decision_function(X_val)
                        
                    # Feature importance
                    if (name == "Logistic Regression"):
                        feature_importance = best_model.coef_[0]
                        feature_importance_named = dict(zip(X.columns, feature_importance))
                    elif (name == "Random Forest"):
                        feature_importance = best_model.feature_importances_
                        feature_importance_named = dict(zip(X.columns, feature_importance))
                    elif (name == "SVM"):
                        # Obtener los vectores de soporte
                        support_vectors = best_model.support_vectors_
                        # Puedes analizar los coeficientes para estimar la importancia relativa
                        coeficients = best_model.coef_[0]
                        feature_importance_named = dict(zip(X.columns, coeficients))
                    elif (name == "XGBoost"):
                        feature_importance = best_model.feature_importances_
                        feature_importance_named = dict(zip(X.columns, feature_importance))
                        
                        
                    feature_importance_named = abs_values_in_dict(feature_importance_named)

                    
                    # Find the minimum and maximum values in the list
                    min_val = min(feature_importance_named.values())
                    max_val = max(feature_importance_named.values())
                    
                    # Apply min-max normalization
                    normalized_data = [(x - min_val) / (max_val - min_val) for x in feature_importance_named.values()]
                    df_feature_importance.loc[len(df_feature_importance)] = [name]+list(normalized_data)

                    # Obtener una lista de los valores ordenados de mayor a menor
                    diccionario_ordenado = sorted(feature_importance_named.values(), reverse=False)

                    # Crear un nuevo diccionario con los valores reemplazados por su posición ordenada
                    diccionario_ordenado = {k: diccionario_ordenado.index(v) + 1 for k, v in feature_importance_named.items()}
            
                    lista_a_agregar = [name]
                    for column in X.columns:
                        lista_a_agregar.append(diccionario_ordenado[column])
                        
                    df_feature_importance_count.loc[len(df_feature_importance_count)] = lista_a_agregar
                    
                    accuracy = accuracy_score(y_val, y_pred)
                    auc = roc_auc_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred)
                    recall = recall_score(y_val, y_pred)
                    precision = precision_score(y_val, y_pred)
                    
                    # Calcula sensibilidad, especificidad y UAR
                    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                    sens = tp / (tp + fn)
                    spec = tn / (tn + fp)
                    uar = (sens + spec) / 2
        
                    accuracy_list.append(accuracy)
                    auc_list.append(auc)
                    f1_list.append(f1)
                    recall_list.append(recall)
                    precision_list.append(precision)
                    sens_list.append(sens)
                    spec_list.append(spec)
                    uar_list.append(uar)
                    # kappa = cohen_kappa_score(y_val, y_pred)
                    # oob = best_model.oob_score_
                    
                    inner_result = {
                        'Iteracion': i_loop,
                        'Modelo': name,
                        'Mejores hiperparámetros': best_params,
                        'Accuracy': accuracy,
                        'AUC': auc,
                        'F1': f1,
                        'Recall': recall,
                        'Precision': precision,
                        'Sensibilidad': sens,
                        'Especificidad': spec,
                        'UAR': uar
                        # 'kappa': kappa,
                        # 'oob': oob
                    }
                    list_all_results.append(inner_result)
                    
                    # Guardar los decision scores
                    for i in range(len(decision_scores_actual)):
                        decision_scores.append({
                            'Iteracion': i_loop,
                            'i_outer_loop': i_outer_loop,
                            'Modelo': name,
                            'Mejores hiperparámetros': best_params,
                            'Decision Score': decision_scores_actual[i],
                            'target': y_val[test_index[i]],
                            'PatientId': value.iloc[test_index[i]][id_column]  # Assuming 'value' is your DataFrame

                        })
        
        df_results_inner = pd.DataFrame(list_all_results)
        
        # Guarda el DataFrame en un archivo Excel
        df_results_inner.to_excel('C:/Franco/Doctorado/Laboratorio/17_aphasia_verbal_fluency_ucsf/mayor_a_2/nueva/resultados_machine_learning/' + key + '_all.xlsx', index=False)
        df_feature_importance.to_excel('C:/Franco/Doctorado/Laboratorio/17_aphasia_verbal_fluency_ucsf/mayor_a_2/nueva/resultados_machine_learningfeature_importance/' + key + '_all.xlsx', index=False)

        for name, model, params in models:
            df_feature_importance_modelo = df_feature_importance[df_feature_importance["modelo"] == name]

            # Identificar las columnas numéricas
            columnas_numericas = df_feature_importance_modelo.select_dtypes(include=['number'])
            
            # Calcular el promedio de los valores absolutos de las columnas numéricas
            esto = columnas_numericas.abs().mean()        
            
            grafico_barras_horizontales(df_feature_importance_modelo,title=key + "_" + name + '_all',path="C:/Franco/Doctorado/Laboratorio/17_aphasia_verbal_fluency_ucsf/mayor_a_2/nueva/resultados_machine_learning/feature_importance/" + key + "_" + name + '_promedio')

            df_feature_importance_modelo = df_feature_importance_count[df_feature_importance_count["modelo"] == name]

            # Identificar las columnas numéricas
            columnas_numericas = df_feature_importance_modelo.select_dtypes(include=['number'])
            
            # Calcular el promedio de los valores absolutos de las columnas numéricas
            esto = columnas_numericas.abs().mean()        
            
            grafico_barras_horizontales_count(df_feature_importance_modelo,title=key + "_" + name + '_all',path="C:/Franco/Doctorado/Laboratorio/17_aphasia_verbal_fluency_ucsf/mayor_a_2/nueva/resultados_machine_learning/feature_importance/" + key + "_" + name + '_all')

        df_promedio = df_results_inner.groupby('Modelo').agg(['mean', 'std'])
        df_promedio.columns = [f'{col}_{stat}' for col, stat in df_promedio.columns]
        df_promedio = df_promedio.reset_index()
        
        df_mejores = pd.DataFrame()
        # Expandir los diccionarios en columnas separadas
        df = pd.concat([df_results_inner.drop('Mejores hiperparámetros', axis=1), df_results_inner['Mejores hiperparámetros'].apply(pd.Series)], axis=1)
        
        # Agrupar por "Modelo" y calcular el promedio de los valores de cada clave
        df_mean = df.drop(['Iteracion', 'Accuracy','AUC', 'F1','Recall','Precision', 'Sensibilidad', 'Especificidad', 'UAR'], axis=1).groupby('Modelo').mean().reset_index()
        
        # Obtener las columnas (excepto 'Modelo')
        columnas = df_mean.columns.drop('Modelo')
        
        # Crear la nueva columna 'mejores hiperparámetros' sin claves NaN
        df_mean['mejores hiperparámetros'] = df_mean[columnas].apply(lambda row: {k: v for k, v in row.dropna().items()}, axis=1)
        
        # Seleccionar solo las columnas 'Modelo' y 'mejores hiperparámetros'
        df_resultado = df_mean[['Modelo', 'mejores hiperparámetros']]
        
        df_promedio = df_promedio.merge(df_resultado)
        # Calcula la moda de los hiperparámetros
        def calculate_mode(hyperparameters):
            flattened_params = [tuple(sorted(params.items())) for params in hyperparameters]
            counts = {param: flattened_params.count(param) for param in flattened_params}
            max_count = max(counts.values())
            modes = [param for param, count in counts.items() if count == max_count]
            return ', '.join([str(dict(param)) for param in modes])
        
        df_promedio['Moda hiperparámetros'] = df_results_inner.groupby('Modelo')['Mejores hiperparámetros'].apply(calculate_mode).values
        df_promedio.rename(columns = {'mejores hiperparámetros':'Promedio hiperparámetros'}, inplace = True)
        
        # Guardar los scores de decisión en un archivo Excel
        df_decision_scores = pd.DataFrame(decision_scores)
        
        df_promedio.to_excel('C:/Franco/Doctorado/Laboratorio/17_aphasia_verbal_fluency_ucsf/mayor_a_2/nueva/resultados_machine_learning/' + key + '_promedio.xlsx', index=False)
        df_decision_scores.to_excel('C:/Franco/Doctorado/Laboratorio/17_aphasia_verbal_fluency_ucsf/mayor_a_2/nueva/resultados_machine_learning/' + key + '_decision_scores.xlsx', index=False)



