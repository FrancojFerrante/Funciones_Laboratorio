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
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

def pipeline_cross_validation(df,ml_classifier,classifier_name,group_labels,group_column,features,k_fold,random_seed = None,normalization=True,data_input=False,feature_selection=False,multi=False,n_repeats = 1):
    

    print("Classifier: " + classifier_name)
    print("Groups: " + "-".join(group_labels))
    print("Fold: " + str(k_fold))
    # pipe_a = Pipeline(steps=[('imp', SimpleImputer()),
    #                      ('scale', StandardScaler())])
    transformers = []
    pipeline_list = []
    
    if data_input | normalization:
        if data_input:
            transformers.append(('imputer',KNNImputer()))
            
        if normalization:
            transformers.append(('scaler',MinMaxScaler()))
        pipe_transformer = Pipeline(steps=transformers)
        preprocessor = ColumnTransformer(transformers= [('preprocessor',pipe_transformer,features)])
        pipeline_list.append(('preprocessor',preprocessor))
        
    if feature_selection:
        svm_model = svm.SVC(kernel = "linear",max_iter=100000)
        if not multi:
            pipeline_list.append(('feat_sel',RFECV(estimator=svm_model,step=1,scoring='roc_auc')))
        else:
            pipeline_list.append(('feat_sel',RFECV(estimator=svm_model,step=1,scoring='accuracy')))
                        
    pipeline_list.append(('model',ml_classifier))
    kf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=random_seed)
    
    if multi:
        scoring = {'acc': 'accuracy'}
    else:
        scoring = {'acc': 'accuracy',
                    'prec_micro': 'precision_micro',
                    'rec_micro': 'recall_micro',
                    'auc':'roc_auc',
                    'f1_score':'f1_micro'
                   }
    
    pipeline = Pipeline(pipeline_list)

    # scores["estimator"] devuelve tantos Pipelines como n_splits en cross-validation
    scores = cross_validate(pipeline, df[features], df[group_column].values.ravel(), scoring=scoring,
                         cv=kf, return_train_score=False,return_estimator=True,verbose=1,n_jobs=-1)
    
    scores["classifier"]=classifier_name
    scores["group"]=group_labels
    scores["k_fold"]=k_fold
    if multi:
        df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy"])
        df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),classifier_name,str(k_fold),str(normalization),np.mean(scores['test_acc'])]
    else:
        df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"])
        df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),classifier_name,str(k_fold),str(normalization),np.mean(scores['test_acc']),np.mean(scores['test_prec_micro']),np.mean(scores['test_rec_micro']),np.mean(scores['test_auc']),np.mean(scores['test_f1_score'])]
   
    return scores,df_resultados

def pipeline_cross_validation_hyper_opt(df,ml_classifier,classifier_name,group_labels,group_column,features,k_fold,random_seed = None,normalization=True,data_input=False,feature_selection=False,multi=False,n_repeats = 1):
    
    # pipe_a = Pipeline(steps=[('imp', SimpleImputer()),
    #                      ('scale', StandardScaler())])
    transformers = []
    pipeline_list = []
    
    if data_input | normalization:
        if data_input:
            transformers.append(('imputer',KNNImputer()))
            
        if normalization:
            transformers.append(('scaler',MinMaxScaler()))
        pipe_transformer = Pipeline(steps=transformers)
        preprocessor = ColumnTransformer(transformers= [('preprocessor',pipe_transformer,features)])
        pipeline_list.append(('preprocessor',preprocessor))
        
    if feature_selection:
        svm_model = svm.SVC(kernel = "linear",max_iter=100000)
        if not multi:
            pipeline_list.append(('feat_sel',RFECV(estimator=svm_model,step=1,scoring='roc_auc')))
        else:
            pipeline_list.append(('feat_sel',RFECV(estimator=svm_model,step=1,scoring='accuracy')))
                        
            # pipe = Pipeline([('imputer', KNNImputer(n_neighbors=5)),
            #                 ('scaler', MinMaxScaler()),
            #                   ('selector', SelectKBest(f_classif, k=5)),
            #                   ('classifier', LogisticRegression(random_state=0))])

            # search_space = [{'selector__k': [3,5,10]},
            #                 {'imputer': [KNNImputer()],
            #                       'imputer__n_neighbors': [3,5,10]},
            #                 {'classifier': [XGBClassifier(random_state=0,n_estimators=5000,learning_rate=0.01)],
            #                   'classifier__loss': ["deviance","exponential"],
            #                   'classifier__learning_rate': [0.001,0.01,0.1,1],
            #                   'classifier__n_estimators': [100,1000,3000,5000,10000,50000],
            #                   'classifier__max_depth': [1,3,5,10,20]}]

            # search_space = [{'selector__k': [1, 2, 3, 4,5,6,7,8,9,10,None]},
            #                 {'imputer': [KNNImputer()],
            #                      'imputer__n_neighbors': [3,4,5,6,7,8,9,10]},
            #                 # {'impute__strategy':[KNNImputer(),"mean", "median", "most_frequent"]},
            #                 {'classifier': [LogisticRegression(random_state=0,max_iter=100000)],
            #                   'classifier__C': [0.01, 0.1, 1.0]},
            #                 {'classifier': [svm.SVC(random_state=0,max_iter=100000)],
            #                   'classifier__kernel': ["linear", "poly", "rbf", "sigmoid", "precomputed"]},
            #                 {'classifier': [XGBClassifier(random_state=0,n_estimators=5000,learning_rate=0.01)]}]

            # clf = GridSearchCV(pipe, search_space, cv=5, verbose=0,scoring="accuracy",)
            # clf = clf.fit(df_combinado_ctr_ad[columnas_estadisticas], df_combinado_ctr_ad["Grupo"])

            # clf.best_estimator_

            # clf.best_score_
            
    pipeline_list.append(('model',ml_classifier))
    kf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=random_seed)
    
    if multi:
        scoring = {'acc': 'accuracy'}
    else:
        scoring = {'acc': 'accuracy',
                    'prec_micro': 'precision_micro',
                    'rec_micro': 'recall_micro',
                    'auc':'roc_auc',
                    'f1_score':'f1_micro'
                   }
    
    pipeline = Pipeline(pipeline_list)

    # scores["estimator"] devuelve tantos Pipelines como n_splits en cross-validation
    scores = cross_validate(pipeline, df[features], df[group_column], scoring=scoring,
                         cv=kf, return_train_score=False,return_estimator=True)
    
    scores["classifier"]=classifier_name
    scores["group"]=group_labels
    scores["k_fold"]=k_fold
    if multi:
        df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy"])
        df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),classifier_name,str(k_fold),str(normalization),np.mean(scores['test_acc'])]
    else:
        df_resultados = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"])
        df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),classifier_name,str(k_fold),str(normalization),np.mean(scores['test_acc']),np.mean(scores['test_prec_micro']),np.mean(scores['test_rec_micro']),np.mean(scores['test_auc']),np.mean(scores['test_f1_score'])]
   
    return scores,df_resultados
    
def menu_clasificador(clasificador, df,df_labels,columna_features,columnas_grupo,k_folds,path_confusion_matrix,path_feature_importance,data_input=False,feature_selection=0,multi=False, random_seed = None,n_repeats=1):
    
    
    df_clasificador_multi = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold",\
                                                  "Normalization","Accuracy","Precision","Recall","AUC","F1"])
    scores_list = []
    model = None
    
    if clasificador == "regresion_logistica":
        model = LogisticRegression(max_iter=100000,)
    elif clasificador == "svm":
        model = svm.SVC(kernel = "linear",max_iter=100000)
    elif clasificador == "xgboost":
        model = XGBClassifier()
        
    for i_label, df_combinado in enumerate(df):
        scores_list.append([])
        for folds_counter,k_fold in enumerate(k_folds):
            (scores_clasi, df_clasif) = pipeline_cross_validation(df_combinado,model, clasificador,\
                                      df_labels[i_label], columnas_grupo, columna_features, k_fold,random_seed = random_seed,\
                                      normalization=True, data_input = data_input,feature_selection=feature_selection,multi=multi,n_repeats=n_repeats)
            df_clasificador_multi = pd.concat([df_clasificador_multi, df_clasif])
            scores_list[i_label].append(scores_clasi)
                

    return (scores_list,df_clasificador_multi)

def clasificador_personalizado(ml_classifier,ml_classifier_name, df,df_labels,columna_features,columnas_grupo,tipo_columnas,k_folds,path,path_confusion_matrix,path_feature_importance,data_input=False,feature_selection=0,multi=False, random_seed = None,n_repeats=1):
    
    print("Base: " + tipo_columnas)
    print("------------------------")
    
    df_clasificador_multi = pd.DataFrame(columns=["Random-Seed","Feature","Grupo","Clasificador","k-fold",\
                                                  "Normalization","Accuracy","Precision","Recall","AUC","F1"])
    scores_list = []
    model = ml_classifier
        
    for i_label, df_combinado in enumerate(df):
        scores_list.append([])
        for folds_counter,k_fold in enumerate(k_folds):
            (scores_clasi, df_clasif) = pipeline_cross_validation(df_combinado,model, ml_classifier_name,\
                                      df_labels[i_label], columnas_grupo, columna_features, k_fold,random_seed = random_seed,\
                                      normalization=True, data_input = data_input,feature_selection=feature_selection,multi=multi,n_repeats=n_repeats)
            df_clasificador_multi = pd.concat([df_clasificador_multi, df_clasif])

            scores_list[i_label].append(scores_clasi)
    
    df_clasificador_multi.to_excel(path+"/resultados_machine_learning/resultados_"+ml_classifier_name+"_"+\
                                              tipo_columnas+".xlsx")           

    return (scores_list,df_clasificador_multi)

def tres_clasificadores(clasificadores,df,df_labels,columnas_features,columnas_grupo,tipo_columnas,k_folds,path,data_input=False,feature_selection=False,multi=False,random_seed=None,n_repeats=1):
    
    scores_list = []
    resultados=[]
    
    print("Base: " + tipo_columnas)
    print("------------------------")

    for clasificador in clasificadores:
        (scores,df_clasificador_multi) = menu_clasificador(clasificador,df,df_labels,columnas_features,columnas_grupo,k_folds,path + "//imagenes_matriz_confusion_"+clasificador+"//multi_feature"\
                                                            + tipo_columnas,path+"/feature_importance_"+clasificador+"/multi_feature_"\
                                                            + tipo_columnas+"_", data_input = data_input,feature_selection=feature_selection,multi=multi,random_seed=random_seed,n_repeats=n_repeats)
    
        df_clasificador_multi.to_excel(path+"/resultados_machine_learning/resultados_"+clasificador+"_"+\
                                                 tipo_columnas+".xlsx")
        resultados.append(df_clasificador_multi)
        scores_list.append(scores)

    return (scores_list,resultados)

# Ploteo feature importance entregado por el clasificador
def feature_importance_not_feat_selection(scores_list,databases_labels,groups_label,algorithm_label,n_feature_importance,n_repeats,cwd):
        
    for i_base, base in enumerate(scores_list):
        if "feat" not in databases_labels[i_base]:

            for i_algorithm, algorithm in enumerate(base):
                for i_group,group in enumerate(algorithm):
                    for i_fold,fold in enumerate(group):
                        # creating the dataset
                        importances_matrix = []

                        for i_pipe,pipe in enumerate(fold["estimator"]):
                            invert_op = hasattr(pipe["model"], "coef_")
                            if invert_op:
                                importances_matrix.append([abs(ele) for ele in pipe["model"].coef_])
                            else:
                                importances_matrix.append(pipe["model"].feature_importances_)


                            # if (algorithm_label[i_algorithm] == "regresion_logistica") | (algorithm_label[i_algorithm] == "svm"):
                            #     importances_matrix.append([abs(ele) for ele in pipe["model"].coef_])
                            # elif algorithm_label[i_algorithm] == "xgboost":
                            #     importances_matrix.append(pipe["model"].feature_importances_)

                        y_values = np.mean(np.vstack(importances_matrix),axis=0)
                        x_values = fold["estimator"][0]["preprocessor"]._columns[0]
                        data = {'x_values':x_values,'y_values':y_values}
                        df_feature_importance = pd.DataFrame(data).sort_values('y_values', ascending=False).head(n_feature_importance)
                        
                        plt.figure(figsize = (10, 5))
                        plt.bar(df_feature_importance["x_values"],df_feature_importance["y_values"], color ='blue', width = 0.4)
                        plt.xlabel("Base")
                        plt.ylabel("Feature importance")
                        plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 90-degrees
                        plt.title(databases_labels[i_base] + " with " + algorithm_label[i_algorithm] + " " + groups_label[i_group])

                        # plt.show()
                        plt.savefig(cwd+"_"+algorithm_label[i_algorithm]+"/"+databases_labels[i_base]+"_"+groups_label[i_group]+"_"+str(n_repeats)+".png",\
                                    bbox_inches='tight')

def feature_importance_feat_selection(scores_list,databases_labels,groups_label,algorithm_label,n_repeats,cwd):
    # ploteo la feature importance por cantidad de veces que fue elegida la feature.

    for i_base, base in enumerate(scores_list):
        if "feat_sel" in databases_labels[i_base]:
            for i_algorithm, algorithm in enumerate(base):
                for i_group,group in enumerate(algorithm):
                    for i_fold,fold in enumerate(group):
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
                        
                        fig = plt.figure(figsize = (10, 5))
            
                        # creating the bar plot
                        plt.bar(courses, values, color ='blue',
                                width = 0.4)
                         
                        plt.xlabel("Base")
                        plt.ylabel("N° times feature selected")
                        plt.xticks(rotation=90)
                        plt.title(databases_labels[i_base] + " with " + algorithm_label[i_algorithm] + " " + groups_label[i_group])
                        # plt.show()
                        
                        plt.savefig(cwd+"_"+algorithm_label[i_algorithm]+"/"+databases_labels[i_base]+"_"+groups_label[i_group]+"_"+str(n_repeats)+".png",\
                                    bbox_inches='tight')

           

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
