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

def logistic_regression_cross_validation(df,group_labels,group_column,features,k_fold,random_seed = 123,normalization=False,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance="",data_input=False,feature_selection=0):
    
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

def svm_cross_validation(df,group_labels,group_column,features,k_fold,random_seed = 123,normalization=False,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance="",data_input=False,feature_selection=0):
    
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


def xgboost_cross_validation(df,group_labels,group_column,features,k_fold,random_seed = 123,normalization=False,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance="",data_input=False,feature_selection=0):
    
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

    df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features","-".join(group_labels),"XGBoost",str(k_fold),str(normalization),np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(auc),np.mean(f1)]
    
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

