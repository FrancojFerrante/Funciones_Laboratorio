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

def logistic_regression_cross_validation(df,group_labels,group_column,features,k_fold,random_seed = 123,normalization=False,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance=""):
    
    logisticRegr = LogisticRegression(max_iter=100000)

    df_resultados = pd.DataFrame(columns=[["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"]])
        
    df_actual = df.dropna(subset = features)
    
    if normalization:
        df_actual[features]=(df_actual[features]-df_actual[features].min())/(df_actual[features].max()-df_actual[features].min())
        
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
    accuracy = []
    precision = []
    recall = []
    auc = []
    f1 = []
    cm_total = [[0,0],[0,0]]
    
    importances_matrix = []

    for train_index, test_index in kf.split(df_actual, df_actual[group_column]):
        X_train = df_actual.iloc[train_index].loc[:, features]
        X_test = df_actual.iloc[test_index].loc[:,features]
        y_train = df_actual.iloc[train_index].loc[:,group_column]
        y_test = df_actual.iloc[test_index].loc[:,group_column]
    
        #Train the model
        logisticRegr.fit(X_train, y_train) #Training the model
        predictions = logisticRegr.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predictions))
        precision.append(metrics.precision_score(y_test, predictions))
        recall.append(metrics.recall_score(y_test, predictions))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
        auc.append(metrics.auc(fpr, tpr))
        f1.append(metrics.f1_score(y_test, predictions))
        
        cm_actual = metrics.confusion_matrix(y_test.values, predictions)
        cm_total[0][0] += cm_actual[0][0]
        cm_total[1][0] += cm_actual[1][0]
        cm_total[0][1] += cm_actual[0][1]
        cm_total[1][1] += cm_actual[1][1]
    
        importances_matrix.append(logisticRegr.coef_[0])

    df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features",group_labels[0]+"-"+group_labels[1],"Regresión logística",str(k_fold),str(normalization),np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(auc),np.mean(f1)]
    
    if n_importances>0:
        df_importances = pd.DataFrame(data={
            'attribute': features,
            'importance': 0
        })
        
        df_importances["importance"] = np.mean(np.vstack(importances_matrix),axis=0)
        df_importances["importance_abs"] = np.abs(df_importances["importance"])
        df_importances = df_importances.sort_values(by='importance_abs', ascending=False)
    
        n = 20
        df_importances = df_importances.nlargest(n, 'importance_abs')
    
        df_importances = df_importances.sort_values(by='importance', ascending=False)
    
        fig=plt.figure();
        plt.bar(x=df_importances['attribute'], height=df_importances['importance'], color='#087E8B')
        plt.title('Feature importances obtained from coefficients')
        plt.xticks(rotation='vertical')
        plt.savefig(path_feature_importance+".png",bbox_inches='tight')
        plt.close(fig)
    
    
    if path_confusion_matrix!= "":
        fig=plt.figure();
        sns.heatmap(cm_total, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', yticklabels=[group_labels[0],group_labels[1]], xticklabels=[group_labels[0],group_labels[1]]);
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = "Multi-features. k-folds: "+str(k_fold)
        plt.title(all_sample_title);
        plt.savefig(path_confusion_matrix+'.png')
        plt.close(fig)
    
    if path_excel!= "":
        df_resultados.to_excel(path_excel+".xlsx")
    
    return df_resultados

def svm_cross_validation(df,group_labels,group_column,features,k_fold,random_seed = 123,normalization=False,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance=""):
    
    svm_model = svm.SVC(kernel = "linear",max_iter=100000)

    df_resultados = pd.DataFrame(columns=[["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"]])
        
    df_actual = df.dropna(subset = features)
    
    if normalization:
        df_actual[features]=(df_actual[features]-df_actual[features].min())/(df_actual[features].max()-df_actual[features].min())
        
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
    accuracy = []
    precision = []
    recall = []
    auc = []
    f1 = []
    cm_total = [[0,0],[0,0]]
    
    importances_matrix = []

    for train_index, test_index in kf.split(df_actual, df_actual[group_column]):
        X_train = df_actual.iloc[train_index].loc[:, features]
        X_test = df_actual.iloc[test_index].loc[:,features]
        y_train = df_actual.iloc[train_index].loc[:,group_column]
        y_test = df_actual.iloc[test_index].loc[:,group_column]
    
        #Train the model
        svm_model.fit(X_train, y_train) #Training the model
        predictions = svm_model.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predictions))
        precision.append(metrics.precision_score(y_test, predictions))
        recall.append(metrics.recall_score(y_test, predictions))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
        auc.append(metrics.auc(fpr, tpr))
        f1.append(metrics.f1_score(y_test, predictions))
        
        cm_actual = metrics.confusion_matrix(y_test.values, predictions)
        cm_total[0][0] += cm_actual[0][0]
        cm_total[1][0] += cm_actual[1][0]
        cm_total[0][1] += cm_actual[0][1]
        cm_total[1][1] += cm_actual[1][1]
        
        importances_matrix.append(svm_model.coef_[0])

    df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features",group_labels[0]+"-"+group_labels[1],"SVM",str(k_fold),str(normalization),np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(auc),np.mean(f1)]
    
    if n_importances>0:
        df_importances = pd.DataFrame(data={
            'attribute': features,
            'importance': 0
        })
        
        df_importances["importance"] = np.mean(np.vstack(importances_matrix),axis=0)
        df_importances["importance_abs"] = np.abs(df_importances["importance"])
        df_importances = df_importances.sort_values(by='importance_abs', ascending=False)
    
        n = 20
        df_importances = df_importances.nlargest(n, 'importance_abs')
    
        df_importances = df_importances.sort_values(by='importance', ascending=False)
    
        fig=plt.figure();
        plt.bar(x=df_importances['attribute'], height=df_importances['importance'], color='#087E8B')
        plt.title('Feature importances obtained from coefficients')
        plt.xticks(rotation='vertical')
        plt.savefig(path_feature_importance+".png",bbox_inches='tight')
        plt.close(fig)
    
    
    if path_confusion_matrix!= "":
        fig=plt.figure();
        sns.heatmap(cm_total, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', yticklabels=[group_labels[0],group_labels[1]], xticklabels=[group_labels[0],group_labels[1]]);
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = "Multi-features. k-folds: "+str(k_fold)
        plt.title(all_sample_title);
        plt.savefig(path_confusion_matrix+'.png')
        plt.close(fig)
    
    if path_excel!= "":
        df_resultados.to_excel(path_excel+".xlsx")
    
    return df_resultados


def xgboost_cross_validation(df,group_labels,group_column,features,k_fold,random_seed = 123,normalization=False,path_confusion_matrix = "", path_excel = "",n_importances=0,path_feature_importance=""):
    
    xg_model = XGBClassifier(max_iter=100000)

    df_resultados = pd.DataFrame(columns=[["Random-Seed","Feature","Grupo","Clasificador","k-fold","Normalization","Accuracy","Precision","Recall","AUC","F1"]])
        
    df_actual = df.dropna(subset = features)
    
    if normalization:
        df_actual[features]=(df_actual[features]-df_actual[features].min())/(df_actual[features].max()-df_actual[features].min())
        
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
    accuracy = []
    precision = []
    recall = []
    auc = []
    f1 = []
    cm_total = [[0,0],[0,0]]
    
    importances_matrix = []

    for train_index, test_index in kf.split(df_actual, df_actual[group_column]):
        X_train = df_actual.iloc[train_index].loc[:, features]
        X_test = df_actual.iloc[test_index].loc[:,features]
        y_train = df_actual.iloc[train_index].loc[:,group_column]
        y_test = df_actual.iloc[test_index].loc[:,group_column]
    
        #Train the model
        xg_model.fit(X_train, y_train) #Training the model
        predictions = xg_model.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predictions))
        precision.append(metrics.precision_score(y_test, predictions))
        recall.append(metrics.recall_score(y_test, predictions))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
        auc.append(metrics.auc(fpr, tpr))
        f1.append(metrics.f1_score(y_test, predictions))
        
        cm_actual = metrics.confusion_matrix(y_test.values, predictions)
        cm_total[0][0] += cm_actual[0][0]
        cm_total[1][0] += cm_actual[1][0]
        cm_total[0][1] += cm_actual[0][1]
        cm_total[1][1] += cm_actual[1][1]
    
        importances_matrix.append(xg_model.feature_importances_)

    df_resultados.loc[len(df_resultados)] = [random_seed,"Multi-features",group_labels[0]+"-"+group_labels[1],"SVM",str(k_fold),str(normalization),np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(auc),np.mean(f1)]
    
    if n_importances>0:
        df_importances = pd.DataFrame(data={
            'attribute': features,
            'importance': 0
        })
        
        df_importances["importance"] = np.mean(np.vstack(importances_matrix),axis=0)
        df_importances["importance_abs"] = np.abs(df_importances["importance"])
        df_importances = df_importances.sort_values(by='importance_abs', ascending=False)
    
        n = 20
        df_importances = df_importances.nlargest(n, 'importance_abs')
    
        df_importances = df_importances.sort_values(by='importance', ascending=False)
    
        fig=plt.figure();
        plt.bar(x=df_importances['attribute'], height=df_importances['importance'], color='#087E8B')
        plt.title('Feature importances obtained from coefficients')
        plt.xticks(rotation='vertical')
        plt.savefig(path_feature_importance+".png",bbox_inches='tight')
        plt.close(fig)
    
    
    if path_confusion_matrix!= "":
        fig=plt.figure();
        sns.heatmap(cm_total, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', yticklabels=[group_labels[0],group_labels[1]], xticklabels=[group_labels[0],group_labels[1]]);
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = "Multi-features. k-folds: "+str(k_fold)
        plt.title(all_sample_title);
        plt.savefig(path_confusion_matrix+'.png')
        plt.close(fig)
    
    if path_excel!= "":
        df_resultados.to_excel(path_excel+".xlsx")
    
    return df_resultados