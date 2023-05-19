#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:56:17 2022

@author: daniel
"""

from sklearn import preprocessing, svm, model_selection,metrics
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer

# Para cualquier modelo
class ResultsGeneral_biclass_experiments_cv():
    def __init__(self):
        self.ResultsByFold = {}
        self.ResultsGeneral = {}
        self.ResultsGeneralByFold = {}
        self.Model_dict = {}
        
    def AddResultsInFold(self, keyFold, y_real_total, y_pred_total, params_dict, score_pred_total, normalizer, model):
        dictResults = {}
        tn, fp, fn, tp = metrics.confusion_matrix(y_real_total, y_pred_total).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_real_total,score_pred_total)
        
        
        dictResults['Acc'] = metrics.accuracy_score(y_real_total, y_pred_total)*100
        
        dictResults['Sen'] = tp/(tp+fn)*100
        dictResults['Spe'] = tn/(tn+fp)*100
        dictResults['F1'] = metrics.f1_score(y_real_total,y_pred_total)*100
        dictResults['auc'] = metrics.auc(fpr,tpr)
        dictResults['parms'] = params_dict

        dictResults['Normalizer'] = normalizer
        dictResults['y_real'] = y_real_total
        dictResults['y_pred'] = y_pred_total
        dictResults['score'] = score_pred_total
        dictResults['model'] = model


        self.ResultsByFold[keyFold] = dictResults
        
    def computeGeneralResults(self):
        dict_folds = list(self.ResultsByFold.values())
        results_folds = {}
        for key in dict_folds[0].keys():
            results_folds[key] = [d_aux[key] for d_aux in dict_folds]
            
        for key_target in list(dict_folds[0].keys())[0:5]:
            
            self.ResultsGeneral[key_target+' mean'] = np.mean(results_folds[key_target]) 
            self.ResultsGeneral[key_target+' std'] = np.std(results_folds[key_target]) 
        
        self.ResultsGeneral['y_real_complete'] = np.hstack(results_folds['y_real'])
        self.ResultsGeneral['y_pred_complete'] = np.hstack(results_folds['y_pred'])
        self.ResultsGeneral['y_score_complete'] = np.hstack(results_folds['score'])        
        self.ResultsGeneralByFold = results_folds

class ResultsGeneral_multiclass_experiments_cv():
    def __init__(self):
        self.ResultsByFold = {}
        self.ResultsGeneral = {}
        self.ResultsGeneralByFold = {}
        self.Model_dict = {}
        
    def AddResultsInFold(self, keyFold, y_real_total, y_pred_total, params_dict, score_pred_total, normalizer, model):
        dictResults = {}

        
        
        dictResults['Acc'] = metrics.accuracy_score(y_real_total, y_pred_total)*100
        dictResults['F1'] = metrics.f1_score(y_real_total,y_pred_total, average='weighted')*100
        dictResults['UAR'] = metrics.balanced_accuracy_score(y_real_total,y_pred_total) * 100


        dictResults['Normalizer'] = normalizer
        dictResults['y_real'] = y_real_total
        dictResults['y_pred'] = y_pred_total
        dictResults['scores'] = score_pred_total
        dictResults['model'] = model


        self.ResultsByFold[keyFold] = dictResults
        
    def computeGeneralResults(self):
        dict_folds = list(self.ResultsByFold.values())
        results_folds = {}
        for key in dict_folds[0].keys():
            results_folds[key] = [d_aux[key] for d_aux in dict_folds]
            
        for key_target in list(dict_folds[0].keys())[0:3]:
            
            self.ResultsGeneral[key_target+' mean'] = np.mean(results_folds[key_target]) 
            self.ResultsGeneral[key_target+' std'] = np.std(results_folds[key_target]) 
        
        self.ResultsGeneral['y_real_complete'] = np.hstack(results_folds['y_real'])
        self.ResultsGeneral['y_pred_complete'] = np.hstack(results_folds['y_pred'])
#        self.ResultsGeneral['y_score_complete'] = np.hstack(results_folds['score'])

        self.ResultsGeneralByFold = results_folds 
        
# Para SVM y compatble con SimpleClassifier
class ClassifierResultsBiclassCrossValidation():
    def __init__(self):
        self.ResultsByFold = {}
        self.ResultsGeneral = {}
        self.ResultsGeneralByFold = {}
        self.Model_dict = {}
        
    def AddResultsInFold(self, keyFold, y_real_total, y_pred_total, best_C, best_G, score_pred_total, normalizer, model):
        dictResults = {}
        tn, fp, fn, tp = metrics.confusion_matrix(y_real_total, y_pred_total).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_real_total,score_pred_total)
        
        
        dictResults['Acc'] = metrics.accuracy_score(y_real_total, y_pred_total)*100
        
        dictResults['Sen'] = tp/(tp+fn)*100
        dictResults['Spe'] = tn/(tn+fp)*100
        dictResults['F1'] = metrics.f1_score(y_real_total,y_pred_total)*100
        dictResults['auc'] = metrics.auc(fpr,tpr)
        dictResults['C'] = best_C
        dictResults['Gamma'] = best_G

        dictResults['Normalizer'] = normalizer
        dictResults['y_real'] = y_real_total
        dictResults['y_pred'] = y_pred_total
        dictResults['score'] = score_pred_total
        dictResults['model'] = model

        self.ResultsByFold[keyFold] = dictResults
        
    def computeGeneralResults(self):
        dict_folds = list(self.ResultsByFold.values())
        results_folds = {}
        for key in dict_folds[0].keys():
            results_folds[key] = [d_aux[key] for d_aux in dict_folds]
            
        for key_target in list(dict_folds[0].keys())[0:5]:
            
            self.ResultsGeneral[key_target+' mean'] = np.mean(results_folds[key_target]) 
            self.ResultsGeneral[key_target+' std'] = np.std(results_folds[key_target]) 
        
        self.ResultsGeneral['y_real_complete'] = np.hstack(results_folds['y_real'])
        self.ResultsGeneral['y_pred_complete'] = np.hstack(results_folds['y_pred'])
        self.ResultsGeneral['y_score_complete'] = np.hstack(results_folds['score'])
        self.ResultsGeneral['best C'] = stats.mode(results_folds['C'])[0][0]
        self.ResultsGeneral['best Gamma'] = stats.mode(results_folds['Gamma'])[0][0]        
        self.ResultsGeneralByFold = results_folds


class ClassifierResultsMulticlassCrossValidation():
    def __init__(self):
        self.ResultsByFold = {}
        self.ResultsGeneral = {}
        self.ResultsGeneralByFold = {}
        self.Model_dict = {}
        
    def AddResultsInFold(self, keyFold, y_real_total, y_pred_total, best_C, best_G, score_pred_total, normalizer, model):
        dictResults = {}

        
        
        dictResults['Acc'] = metrics.accuracy_score(y_real_total, y_pred_total)*100
        dictResults['F1'] = metrics.f1_score(y_real_total,y_pred_total, average='weighted')*100
        dictResults['UAR'] = metrics.balanced_accuracy_score(y_real_total,y_pred_total) * 100
        dictResults['C'] = best_C
        dictResults['Gamma'] = best_G

        dictResults['Normalizer'] = normalizer
        dictResults['y_real'] = y_real_total
        dictResults['y_pred'] = y_pred_total
        dictResults['score'] = score_pred_total
        dictResults['model'] = model


        self.ResultsByFold[keyFold] = dictResults
        
    def computeGeneralResults(self):
        dict_folds = list(self.ResultsByFold.values())
        results_folds = {}
        for key in dict_folds[0].keys():
            results_folds[key] = [d_aux[key] for d_aux in dict_folds]
            
        for key_target in list(dict_folds[0].keys())[0:5]:
            
            self.ResultsGeneral[key_target+' mean'] = np.mean(results_folds[key_target]) 
            self.ResultsGeneral[key_target+' std'] = np.std(results_folds[key_target]) 
        
        self.ResultsGeneral['y_real_complete'] = np.hstack(results_folds['y_real'])
        self.ResultsGeneral['y_pred_complete'] = np.hstack(results_folds['y_pred'])
        self.ResultsGeneral['y_score_complete'] = np.hstack(results_folds['score'])
        self.ResultsGeneral['best C'] = stats.mode(results_folds['C'])[0][0]
        self.ResultsGeneral['best Gamma'] = stats.mode(results_folds['Gamma'])[0][0]
        
        self.ResultsGeneralByFold = results_folds        
        
          


def SimpleClassifier(X_train, Y_train, X_test, Y_test, number_folds_validation, results_object, keyFold, kernel_linear=False, n_job=-1):
    
    imputer_transfor=KNNImputer().fit(X_train)
    X_train,X_test=imputer_transfor.transform(X_train),imputer_transfor.transform(X_test)
    
    normalize_transfor=preprocessing.StandardScaler().fit(X_train)
    X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)
    
    C_array =np.geomspace(1e-4,1e3,8)
    G_array=np.geomspace(1e-6,1e6,13)
    
    #Best parameters and testSet
    if kernel_linear:
        parameters = {'kernel':['linear'], 'C':C_array}
        svc = svm.SVC(class_weight = 'balanced')
        clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
        clf.fit(X_train_N,Y_train)                
        best_C=clf.best_estimator_.C
        best_G='scale'
    
    else:
        parameters = {'kernel':['rbf'], 'C':C_array,'gamma':G_array }
        svc = svm.SVC(class_weight = 'balanced')
        clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
        clf.fit(X_train_N,Y_train)                
        best_C=clf.best_estimator_.C
        best_G=clf.best_estimator_.gamma
    
    print('####'+str(clf.best_estimator_.kernel)+'#####')
    if kernel_linear:
        SVM_Classifier = svm.SVC(kernel = clf.best_estimator_.kernel , C=best_C, class_weight = 'balanced')
    else:    
        SVM_Classifier = svm.SVC(kernel = clf.best_estimator_.kernel , gamma=best_G, C=best_C, class_weight = 'balanced')
    
    SVM_Classifier.fit(X_train_N,Y_train)

    y_pred=SVM_Classifier.predict(X_test_N)
    score_test = SVM_Classifier.decision_function(X_test_N) 
    
    results_object.AddResultsInFold(keyFold, Y_test, y_pred, best_C, best_G, score_test, normalize_transfor, SVM_Classifier)
    

    return results_object