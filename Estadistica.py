# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:50:40 2021

@author: franc
"""
# Estadistica.py>
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import pingouin as pg

import matplotlib.pyplot as plt
plt.ioff()
import seaborn as sns

from statsmodels.stats.multicomp import pairwise_tukeyhsd

           
import matplotlib.pyplot as plt

from scipy.stats import levene, f_oneway, kruskal, friedmanchisquare, wilcoxon, bartlett
from pathlib import Path
import scikit_posthocs as sp
from statsmodels.stats.anova import AnovaRM
import itertools 

def chi_cuadrado_transcripciones(df_list, categoria, posibilidades):
    
    if (len(df_list)<=0):
        print("No es posible realizar el análisis. No se han pasado dataframes como argumentos")
        return
    # Me quedo con las filas que tienen transcripciones
    df_transcripciones = []
    if ('has_Transcription' in df_list[0].columns):
        for df in df_list:
            df_transcripciones.append(df[df['has_Transcription']==1])
        
    
    cantidad_transcripciones_categoria0 = []
    cantidad_transcripciones_categoria1 = []
    # Cuento la cantidad para cada categoría, para cada enfermedad
    for df in df_transcripciones:
        
        cantidad_transcripciones_categoria0.append(df[df[categoria] == posibilidades[0]][categoria].count())
        cantidad_transcripciones_categoria1.append(df[df[categoria] == posibilidades[1]][categoria].count())

    # Armo la tabla de contingencia
    contingencia = []
    for i in range(0,len(cantidad_transcripciones_categoria0)):
        contingencia.append([cantidad_transcripciones_categoria0[i],cantidad_transcripciones_categoria1[i]])

    data = contingencia
    _, p, _, _ = chi2_contingency(data,correction=True)
    chi2, _, _, _ = chi2_contingency(data,correction=False)

    n = 0
    for i in range(0,len(cantidad_transcripciones_categoria0)):
        n+=cantidad_transcripciones_categoria0[i]
        n+=cantidad_transcripciones_categoria1[i]
    phi = np.sqrt(chi2/n)
      
    return p,phi

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

def remove_outlier_IQR(df,columna1,columna2):
    Q1_1=df[columna1].quantile(0.25)
    Q3_1=df[columna1].quantile(0.75)
    Q1_2=df[columna2].quantile(0.25)
    Q3_2=df[columna2].quantile(0.75)
    IQR_1=Q3_1-Q1_1
    IQR_2=Q3_2-Q1_2
    
    df_final = pd.DataFrame(columns=df.columns)
    for i,row in df.iterrows():
        if ((row[columna1]>=(Q1_1-1.5*IQR_1))) & ((row[columna1]<=(Q3_1+1.5*IQR_1))) &\
            ((row[columna2]>=(Q1_2-1.5*IQR_2))) & ((row[columna2]<=(Q3_2+1.5*IQR_2))):
            df_final = df_final.append(row,ignore_index=True)
    return df_final

def remove_outlier_sd(df,columna1,columna2,sd_condition = 3):
    mean_1 = np.mean(df[columna1], axis=0)
    sd_1 = np.std(df[columna1], axis=0)
    mean_2 = np.mean(df[columna2], axis=0)
    sd_2 = np.std(df[columna2], axis=0)
    
    df_sin_outliers = pd.DataFrame(columns=df.columns)
    
    for i,row in df.iterrows():
        if ((row[columna1] >= mean_1 - (sd_condition * sd_1)) &
            (row[columna1] <= mean_1 + (sd_condition * sd_1)) &
            (row[columna2] >= mean_2 - (sd_condition * sd_2)) &
            (row[columna2] <= mean_2 + (sd_condition * sd_2))):
            df_sin_outliers = df_sin_outliers.append(row,ignore_index=True)
    
    return df_sin_outliers


def anova_mixto_2x2(df_controles,df_no_controles,feature,columna_id,columna_grupo,columna_feature_1,columna_feature_2,txt_no_control):
    """
    Calcula anova mixto 2x2 para los dos grupos y las dos features pasadas. Devuelve el resultado del anova mixto sumando 
    una columna que identifique a la feature, otra a los grupos comparados y dos que indiquen el n para el grupo control y para el no control.

    Parameters
    ----------
    resultado : pandas.DataFrame
        Dataframe with columns ["Grupo","Prueba","Source","SS","DF1","DF2","MS","MSE","F","p-unc","np2","n_ctr","n_no_ctr"].
    df_controles : pandas.DataFrame
        df con los datos de los controles.
    df_no_controles : pandas.DataFrame
        df con los datos de los no controles.
    feature : string
        Nombre de la feature sobre la cual se está haciendo anova mixto.
    columna_id : string
        Columna que posee el id de los participantes.
    columna_grupo : string
        Columna que posee el grupo al que pertenece el participante.
    columna_feature_1 : string
        Columna que posee los valores de la primer feature para cada participante.
    columna_feature_2 : string
        Columna que posee los valores de la segunda feature para cada participante.
    txt_no_control : string
        Texto que identifica al tipo de grupo no ctr. Ej: "AD", "FTD", "PD", etc.

    Returns
    -------
    resultado : pandas.Dataframe
        Dataframe con la misma estructura que el recibido pero con nuevas filas.

    """
    
    df_ctr_features = df_controles[[columna_id,columna_grupo,columna_feature_1,columna_feature_2]]
    df_no_ctr_features = df_no_controles[[columna_id,columna_grupo,columna_feature_1,columna_feature_2]]
    
    df_combinado = df_ctr_features.append(df_no_ctr_features)

    df_acomodado = pd.melt(df_combinado.reset_index(), id_vars=[columna_id,columna_grupo], value_vars=[columna_feature_1, columna_feature_2])
    df_acomodado.rename(columns={"variable": "fluencia", "value": feature}, inplace=True)

    df_acomodado['fluencia'] = df_acomodado['fluencia'].astype(str)
    df_acomodado['Grupo'] = df_acomodado['Grupo'].astype(str)
    df_acomodado['Codigo'] = df_acomodado['Codigo'].astype(str)
    df_acomodado[feature] = pd.to_numeric(df_acomodado[feature])

    if np.all((df_acomodado[feature] == 0)) == True:
        print("Son todos 0 en",feature)
    else:
        df_resultado = pg.mixed_anova(dv=feature, between='Grupo', within='fluencia', subject='Codigo', data=df_acomodado)
        new_col = [feature, feature, feature]  # can be a list, a Series, an array or a scalar   
        df_resultado.insert(loc=0, column='Prueba', value=new_col)
        new_col = ["CTR-"+txt_no_control, "CTR-"+txt_no_control, "CTR-"+txt_no_control]  # can be a list, a Series, an array or a scalar   
        df_resultado.insert(loc=0, column='Grupo', value=new_col)
        df_resultado["n_ctr"] = len(df_ctr_features)
        df_resultado["n_no_ctr"] = len(df_no_ctr_features)
        resultado = df_resultado[["Grupo","Prueba","Source","SS","DF1","DF2","MS","F","p-unc","np2","n_ctr","n_no_ctr"]]
        resultado['MSE'] = resultado['MS']/resultado["F"]

    return resultado

def anova_mixto_2x2_con_y_sin_outliers(columnas,prueba_1,prueba_2,df_controles,dfs_no_control,texto_no_control, outlier_condition, path_files="", save_boxplot = True, save_excel_with_outliers = True, save_excel_without_outliers = True):
    
    resultado_total = pd.DataFrame(columns=["Grupo","Prueba","Source","SS","DF1","DF2","MS","MSE","F","p-unc","np2","n_ctr","n_no_ctr"])
    resultado_total_sin_outliers = pd.DataFrame(columns=["Grupo","Prueba","Source","SS","DF1","DF2","MS","MSE","F","p-unc","np2","n_ctr","n_no_ctr"])
    resul_post_hoc = pd.DataFrame(columns=["Factor_a","Factor_b","mean_a","mean_b","diff","se","t","p-value","cohens_d"])
    resul_post_hoc_sin_outliers = pd.DataFrame(columns=["Factor_a","Factor_b","mean_a","mean_b","diff","se","t","p-value","cohens_d"])
    
    for columna in columnas:
        print(columna)
        separado = columna.split("_")
        col_fonologica = separado[0]+"_"+prueba_1+"_"+"_".join(separado[1:])
        col_semantica = separado[0]+"_"+prueba_2+"_"+"_".join(separado[1:])

        # Elimino outliers para controles
        if outlier_condition == "iqr":
            df_ctr_sin_outliers = remove_outlier_IQR(df_controles[["Codigo","Grupo",col_fonologica,col_semantica]],\
                                                         col_fonologica,col_semantica)
        elif outlier_condition == "sd":
            df_ctr_sin_outliers = remove_outlier_sd(df_controles[["Codigo","Grupo",col_fonologica,col_semantica]],col_fonologica,col_semantica,3)
        
        df_combinado_with_outliers_plot = df_controles[["Codigo","Grupo",col_fonologica,col_semantica]]
        df_combinado_without_outliers_plot = df_ctr_sin_outliers
        for i_no_control,df_no_control in enumerate(dfs_no_control):
            resultado = anova_mixto_2x2(df_controles,df_no_control,columna,"Codigo","Grupo",\
                                   col_fonologica,col_semantica,texto_no_control[i_no_control])
            resultado_total = resultado_total.append(resultado)
        
           
            # Elimino outliers para no controles
            if outlier_condition == "iqr":
                df_no_control_sin_outliers = remove_outlier_IQR(df_no_control[["Codigo","Grupo",col_fonologica,col_semantica]],\
                                                                    col_fonologica,col_semantica)
            elif outlier_condition == "sd":
                df_no_control_sin_outliers = remove_outlier_sd(df_no_control[["Codigo","Grupo",col_fonologica,col_semantica]], col_fonologica, col_semantica,3)
                
            resultado_sin_outliers = anova_mixto_2x2(df_ctr_sin_outliers,df_no_control_sin_outliers,columna,"Codigo","Grupo",\
                                   col_fonologica,col_semantica,texto_no_control[i_no_control])
            resultado_total_sin_outliers = resultado_total_sin_outliers.append(resultado_sin_outliers)

            df_combinado = df_controles[["Codigo","Grupo",col_fonologica,col_semantica]].append(df_no_control[["Codigo","Grupo",col_fonologica,col_semantica]])
            df_combinado.to_csv(path_files+"//features_with_outliers/" + columna+"_CTR-"+texto_no_control[i_no_control]+".csv")

            # Check if save excel with outliers
            if save_excel_with_outliers & (len(resultado[(resultado["p-unc"]<0.05) & (resultado["Source"] == "Interaction")]) > 0):
                df_acomodado = pd.melt(df_combinado.reset_index(), id_vars=["Codigo",'Grupo'], value_vars=[col_fonologica, col_semantica])                
                aux_nombre = col_fonologica.split("_")
                df_acomodado.rename(columns={"variable": "fluencia", "value": aux_nombre[0]+"_"+"_".join(aux_nombre[2:])}, inplace=True)
                
                # df_acomodado.pairwise_tukey(dv='body_mass_g', between='species').round(3)

                df_acomodado.to_csv(path_files+"//databases_with_outliers_interaccion_significativos_single_column/" + columna+"_CTR-"+texto_no_control[i_no_control]+".csv")
                df_combinado.to_csv(path_files+"//databases_with_outliers_interaccion_significativos/" + columna+"_CTR-"+texto_no_control[i_no_control]+".csv")
            
            # Check if save excel without outliers
            if save_excel_without_outliers & (len(resultado_sin_outliers[(resultado_sin_outliers["p-unc"]<0.05) & (resultado_sin_outliers["Source"] == "Interaction")]) > 0):
                df_combinado = df_ctr_sin_outliers.append(df_no_control_sin_outliers)
                df_acomodado = pd.melt(df_combinado.reset_index(), id_vars=["Codigo",'Grupo'], value_vars=[col_fonologica, col_semantica])
                aux_nombre = col_fonologica.split("_")
                df_acomodado.rename(columns={"variable": "fluencia", "value": aux_nombre[0]+"_"+"_".join(aux_nombre[2:])}, inplace=True)
                df_acomodado.to_csv(path_files+"//databases_without_outliers_interaccion_significativos_single_column/" + columna+"_CTR-"+texto_no_control[i_no_control]+".csv")
                df_combinado.to_csv(path_files+"//databases_without_outliers_interaccion_significativos/" + columna+"_CTR-"+texto_no_control[i_no_control]+".csv")

            df_combinado_with_outliers_plot = df_combinado_with_outliers_plot.append(df_no_control[["Codigo","Grupo",col_fonologica,col_semantica]])
            df_combinado_without_outliers_plot = df_combinado_without_outliers_plot.append(df_no_control_sin_outliers)
        
        if save_boxplot:
            # With outliers
            df_combinado_with_outliers_plot = pd.melt(df_combinado_with_outliers_plot.reset_index(), id_vars=["Codigo",'Grupo'], value_vars=[col_fonologica, col_semantica])
            df_combinado_with_outliers_plot.rename(columns={"variable": "fluencia", "value": columna}, inplace=True)
            df_combinado_with_outliers_plot["fluencia"].replace({col_fonologica: "Fluencia fonológica", col_semantica: "Fluencia semántica"}, inplace=True)
    
    
            fig=plt.figure();
            sns.boxplot(x='Grupo', y=columna, hue='fluencia', data=df_combinado_with_outliers_plot)
            plt.savefig(path_files+"//imagenes_with_outliers_boxplots/"+columna + '.png')
            plt.close(fig)
            
            # Without outliers
            df_combinado_without_outliers_plot = pd.melt(df_combinado_without_outliers_plot.reset_index(), id_vars=["Codigo",'Grupo'], value_vars=[col_fonologica, col_semantica])
            df_combinado_without_outliers_plot.rename(columns={"variable": "fluencia", "value": columna}, inplace=True)
            df_combinado_without_outliers_plot["fluencia"].replace({col_fonologica: "Fluencia fonológica", col_semantica: "Fluencia semántica"}, inplace=True)
    
    
            fig=plt.figure();
            sns.boxplot(x='Grupo', y=columna, hue='fluencia', data=df_combinado_without_outliers_plot)
            plt.savefig(path_files+"//imagenes_without_outliers_boxplots/"+columna + '.png')
            plt.close(fig)
              
    return resultado_total, resultado_total_sin_outliers

def check_assumptions_anova(data,factors,interaction=False,variables=None,threshold_pval=.05,shapiro=True):

    if variables == None:
        variables = [col for col in data.columns if isinstance(data[col][0],float,int)]

    shapiro_results = pd.DataFrame(index=variables)
    homoscedasticity_results = pd.DataFrame(index=variables)

    if (interaction==True) & (len(factors) == 2):
        for i, row in data.iterrows():
            data.loc[i,f'{factors[0]}_{factors[1]}'] = data.loc[i,factors[0]] + '_' + data.loc[i,factors[1]]
        
        factors.append(f'{factors[0]}_{factors[1]}')

    normality_ok = pd.DataFrame(index=variables,columns=[f'normality_{factor}' for factor in factors])
    homoscedasticity_ok = pd.DataFrame(index=variables,columns=[f'homoscedasticity_{factor}' for factor in factors])
    for factor in factors:
        levels = np.unique(data[factor])

        for variable in variables:
            normality_ok.loc[variable,f'normality_{factor}'] = True
            homoscedasticity_ok.loc[variable,f'homoscedasticity_{factor}'] = True
            df_all = []

            for level in levels:
                df = data[data[factor] == level]
                shapiro_test = shapiro(df[~df[variable].isna()][variable]) #Hay que chequear los supuestos para cada subconjunto por separado? 
                shapiro_results.loc[variable,f'statistic_Shapiro_{level}'] = round(shapiro_test[0],3)
                shapiro_results.loc[variable,f'p-value_Shapiro_{level}'] = round(shapiro_test[1],3)

                df_all.append(df[~df[variable].isna()][variable])
                
                if shapiro_test[1] < threshold_pval:
                    normality_ok.loc[variable,f'normality_{factor}'] = False

            if sum([shapiro_results.loc[variable,f'p-value_Shapiro_{level}'] < .05 for level in levels]) != 0:
                homoscedasticity_test = levene(*df_all) #El parámetro 'center' por default es 'median'. Lo cambiamos? Agregar test de Bartlett para datos normales.
                test = 'Levene'
            else:
                homoscedasticity_test = bartlett(*df_all) 
                test = 'Bartlett'
            
            homoscedasticity_results.loc[variable,f'statistics_homoscedastiticy_{factor}'] = round(homoscedasticity_test[0],3) 
            homoscedasticity_results.loc[variable,f'p-value_homoscedasticity_{factor}'] = round(homoscedasticity_test[1],3) 
            homoscedasticity_results.loc[variable,f'homoscedasticity_test_name'] = test

            if homoscedasticity_test[1] < threshold_pval:
                homoscedasticity_ok.loc[variable,f'homoscedasticity_{factor}'] = False
            
    return {'shapiro': shapiro_results,
        'homoscedasticity': homoscedasticity_results,
        'normality_ok': normality_ok,
        'homoscedasticity_ok': homoscedasticity_ok}

def get_summary(data,groups,interactions=True,variables=None):
    
    if variables == None:
        variables = [col for col in data.columns if isinstance(data[col][0],(int,float))]
    
    summary_groups = pd.DataFrame(index=variables)

    for variable in variables:
        for group in groups:
            elements = np.unique(data[group])

            for element in elements:
                summary_groups.loc[variable,f'N_{element}'] = sum(~data[data[group] == element][variable].isna())
                summary_groups.loc[variable,f'mean_{element}'] = round(np.nanmean(data[data[group] == element][variable]),3)
                summary_groups.loc[variable,f'std_{element}'] = round(np.nanstd(data[data[group] == element][variable]),3)

#TODO: Extenderlo para ¿todas? las combinaciones posibles de factores.
        if interactions == True:
            for element1, element2 in itertools.product(np.unique(data[groups[0]]),np.unique(data[groups[1]])):
                summary_groups.loc[variable,f'N_{element1}_{element2}'] = sum(~data[(data[groups[0]] == element1) & (data[groups[1]] == element2)][variable].isna())
                summary_groups.loc[variable,f'mean_{element1}_{element2}'] = round(np.nanmean(data[(data[groups[0]] == element1) & (data[groups[1]] == element2)][variable]),3)
                summary_groups.loc[variable,f'std_{element1}_{element2}'] = round(np.nanstd(data[(data[groups[0]] == element1) & (data[groups[1]] == element2)][variable]),3)

    return summary_groups

def stats_between_factor(data,factor,assumptions_anova,path_to_save,variables=None,correction_method='fdr_bh'):
    '''
        correction_method : str, optional
        Method for adjusting p values. See `statsmodels.sandbox.stats.multicomp`
        for details. Available methods are:
        'bonferroni' : one-step correction
        'sidak' : one-step correction
        'holm-sidak' : step-down method using Sidak adjustments
        'holm' : step-down method using Bonferroni adjustments
        'simes-hochberg' : step-up method  (independent)
        'hommel' : closed method based on Simes tests (non-negative)
        'fdr_bh' (DEFAULT): Benjamini/Hochberg  (non-negative)
        'fdr_by' : Benjamini/Yekutieli (negative)
        'fdr_tsbh' : two stage fdr correction (non-negative)
        'fdr_tsbky' : two stage fdr correction (non-negative)
    '''
    levels = np.unique(data[factor])

    if variables == None:
        variables = [col for col in data.columns if isinstance(data[col][0],(float,int))]
    
    df_all = []
    stat_results = pd.DataFrame(index=variables)
    
    for variable in variables:
        assumptions = assumptions_anova[variable]
        stat_results.loc[variable,'assumptions'] = assumptions
        stat_results.loc[variable,'groups'] = '-'.join([str(level) for level in levels]) #TODO: buscar alternativa para esto

        df_all = []

        for level in levels:
            df_all.append(data[data[factor] == level][variable])

        if assumptions:
            stats, p= f_oneway(*df_all)

            stat_results.loc[variable,f'main_effect_{factor}_stat'] = round(stats,3) 
            stat_results.loc[variable,f'main_effect_{factor}_p-value'] = round(p,3) 
            stat_results.loc[variable,'model'] = 'One-way ANOVA'
            
            if (p < .05) & (len(df_all)>2): #Conviene poner un threshold configurable?
                p_values = pairwise_tukeyhsd(endog=data[variable],groups=data[factor],alpha=.05) #TODO: Agregar otras opciones de post-hocs
                stat_results.loc[variable,'posthoc_p-values'] = '\n'.join([str(round(pval,3)) for pval in p_values.pvalues])
                stat_results.loc[variable,'post-hoc method'] = "Tukey's HSD"

        else:
            stats, p= kruskal(*df_all,nan_policy='omit')
            #stats, p = pg.kruskal(data=data[~data[variable].isna()],dv=variable,between=factor)
            stat_results.loc[variable,f'main_effect_{factor}_stat'] = round(stats,3) 
            stat_results.loc[variable,f'main_effect_{factor}_p-value'] = round(p,3) 
            stat_results.loc[variable,'model'] = 'Kruskal-Wallis'
            
            if (p < .05) & (len(df_all)>2): #Conviene poner un threshold configurable?
                p_values = sp.posthoc_dunn(df_all,p_adjust=correction_method) #TODO: Otras alternativas de post-hoc?
                stat_results.loc[variable,'posthoc_p-values'] = '--'.join([str(round(pval,3)) for pval in np.ravel(np.triu(p_values,1)) if pval!=0])
                stat_results.loc[variable,'post-hoc method'] = "Dunn's test"
    
        if p < .05:
            fig = plt.figure()
            ax = fig.add_subplot()

            sns.boxplot(x=factor,y=variable,data=data,ax=ax,color='#FF5733')
            ax.set_xlabel(factor)
            ax.set_ylabel(variable)
            
            Path(path_to_save,'Figures').mkdir(exist_ok=True)

            plt.savefig(Path(Path(path_to_save,'Figures'),f'boxplot_{variable}_{factor}.png'))
    return stat_results

def stats_within_factor(data,factor,subject,assumptions_anova,path_to_save,variables=None):
    
    levels = np.unique(data[factor])

    if variables == None:
        variables = [col for col in data.columns if isinstance(data[col][0],(float,int))]
    
    df_all = []
    stat_results = pd.DataFrame(index=variables)
    
    for variable in variables:
        reject_subjects = data.loc[data[variable].isna(),subject]
        data_clean = data[[subj not in reject_subjects.values for subj in data[subject]]]

        assumptions = assumptions_anova[variable]
        stat_results.loc[variable,'assumptions'] = assumptions
        stat_results.loc[variable,'levels'] = '-'.join([str(level) for level in levels])

        df_all = []

        for level in levels:
            df_all.append(data_clean[data_clean[factor] == level][variable])

        if assumptions:
            model = AnovaRM(data=data_clean,depvar=variable,subject=subject,within=[factor]).fit().anova_table

            stat_results.loc[variable,f'main_effect_{factor}_stat'] = round(model.iloc[0,0],3) #TODO: Checkear estos índices
            stat_results.loc[variable,f'main_effect_{factor}_p-value'] = round(model.iloc[0,3],3) 
            stat_results.loc[variable,'method'] = 'Repeated measures ANOVA'
            
            p = model.iloc[0,3]

            if np.isnan(p):
                print(data_clean)

            if (p < .05) & (len(df_all)>2):
                p_values = pairwise_tukeyhsd(endog=data_clean[variable],groups=data_clean[factor],alpha=.05) #TODO: Acá también se usa Tukey?
                stat_results.loc[variable,'posthoc_p-values'] = '--'.join([str(round(pval,3)) for pval in p_values.pvalues])
                stat_results.loc[variable,'post-hoc method'] = "Tukey's HSD"

        else:

            if len(df_all)>2:
                stats, p= friedmanchisquare(*df_all)
                test = 'Friedman'
            else:  
                stats, p= wilcoxon(*df_all)
                test = 'Wilcoxon'

            stat_results.loc[variable,f'main_effect_{factor}_stat'] = round(stats,3)
            stat_results.loc[variable,f'main_effect_{factor}_p-value'] = round(p,3) 
            stat_results.loc[variable,'method'] = test
            
            if (p < .05) & (len(df_all)>2):
                p_values = sp.posthoc_nemenyi_friedman(np.array(df_all)) #TODO: Checkear alternativas.
                stat_results.loc[variable,'posthoc_p-values'] = '--'.join([str(round(pval,3)) for pval in np.ravel(np.triu(p_values,1)) if pval!=0])
                stat_results.loc[variable,'post-hoc method'] = "Nemenyi"
        
        if p < .05:
            fig = plt.figure()

            ax = fig.add_subplot()
            sns.boxplot(x=factor,y=variable,data=data,ax=ax,color='#FF5733')
            ax.set_xlabel(factor)
            ax.set_ylabel(variable)
            
            fig_dir = Path(path_to_save,'Figures')
            fig_dir.mkdir(exist_ok=True)

            plt.savefig(Path(fig_dir,f'boxplot_{variable}_{factor}.png'))
    
    return stat_results

def two_by_two_ANOVA(data,within,between,subject,path_to_save,correction=True,variables=None):
    if variables == None:
        variables = [col for col in data.columns if isinstance(data[col][0],(int,float))]
    
    mixed_anova = pd.DataFrame(index=variables)

    if (f'{between}_{within}' not in data.columns) & (f'{within}_{between}' not in data.columns):
        for i, row in data.iterrows():
            data.loc[i,f'{between}_{within}'] = f'{row[between]}_{row[within]}'
        
        interaction_factor = f'{between}_{within}'
    
    else:
        interaction_factor = f'{between}_{within}' if f'{between}_{within}' in data.columns else f'{within}_{between}'

    for variable in variables:
        results = pg.mixed_anova(data=data,dv=variable,within=within,between=between,subject=subject,correction=correction)  #TODO: Explorar mejor el parámetro 'correction'
        mixed_anova.loc[variable,'method'] = 'Mixed effects ANOVA'
        for factor in [between,within,'Interaction']:
            mixed_anova.loc[variable,f'{factor}_stat'] = round(results[results['Source'] == factor]['F'].values[0],3)
            mixed_anova.loc[variable,f'{factor}_p-value'] = round(results[results['Source'] == factor]['p-unc'].values[0],3)
            mixed_anova.loc[variable,f'{factor}_np2'] = round(results[results['Source'] == factor]['np2'].values[0],3)

            if (mixed_anova.loc[variable,f'{factor}_p-value'] < .05) & (factor != 'Interaction'):
                fig = plt.figure()

                ax = fig.add_subplot()
                
                sns.boxplot(x=factor,y=variable,data=data,ax=ax,color='#FF5733')

                ax.set_xlabel(factor)
                ax.set_ylabel(variable)
                
                fig_dir = Path(path_to_save,'Figures')
                fig_dir.mkdir(exist_ok=True)
            
                plt.savefig(Path(fig_dir,f'boxplot_{variable}_{factor}.png'))
    
        # if results[results['Source']=='Interaction']['p-unc'].values[0]< .05:

        p_values = pairwise_tukeyhsd(endog=data[variable],groups=data[interaction_factor],alpha=.05)
        mixed_anova.loc[variable,'posthoc_interactions'] = '\n'.join([f'{group1}_vs_{group2}' for (group1,group2) in itertools.combinations(p_values.groupsunique,r=2)])
        mixed_anova.loc[variable,'posthoc_p-values'] = '\n'.join([str(round(pval,3)) for pval in p_values.pvalues])
        mixed_anova.loc[variable,'post-hoc method'] = "Tukey's HSD"

        fig = plt.figure()

        ax = fig.add_subplot()

        sns.boxplot(x=between,y=variable,hue=within,data=data)
        ax.set_xlabel(between)
        ax.set_ylabel(variable)
            
        fig_dir = Path(path_to_save,'Figures')
        fig_dir.mkdir(exist_ok=True)

        plt.savefig(Path(fig_dir,f'boxplot_{variable}_{factor}.png'))  

    return mixed_anova

def two_by_two_ANOVA_dictionary(data,within,between,subject,path_to_save,correction=True,variables=None):
    if variables == None:
        variables = [col for col in data.columns if isinstance(data[col][0],(int,float))]
    
    mixed_anova = {}
    mixed_anova["ANOVA"] = pd.DataFrame(index=variables)
    mixed_anova["posthoc"] = None

    if (f'{between}_{within}' not in data.columns) & (f'{within}_{between}' not in data.columns):
        for i, row in data.iterrows():
            data.loc[i,f'{between}_{within}'] = f'{row[between]}_{row[within]}'
        
        interaction_factor = f'{between}_{within}'
    
    else:
        interaction_factor = f'{between}_{within}' if f'{between}_{within}' in data.columns else f'{within}_{between}'

    for variable in variables:
        # Elimino sujetos que tienen nan en alguna de sus within variables
        reject_subjects = data.loc[data[variable].isna(),subject]
        data_clean = data[[subj not in reject_subjects.values for subj in data[subject]]]
        
        results = pg.mixed_anova(data=data_clean,dv=variable,within=within,between=between,subject=subject,correction=correction)  #TODO: Explorar mejor el parámetro 'correction'
        mixed_anova["ANOVA"].loc[variable,'method'] = 'Mixed effects ANOVA'
        for factor in [between,within,'Interaction']:
            mixed_anova["ANOVA"].loc[variable,f'{factor}_stat'] = round(results[results['Source'] == factor]['F'].values[0],3)
            mixed_anova["ANOVA"].loc[variable,f'{factor}_p-value'] = round(results[results['Source'] == factor]['p-unc'].values[0],3)
            mixed_anova["ANOVA"].loc[variable,f'{factor}_np2'] = round(results[results['Source'] == factor]['np2'].values[0],3)

            if (mixed_anova["ANOVA"].loc[variable,f'{factor}_p-value'] < .05) & (factor != 'Interaction'):
                fig = plt.figure()

                ax = fig.add_subplot()
                
                sns.boxplot(x=factor,y=variable,data=data,ax=ax,color='#FF5733')

                ax.set_xlabel(factor)
                ax.set_ylabel(variable)
                
                fig_dir = Path(path_to_save,'Figures')
                fig_dir.mkdir(exist_ok=True)
            
                plt.savefig(Path(fig_dir,f'boxplot_{variable}_{factor}.png'))
    
        # if results[results['Source']=='Interaction']['p-unc'].values[0]< .05:


        p_values = pairwise_tukeyhsd(endog=data_clean[variable],groups=data_clean[interaction_factor],alpha=.05)
        # posthocs = pg.pairwise_ttests(dv=variable, within=within, subject=subject,
        #                       between=between, padjust='bonf', data=data_clean,effsize="cohen", interaction=True)
        # Armo los grupos al revés por como lo devuelve el pairwise_tukeyhsd
        mixed_anova["posthoc"] = pd.DataFrame(index=[f'{group2}_vs_{group1}' for (group1,group2) in itertools.combinations(p_values.groupsunique,r=2)])

        mixed_anova["posthoc"]['p-values'] = [str(round(pval,3)) for pval in p_values.pvalues]
        mixed_anova["posthoc"]['mean_differences'] = [str(round(pval,3)) for pval in p_values.meandiffs]
        mixed_anova["posthoc"]['CI_left'] = [str(round(CI_left[0],3)) for CI_left in p_values.confint]
        mixed_anova["posthoc"]['CI_right'] = [str(round(CI_right[1],3)) for CI_right in p_values.confint]
        mixed_anova["posthoc"]["df"] = [(len(data_clean[data_clean["Grupo_fluencia"] == group1]) + len(data_clean[data_clean["Grupo_fluencia"] == group2]) - 2) for (group1,group2) in itertools.combinations(p_values.groupsunique,r=2)]
        mixed_anova["posthoc"]["cohen_d"] = [pg.compute_effsize(data_clean[data_clean["Grupo_fluencia"] == group2][variable], data_clean[data_clean["Grupo_fluencia"] == group1][variable], paired=False, eftype='cohen') for (group1,group2) in itertools.combinations(p_values.groupsunique,r=2)]
        # mixed_anova["posthoc"]["t_ratio"] = [pg.pairwise_ttests(data[data["Grupo_fluencia"] == group1][variable], data[data["Grupo_fluencia"] == group2][variable], paired=False, eftype='cohen') for (group1,group2) in itertools.combinations(p_values.groupsunique,r=2)]


        # for (group1,group2) in itertools.combinations(p_values.groupsunique,r=2):
            
        mixed_anova["posthoc"]['method'] = "Tukey's HSD"

        fig = plt.figure()

        ax = fig.add_subplot()

        sns.boxplot(x=between,y=variable,hue=within,data=data)
        ax.set_xlabel(between)
        ax.set_ylabel(variable)
            
        fig_dir = Path(path_to_save,'Figures')
        fig_dir.mkdir(exist_ok=True)

        plt.savefig(Path(fig_dir,f'boxplot_{variable}_{factor}.png'))  

    return mixed_anova