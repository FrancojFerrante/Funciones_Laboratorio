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
                df_acomodado.rename(columns={"variable": "fluencia", "value": "log_frq_promedio"}, inplace=True)
                df_acomodado.to_csv(path_files+"//databases_with_outliers_interaccion_significativos_single_column/" + columna+"_CTR-"+texto_no_control[i_no_control]+".csv")
                df_combinado.to_csv(path_files+"//databases_with_outliers_interaccion_significativos/" + columna+"_CTR-"+texto_no_control[i_no_control]+".csv")
            
            # Check if save excel without outliers
            if save_excel_without_outliers & (len(resultado_sin_outliers[(resultado_sin_outliers["p-unc"]<0.05) & (resultado_sin_outliers["Source"] == "Interaction")]) > 0):
                df_combinado = df_ctr_sin_outliers.append(df_no_control_sin_outliers)
                df_acomodado = pd.melt(df_combinado.reset_index(), id_vars=["Codigo",'Grupo'], value_vars=[col_fonologica, col_semantica])
                df_acomodado.rename(columns={"variable": "fluencia", "value": "log_frq_promedio"}, inplace=True)
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