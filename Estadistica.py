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

def anova_mixto_2x2(resultado,df_controles,df_no_controles,feature,columna_id,columna_grupo,columna_feature_1,columna_feature_2,txt_no_control):
    """
    Calcula anova mixto 2x2 para los dos grupos y las dos features pasadas. Agrega el resultado del anova al dataframe resultado sumando 
    una columna que identifique a la feature, otra a los grupos comparados y dos que indiquen el n para el grupo control y para el no control.

    Parameters
    ----------
    resultado : pandas.DataFrame
        Dataframe with columns ["Grupo","Prueba","Source","p-unc","np2","n_ctr","n_no_ctr"].
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
        resultado = resultado.append(df_resultado[["Grupo","Prueba","Source","p-unc","np2","n_ctr","n_no_ctr"]])
    return resultado

def anova_mixto_2x2_con_y_sin_outliers(columnas,prueba_1,prueba_2,df_controles,dfs_no_control,texto_no_control,path_figure="", save_boxplot = True):
    
    resultado = pd.DataFrame(columns=["Grupo","Prueba","Source","p-unc","np2","n_ctr","n_no_ctr"])
    resultado_sin_outliers = pd.DataFrame(columns=["Grupo","Prueba","Source","p-unc","np2","n_ctr","n_no_ctr"])
    
    for columna in columnas:
        separado = columna.split("_")
        col_fonologica = separado[0]+"_"+prueba_1+"_"+"_".join(separado[1:])
        col_semantica = separado[0]+"_"+prueba_2+"_"+"_".join(separado[1:])

        # Elimino outliers para controles
        df_ctr_sin_outliers = remove_outlier_IQR(df_controles[["Codigo","Grupo",col_fonologica,col_semantica]],\
                                                     col_fonologica,col_semantica)
            
        df_combinado_plot = df_ctr_sin_outliers
        for i_no_control,df_no_control in enumerate(dfs_no_control):
            resultado = anova_mixto_2x2(resultado,df_controles,df_no_control,columna,"Codigo","Grupo",\
                                   col_fonologica,col_semantica,texto_no_control[i_no_control])
            
        # Elimino outliers para no controles
            df_no_control_sin_outliers = remove_outlier_IQR(df_no_control[["Codigo","Grupo",col_fonologica,col_semantica]],\
                                                                col_fonologica,col_semantica)
          
            resultado_sin_outliers = anova_mixto_2x2(resultado_sin_outliers,df_ctr_sin_outliers,df_no_control_sin_outliers,columna,"Codigo","Grupo",\
                                   col_fonologica,col_semantica,texto_no_control[i_no_control])

            df_combinado_plot = df_combinado_plot.append(df_no_control_sin_outliers)
        
        if save_boxplot:
            df_combinado_plot = pd.melt(df_combinado_plot.reset_index(), id_vars=["Codigo",'Grupo'], value_vars=[col_fonologica, col_semantica])
            df_combinado_plot.rename(columns={"variable": "fluencia", "value": columna}, inplace=True)
            df_combinado_plot["fluencia"].replace({col_fonologica: "Fluencia fonológica", col_semantica: "Fluencia semántica"}, inplace=True)
    
    
            fig=plt.figure(figsize=(2,2));
            sns.boxplot(x='Grupo', y=columna, hue='fluencia', data=df_combinado_plot)
            plt.savefig(path_figure+"//imagenes_boxplots/"+columna + '.png')
            plt.close(fig)
    
    return resultado, resultado_sin_outliers