# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:50:40 2021

@author: franc
"""
# Estadistica.py>
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd

def chi_cuadrado_transcripciones(df_list, columnas, categoria, posibilidades):
    
    if (len(df_list)<=0):
        print("No es posible realizar el análisis. No se han pasado dataframes como argumentos")
        return
    # Me quedo con las filas que tienen transcripciones
    df_transcripciones = []
    if ('has_Transcription' in df_list[0].columns):
        for df in df_list:
            df_transcripciones.append(df[df['has_Transcription']==1][columnas])
        
    
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
