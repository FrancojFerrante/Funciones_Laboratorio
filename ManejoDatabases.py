# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:34:51 2022

@author: fferrante
"""
# ManejoDatabases.py>

from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
import pandas as pd

def colapso_bins_granularidad(df, cols_totales, cols_colapsadas):
    for columna in df.columns:
        for i,col_total in enumerate(cols_totales):
            if col_total in columna:
                str_columna_sin_bin = columna[0:-(len(col_total))]
                for col_colapsada in cols_colapsadas[i]:
                    df[columna] = df[str_columna_sin_bin+col_total].values + \
                        df[str_columna_sin_bin+col_colapsada].values
                    df = df.drop(str_columna_sin_bin+col_colapsada, 1)
                    
    return df

def agrupado_bins_granularidad(df, cols_segmentos, cols_colapsadas, bin_generico = "bin_7"):
    for columna in df.columns:
        if bin_generico in columna:
            str_columna_sin_bin = columna[0:-len(bin_generico)]
            for i,col_segmento in enumerate(cols_segmentos):
                df[str_columna_sin_bin+col_segmento] = df[str_columna_sin_bin+cols_colapsadas[i][0]]
                for col_colapsada in cols_colapsadas[i][1:]:
                    df[str_columna_sin_bin+col_segmento] = df[str_columna_sin_bin+col_segmento].values + \
                        df[str_columna_sin_bin+col_colapsada].values
    return df

def emparejamiento_estadistico_parejo(df,column_group,variables,tipos,min_p_value):
    
    def calcular_resultado(lista):
        multiplicacion = 1
        suma = sum(lista)
        cantidad_elementos = len(lista)
    
        for elemento in lista:
            multiplicacion *= elemento
    
        resultado = (multiplicacion / suma) * cantidad_elementos
        return resultado

    # Elimino filas que tengan nan en alguna variable
    for variable in variables:
        df = df[df[variable].notna()]

    grupos = df[column_group]

    p_values = []
    for variable,tipo in zip(variables,tipos):
        if tipo == "continua":
            p_values.append(f_oneway(*[df[variable][grupos == g] for g in grupos.unique()])[1])
        if tipo == "categorica":
            tabla_contingencia = pd.crosstab(grupos, df[variable])
            _, p_value_categorica, _, _ = chi2_contingency(tabla_contingencia)
            p_values.append(p_value_categorica)



    iteracion = 0
    # df_emparejado = pd.DataFrame()
    while any(valor < min_p_value for valor in p_values):
        # p_value_age = f_oneway(df["Age"].values)[1]
        # p_value_education = f_oneway(df["Education"].values)[1]
        # _, p_value_sex, _, _ = chi2_contingency(tabla_contingencia)
        print(str(iteracion) + ")")

        grupo_mayor = df[column_group].value_counts().idxmax()
        #df_min = df[df["GROUP"] == valor_mas_frecuente]
        
        
        puntaje_p = 0
        puntaje_p_mayor = 0
        participante_mayor = 0
        # Crear un bucle for para iterar sobre cada participante del grupo mayor
        for participante in df[df[column_group] == grupo_mayor].index:
            # Eliminar el participante del DataFrame copia
            df_copia = df.drop(participante)
            
            grupos = df_copia[column_group]
            
            p_values = []
            for variable,tipo in zip(variables,tipos):
                if tipo == "continua":
                    p_values.append(f_oneway(*[df_copia[variable][grupos == g] for g in grupos.unique()])[1])
                if tipo == "categorica":
                    tabla_contingencia = pd.crosstab(grupos, df_copia[variable])
                    _, p_value_categorica, _, _ = chi2_contingency(tabla_contingencia)
                    p_values.append(p_value_categorica)
            
            puntaje_p = calcular_resultado(p_values)

            if puntaje_p > puntaje_p_mayor:
                puntaje_p_mayor = puntaje_p
                participante_mayor = participante
               
        
        df = df.drop(participante_mayor)
        iteracion += 1
        grupos = df[column_group]

        p_values = []
        for variable,tipo in zip(variables,tipos):
            if tipo == "continua":
                p_values.append(f_oneway(*[df[variable][grupos == g] for g in grupos.unique()])[1])
            if tipo == "categorica":
                tabla_contingencia = pd.crosstab(grupos, df[variable])
                _, p_value_categorica, _, _ = chi2_contingency(tabla_contingencia)
                p_values.append(p_value_categorica)

    return df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        