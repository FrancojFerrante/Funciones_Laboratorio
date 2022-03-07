# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:34:51 2022

@author: fferrante
"""
# ManejoDatabases.py>

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
