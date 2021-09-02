# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:25:20 2021

@author: franc
"""

# MergeFunctions.py>

import numpy as np
import pandas as pd


def merge_dfs_by_id_conserve_nan(df1,df2,id1,id2):
    """
    It merges rows that matches in id1 and id2, adds columns from both dataframes,
    and I also keep the rows that are Nan in both dataframe's ids

    Parameters
    ----------
    df1 : pandas.dataframe
        first dataframe to merge.
    df2 : pandas.dataframe
        second dataframe to merge.
    id1 : string
        Column name to merge of first dataframe.
    id2 : string
        Column name to merge of second dataframe.

    Returns
    -------
    total : pandas.dataframe
        a dataframe with the number of columns equal to df1's column's length + 
        df2's column's length and the merged rows.

    """
    # Me quedon con las filas que coinciden que no son Nan
    total = pd.merge(df1[df1[id1].notna()],df2[df2[id2].notna()], how='outer',
                      left_on=[id1],right_on=[id2])
    
    # Me quedo con las que son nan de la izquierda
    aux = pd.merge(df1,df2[df2[id2].notna()], how='left',
             left_on=[id1],right_on=[id2])
    aux = aux[pd.isnull(aux[id1])]
    
    # Lo uno a total
    total = total.append(aux,ignore_index=True)
    
    # Me quedo con las que son nan de la derecha
    aux = pd.merge(df1[df1[id1].notna()],df2, how='right',
             left_on=[id1],right_on=[id2])
    aux = aux[pd.isnull(aux[id2])]
    
        # Lo uno a total
    total = total.append(aux,ignore_index=True)
    
    # Agrupo las columnas que tenian el mismo nombre
    total = agruparColumnasMismoNombre(df1,df2,total,'_x','_y')
    
    return total

def agruparColumnasMismoNombre(df1,df2,dfTotal, suffix1, suffix2):
    
    # Me quedo con el nombre de las columnas
    columnas1 = df1.columns
    columnas2 = df2.columns
    
    # Encuentro las columnas que son iguales
    columnasComun = columnas1.intersection(columnas2)
    
    for i in columnasComun:
        dfTotal.update(dfTotal[[i+suffix1, *[i+suffix2]]].bfill(axis=1)[i+suffix1])  # in-place
        dfTotal = dfTotal.drop(i+suffix2, axis=1)
        dfTotal = dfTotal.rename({i+suffix1: i}, axis=1)  # new method

   
    return dfTotal


def merge_columns_conserve_non_nan(df, to_drop, to_update):
    """
    Merge two columns from df, conserves the cell that is non nan and deletes 
    the to_drop column. In case that both cells are non nan, keep first.

    Parameters
    ----------
    df : pandas.dataframe
        Dataframe to update.
    to_drop : string
        column to drop.
    to_update : string
        column to update.

    Returns
    -------
    df : pandas.dataframe
        return the dataframe with the to_drop column deleted and the to_update
    column updated.

    """
    
    to_drop = [to_drop]
    df.update(df[[to_update, *to_drop]].bfill(axis=1)[to_update])  # in-place
    df = df.drop(to_drop, axis=1)                           # not in-place, need to assign
    return df

def merge_dfs_by_id_conserve_nan_and_update_column(df1, df2, column_to_drop, column_to_update):
    df2 = merge_dfs_by_id_conserve_nan(df2, df1, column_to_drop, column_to_update)
    df2 = merge_columns_conserve_non_nan(df2, column_to_drop, column_to_update)
    return df2