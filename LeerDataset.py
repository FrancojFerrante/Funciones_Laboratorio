# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 18:48:11 2021

@author: franc
"""

# LeerDataset.py>

import pandas as pd
from os import listdir


def read_csv_to_dataframe(path, page = "", header = 0):
    """
    Return a dataframe with the Excel content

    Parameters
    ----------
    path : string
        The path where the document is.
    page : string, optional
        Name of the page to read. The default is "".
    header : int, optional
        Amount of rows to skip. The default is 0.

    Returns
    -------
    df : pandas.dataframe
        The Excel content.

    """
    
    df = pd.ExcelFile(path)
    if page=="":
        return df
    else:
        df = pd.read_excel(df,page,header=header)
        return df

def file_names_to_dataframe(path,column_name):
    """
    Read all directory's files and return a dataframe with its names

    Parameters
    ----------
    path : string
        The path where the files are.
    column_name : string
        The dataframe's column name.

    Returns
    -------
    df : pandas.dataframe
        A dataframe with a single column with all the files names as rows.

    """
    archivos = []
    for filename in listdir(path):
        newFileName = filename[:-4]
        archivos.append(newFileName)
    archivos = list(dict.fromkeys(archivos))
    df = pd.DataFrame(archivos, columns=[column_name])
    return df