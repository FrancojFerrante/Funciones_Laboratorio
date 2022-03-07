# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 18:48:11 2021

@author: franc
"""

# LeerDataset.py>

import pandas as pd
from os import listdir

import xlsxwriter

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

def anova_to_excel(df,path,filename):
    
    # Open an Excel workbook
    if (filename[-5:] == ".xlsx"):
        workbook = xlsxwriter.Workbook(path+"//"+filename)
    else:
        workbook = xlsxwriter.Workbook(path+"//"+filename+".xlsx")

    # Set up a format
    book_format = workbook.add_format(properties={'bold': True, 'font_color': 'red'})

    # Create a sheet
    worksheet = workbook.add_worksheet('dict_data')

    # Write the headers
    for col_num, header in enumerate(df.columns):
        worksheet.write(0,col_num, header)

    # Save the data from the OrderedDict into the excel sheet
    for row_num,row_data in df.iterrows():
        for col_num, cell_data in enumerate(row_data):
            if row_data["p-unc_y"] <0.05:
                if(str(cell_data)=="nan"):
                    worksheet.write(row_num+1, col_num, "", book_format)
                else:
                    worksheet.write(row_num+1, col_num, cell_data, book_format)
            else:
                if(str(cell_data)=="nan"):
                    worksheet.write(row_num+1, col_num, "")
                else:
                    worksheet.write(row_num+1, col_num, cell_data)
    # Close the workbook
    workbook.close()