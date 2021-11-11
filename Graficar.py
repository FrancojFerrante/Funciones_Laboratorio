# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:53:23 2021

@author: franc
"""

# Graficar.py>


import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt



def plot_multiple_barras_vertical_from_dataframe(df,columnas,labels_x, label_y, label_legend, title,  width, formato=0, padding = 0, fontsize = 8.5, pie_figura = ""):

    valores_a_plotear = []
    for columna in columnas:
        valores_a_plotear.append(df[columna].tolist())
    
    x = np.arange(len(labels_x))  # the label locations
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    
    if (len(valores_a_plotear)%2 == 1):    
        start = (-width)*int(len(valores_a_plotear)/2)
        stop = ((+width)*int(len(valores_a_plotear)/2))+0.0001
        widths = np.arange(start,stop,width)
    else:
        start = (-width)*int(len(valores_a_plotear)/2)+(width/2)
        stop = (((+width)*int(len(valores_a_plotear)/2))-(width/2))+0.0001
        widths = np.arange(start,stop,width)
    
    rectas = []
    for i, valor in enumerate(valores_a_plotear):
        rectas.append(ax.bar(x + widths[i], valor, width, label=label_legend[i]))
        
     # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(label_y)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.legend()
    
    for i,recta in enumerate(rectas):
        if formato==0:
            ax.bar_label(recta, padding=padding, fontsize=fontsize)
        else:
            ax.bar_label(recta, padding=padding, fontsize=fontsize, fmt=formato[i])
     
    fig.tight_layout()
    
    if (pie_figura != ""):
        plt.figtext(0.5, -0.2, pie_figura, wrap=True, horizontalalignment='center', fontsize=12)
        
    plt.show()



def plot_multiple_barras_vertical_from_lists(labels_x, label_y, label_legend, title, width, formato = 0, padding = 0, fontsize =8.5, *args):

    n_args = len(args)
    
    x = np.arange(len(labels_x))  # the label locations
    
    fig, ax = plt.subplots()
    
    
    if (n_args%2 == 1):    
        start = (-width)*int(n_args/2)
        stop = ((+width)*int(n_args/2))+0.0001
        widths = np.arange(start,stop,width)
    else:
        start = (-width)*int(n_args/2)+(width/2)
        stop = (((+width)*int(n_args/2))-(width/2))+0.0001
        widths = np.arange(start,stop,width)
    
    rectas = []
    for i, valor in enumerate(args):
        rectas.append(ax.bar(x + widths[i], valor, width, label=label_legend[i]))
        
     # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(label_y)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.legend()
    
    
    for i,recta in enumerate(rectas):
        if formato==0:
            ax.bar_label(recta, padding=padding, fontsize=fontsize)
        else:
            ax.bar_label(recta, padding=padding, fontsize=fontsize, fmt=formato[i])
     
    fig.tight_layout()
    
    plt.show()