# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:53:23 2021

@author: franc
"""

# Graficar.py>

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import os
import matplotlib as mpl

def plot_pointplot_errorbar_classificacion_metrics(dict_df,k_folds,metric,nombre_ejecucion,n_repeats,cwd,show_figures = False, width = 20, height = 15):
    mpl.rcParams['font.family'] = 'Arial'
    legend_properties = {'weight':'bold','size':15}


    for key_diseases, value_diseases in dict_df.items():
        fig, axs = plt.subplots(ncols=len(k_folds), figsize=(width,height))
        if len(k_folds)>1:
            for k,k_fold in enumerate(k_folds):
                data_aux = value_diseases[value_diseases["k-fold"] == str(k_folds[0])]
                g = sns.pointplot(x="Base", y=metric, hue="Clasificador", errorbar="sd", data=data_aux, ax=axs)
                g.set_xticklabels(g.get_xticklabels(), rotation=90)
                g.set_yticklabels(g.get_yticklabels(), weight='bold')
                axs.set_title(str(k_folds[0]) + " k-folds")
                axs.tick_params(axis='both', which='major', labelsize=20)
                axs.set_xlabel("Base", fontsize=32, weight='bold')
                axs.set_ylabel(metric, fontsize=32, weight='bold')
                plt.setp(axs.get_legend().get_texts(), fontname='arial', fontweight="bold")

        else:
            data_aux = value_diseases[value_diseases["k-fold"] == str(k_folds[0])]
            g = sns.pointplot(x="Base", y=metric, hue="Clasificador",errorbar="sd",data=data_aux, ax=axs)
            g.set_xticklabels(g.get_xticklabels(),rotation=90)
            axs.set_title(str(k_folds[0])+ " k-folds")
            axs.tick_params(axis='both', which='major', labelsize=20)
            axs.set_xticklabels(axs.get_xticklabels(), weight='bold')
            axs.set_yticklabels(axs.get_yticklabels(), weight='bold')
            plt.setp(axs.get_legend().get_texts(), fontname='arial',fontweight="bold") 

            axs.set_xlabel("Base",fontsize=32, weight='bold')
            axs.set_ylabel(metric,fontsize=32, weight='bold')
        fig.suptitle(key_diseases) 
        # plt.xlabel('Decision scores', fontdict=font_axis_labels)
        # plt.ylabel('Density',fontsize=32, fontdict=font_axis_labels)
        plt.legend(prop=legend_properties)
        plt.savefig(cwd+"/"+nombre_ejecucion+"_imagen_machine_learning_"+metric+"_points_"+key_diseases+"_"+str(n_repeats)+".png",\
                    bbox_inches='tight')
        if show_figures:
            plt.show()
        else:
            plt.close('all')

def plot_multiple_barras_vertical_from_dataframe(df,columnas,labels_x, label_y, label_legend, title,  width, formato=0, padding = 0, fontsize = 8.5, pie_figura = "",fig_text_space = -0.2,rotation = 0,legend_position=0):

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
    ax.legend(loc=legend_position)
    
    for i,recta in enumerate(rectas):
        if formato==0:
            ax.bar_label(recta, padding=padding, fontsize=fontsize)
        else:
            ax.bar_label(recta, padding=padding, fontsize=fontsize, fmt=formato[i])
     
    fig.tight_layout()
    
    if (pie_figura != ""):
        plt.figtext(0.5, fig_text_space, pie_figura, wrap=True, horizontalalignment='center', fontsize=12)
    
    plt.xticks(rotation = rotation)
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
    
def radar_plot(df, values, groups, features, title, path, file_name="radar_plot.png", axis_range = [0.4,0.9], title_x=0.51, legend_x = 0.7, legend_y = 1.01, show = True,group_color=None ):
    
    fig = px.line_polar(data_frame=df,
                        r=values,
                        theta=features,
                        line_close=True,
                        color=groups,
                        range_r = axis_range,
                        color_discrete_map=group_color)
        
    fig.update_traces(fill='toself')
    fig.update_layout(
        legend=dict(
            x=legend_x,
            y=legend_y,
            traceorder="normal",
            bgcolor= 'rgba(0,0,0,0)',
            font=dict(
                family="sans-serif",
                size=14,
                color="black"
            ),
        ),
        title_text=title, title_x=title_x,
        legend_title_text=''
    )
    if show:
        fig.show()
    fig.write_image(path+"//"+file_name)

def roc_curve(df,legends,auc_means,auc_stds,xs,ys,colors,titulo,path,filename,width=1200,height=600,title_x=0.41):
    
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash',color='black'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    for i_legend,legend in enumerate(legends):
        fig.add_traces(go.Scatter(name=legend+" (AUC = " + "{:.2f}".format(auc_means[i_legend]) + "±" + "{:.2f}".format(auc_stds[i_legend])+")",\
                                  x=df[xs[i_legend]],y=df[ys[i_legend]], mode = 'lines', line=dict(color=colors[i_legend])))
                                  
    fig['data'][0]['showlegend']=True
    layout = go.Layout(
    autosize=False,
    width=width,
    height=height,
    title_text=titulo, title_x=title_x,
    legend=dict(
        traceorder="normal",
        bgcolor= 'rgba(0,0,0,0)',
        font=dict(
            family="sans-serif",
            size=14,
            color="black"
        ),
    ),
    xaxis=dict(
        title="False Positive Rate",
        title_font = {"size": 20},
        range=[0, 1]
    ),
    yaxis=dict(
        title="True Positive Rate",
        title_font = {"size": 20},
        range=[0, 1]
    ) ) 

    fig.update_layout(layout)
    # fig.show()
    fig.write_image(path+"//"+filename)


def graphicDistributions(scores_o, y_labels_o, bins, positiveClass = 'Positive', negativeClass = 'Negative', colorNegative= 'White', colorPositive = 'Black',path=None, width = 9.5, height = 7.5):
    plt.figure(figsize=(width,height))
    
    # formato para los labels
    font_axis_labels = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 32,
        }

    # barras y líneas de la clase negativa
    ax1 = sns.distplot(scores_o[y_labels_o==0], label=negativeClass,
                 hist_kws={'edgecolor':'black','color':colorNegative, "alpha": 0.7},
                 kde_kws={"color": colorNegative, "linestyle":'--',"linewidth":5}, bins = bins)
    # barras y líneas de la clase positiva
    ax2 = sns.distplot(scores_o[y_labels_o==1], label=positiveClass, color=colorPositive,
                 hist_kws={'edgecolor':'black','color':colorPositive, "alpha": 0.7},
                 kde_kws={"color": colorPositive,"linewidth":5}, bins = bins)
    
    plt.xlabel('Decision scores', fontdict=font_axis_labels)
    plt.ylabel('Density',fontsize=32, fontdict=font_axis_labels)
    plt.legend(fontsize=17)
    plt.setp(ax1.get_legend().get_texts(), fontname='arial',fontweight="bold") 
    plt.setp(ax2.get_legend().get_texts(), fontname='arial',fontweight="bold") 

    # Pongo en negrita los ticks
    for element in ax1.get_xticklabels():
        element.set_fontweight("bold")
    for element in ax1.get_yticklabels():
        element.set_fontweight("bold")
    for element in ax2.get_xticklabels():
        element.set_fontweight("bold")
    for element in ax2.get_yticklabels():
        element.set_fontweight("bold")
    plt.xticks(fontsize=20,fontname = "arial")
    plt.yticks(fontsize=20,fontname = "arial")
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.savefig(path+"distribution.eps")
    plt.figure(figsize=(width,height))
    
    
def plot_confusion_matrix(row, key_group_colors, disease, n_repeats, nombre_ejecucion, show_figures=True, width = 9.5, height = 7.5):

    mpl.rcParams['font.family'] = 'Arial'

    palette = sns.color_palette("light:"+key_group_colors[disease[0]+"-"+disease[1]], as_cmap=True)

    fig=plt.figure(figsize=(width,height))
    cm_total=[[row["true_neg"]/(row["true_neg"]+row["false_pos"]),row["false_pos"]/(row["true_neg"]+row["false_pos"])],[row["false_neg"]/(row["false_neg"]+row["true_pos"]),row["true_pos"]/(row["false_neg"]+row["true_pos"])]]
    ax = sns.heatmap(cm_total, annot=True, linewidths=.5, square = True, cmap=palette, yticklabels=disease, xticklabels=disease, cbar=False, fmt='.2f', annot_kws={'fontsize': 20})
    
    # Set font size and weight for x and y tick labels
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')

    # Set font size and weight for axis labels
    ax.set_xlabel('Predicted label', fontsize=32, weight='bold')
    ax.set_ylabel('Actual label', fontsize=32, weight='bold')
    
    # Remove the title
    ax.set_title("")
    
    plt.title("");

    # Create directory for saving images if it doesn't exist
    cwd = os.getcwd()
    save_path = cwd + "/imagenes_matriz_confusion_" + row["Clasificador"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(save_path + "/" + row["Base"] + "_" + nombre_ejecucion + "_" + row["Grupo"] + "_" + row["k-fold"] + "_" + str(n_repeats) + '.png', dpi=300)
    
    if show_figures:
        plt.show()
    else:
        plt.close('all')
