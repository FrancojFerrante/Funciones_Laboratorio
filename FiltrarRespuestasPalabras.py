# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 18:40:50 2021

@author: franc
"""
import pandas as pd
import matplotlib as plt
import seaborn as sb
import os
import numpy as np

from MergeFunctions import *
from os import listdir
from os.path import isfile, join

# %% Funciones
def duplicadosMismaLista(lista):
    
    contDup = 0
    NoDuplicados = []
    for i in lista:
        if i not in NoDuplicados:
            NoDuplicados.append(i)
        else:
            contDup = contDup+1
    
    return NoDuplicados,contDup

def duplicadosDistintasListas(lista1, lista2, contDup):

    resultado = list(set(lista1).intersection(set(lista2)))
    contDup = contDup + len(resultado)
    lista2 = [ele for ele in lista2 if ele not in resultado]
    return lista2,contDup

# Asuma que ya está el pipeline de stanza 
def getWordsAndLemmas(lista):
    doc = nlp(lista)
    lemmas = []
    words = []
    for sent in doc.sentences: 
        for word in sent.words:
            lemmas.append(word.lemma) 
            words.append(word.text)
            
    return words,lemmas
# %% levanto los datos

cwd = os.getcwd()

# ARGENTINA

# Cargo los excels en variables
ipath = cwd+r'\transcripciones_fluidez.xlsx'
dfFluidez = pd.ExcelFile(ipath)
dfFluidez = pd.read_excel(dfFluidez, 'Hoja 1')

# %% elimino NaN

columnasNan = [
"fluency_s_0_15_todo",
"fluency_s_15_30_todo",
"fluency_s_30_45_todo",
"fluency_s_45_60_todo",
"fluency_a_0_15_todo",
"fluency_a_15_30_todo",
"fluency_a_30_45_todo",
"fluency_a_45_60_todo",
"fluency_s_0_15_errores_francos",
"fluency_s_15_30_errores_francos",
"fluency_s_30_45_errores_francos",
"fluency_s_45_60_errores_francos",
"fluency_a_0_15_errores_francos",
"fluency_a_15_30_errores_francos",
"fluency_a_30_45_errores_francos",
"fluency_a_45_60_errores_francos",
"fluency_s_0_15_correctas",
"fluency_s_15_30_correctas",
"fluency_s_30_45_correctas",
"fluency_s_45_60_correctas",
"fluency_a_0_15_correctas",
"fluency_a_15_30_correctas",
"fluency_a_30_45_correctas",
"fluency_a_45_60_correctas"]

dfFluidez[columnasNan]=dfFluidez[columnasNan].fillna("")

# %% Lleno con 0 las numéricas

columnasNumericas = [
"no_pertenece_categoria_0_15_s",
"no_pertenece_categoria_15_30_s",
"no_pertenece_categoria_30_45_s",
"no_pertenece_categoria_45_60_s",
"palabra_repetida_0_15_s",
"palabra_repetida_15_30_s",
"palabra_repetida_30_45_s",
"palabra_repetida_45_60_s",
"familia_repetida_misma_0_15_s",
"familia_repetida_misma_15_30_s",
"familia_repetida_misma_30_45_s",
"familia_repetida_misma_45_60_s",
"nombre_propio_0_15_s",
"nombre_propio_15_30_s",
"nombre_propio_30_45_s",
"nombre_propio_45_60_s",
"no_pertenece_categoria_0_15_a",
"no_pertenece_categoria_15_30_a",
"no_pertenece_categoria_30_45_a",
"no_pertenece_categoria_45_60_a",
"palabra_repetida_0_15_a",
"palabra_repetida_15_30_a",
"palabra_repetida_30_45_a",
"palabra_repetida_45_60_a",
"familia_repetida_misma_0_15_a",
"familia_repetida_misma_15_30_a",
"familia_repetida_misma_30_45_a",
"familia_repetida_misma_45_60_a",
"nombre_propio_0_15_a",
"nombre_propio_15_30_a",
"nombre_propio_30_45_a",
"nombre_propio_45_60_a"]

dfFluidez[columnasNumericas]=dfFluidez[columnasNumericas].fillna(0)

# %%

for index, row in dfFluidez.iterrows():
    contDupp15=0
    contDupp30=0
    contDupp45=0
    contDupp60=0

    # Borro espacios al principio y al final y separo por espacios
    p15 = row['fluency_s_0_15_correctas'].rstrip().lstrip().split()
    p30 = row['fluency_s_15_30_correctas'].rstrip().lstrip().split()
    p45 = row['fluency_s_30_45_correctas'].rstrip().lstrip().split()
    p60 = row['fluency_s_45_60_correctas'].rstrip().lstrip().split()
    animals15 = row['fluency_a_0_15_correctas'].rstrip().lstrip().split()
    animals30 = row['fluency_a_15_30_correctas'].rstrip().lstrip().split()
    animals45 = row['fluency_a_30_45_correctas'].rstrip().lstrip().split()
    animals60 = row['fluency_a_45_60_correctas'].rstrip().lstrip().split()
    
    # Borro las palabras incomprensibles
    while "palabra_incomprensible" in p15: p15.remove("palabra_incomprensible")    
    while "palabra_incomprensible" in p30: p30.remove("palabra_incomprensible")    
    while "palabra_incomprensible" in p45: p45.remove("palabra_incomprensible")    
    while "palabra_incomprensible" in p60: p60.remove("palabra_incomprensible")    
    while "palabra_incomprensible" in animals15: animals15.remove("palabra_incomprensible")    
    while "palabra_incomprensible" in animals30: animals30.remove("palabra_incomprensible")    
    while "palabra_incomprensible" in animals45: animals45.remove("palabra_incomprensible")    
    while "palabra_incomprensible" in animals60: animals60.remove("palabra_incomprensible") 
    
    # Busco duplicados en la misma columna
    p15NoDup,contDupp15 = duplicadosMismaLista(p15)
    p30NoDup,contDupp30 = duplicadosMismaLista(p30)
    p45NoDup,contDupp45 = duplicadosMismaLista(p45)
    p60NoDup,contDupp60 = duplicadosMismaLista(p60)
    animals15NoDup,contDupanimals15 = duplicadosMismaLista(animals15)
    animals30NoDup,contDupanimals30 = duplicadosMismaLista(animals30)
    animals45NoDup,contDupanimals45 = duplicadosMismaLista(animals45)
    animals60NoDup,contDupanimals60 = duplicadosMismaLista(animals60)

    # Busco duplicados entre pares de columnas, cuento y elimino
    p30NoDup,contDupp30 = duplicadosDistintasListas(p15NoDup,p30NoDup,contDupp30)
    p45NoDup,contDupp45 = duplicadosDistintasListas(p15NoDup,p45NoDup,contDupp45)
    p60NoDup,contDupp60 = duplicadosDistintasListas(p15NoDup,p60NoDup,contDupp60)
    p45NoDup,contDupp45 = duplicadosDistintasListas(p30NoDup,p45NoDup,contDupp45)
    p60NoDup,contDupp60 = duplicadosDistintasListas(p30NoDup,p60NoDup,contDupp60)
    p60NoDup,contDupp60 = duplicadosDistintasListas(p45NoDup,p60NoDup,contDupp60)
    animals30NoDup,contDupanimals30 = duplicadosDistintasListas(animals15NoDup,animals30NoDup,contDupanimals30)
    animals45NoDup,contDupanimals45 = duplicadosDistintasListas(animals15NoDup,animals45NoDup,contDupanimals45)
    animals60NoDup,contDupanimals60 = duplicadosDistintasListas(animals15NoDup,animals60NoDup,contDupanimals60)
    animals45NoDup,contDupanimals45 = duplicadosDistintasListas(animals30NoDup,animals45NoDup,contDupanimals45)
    animals60NoDup,contDupanimals60 = duplicadosDistintasListas(animals30NoDup,animals60NoDup,contDupanimals60)
    animals60NoDup,contDupanimals60 = duplicadosDistintasListas(animals45NoDup,animals60NoDup,contDupanimals60)

    # Asigno al dataframe los duplicados encontrados
    dfFluidez.at[index,'palabra_repetida_0_15_s'] = contDupp15
    dfFluidez.at[index,'palabra_repetida_15_30_s'] = contDupp30
    dfFluidez.at[index,'palabra_repetida_30_45_s'] = contDupp45
    dfFluidez.at[index,'palabra_repetida_45_60_s'] = contDupp60
    dfFluidez.at[index,'palabra_repetida_0_15_a'] = contDupanimals15
    dfFluidez.at[index,'palabra_repetida_15_30_a'] = contDupanimals30
    dfFluidez.at[index,'palabra_repetida_30_45_a'] = contDupanimals45
    dfFluidez.at[index,'palabra_repetida_45_60_a'] = contDupanimals60
    
    # Lo convierto a un string separado por espacios
    p15NoDup = ' '.join([str(item) for item in p15NoDup])   
    p30NoDup = ' '.join([str(item) for item in p30NoDup])   
    p45NoDup = ' '.join([str(item) for item in p45NoDup])   
    p60NoDup = ' '.join([str(item) for item in p60NoDup])   
    animals15NoDup = ' '.join([str(item) for item in animals15NoDup])   
    animals30NoDup = ' '.join([str(item) for item in animals30NoDup])   
    animals45NoDup = ' '.join([str(item) for item in animals45NoDup])   
    animals60NoDup = ' '.join([str(item) for item in animals60NoDup])   

    # Lo guardo en el dataFrame
    dfFluidez.at[index,'fluency_s_0_15_correctas'] = p15NoDup
    dfFluidez.at[index,'fluency_s_15_30_correctas'] = p30NoDup
    dfFluidez.at[index,'fluency_s_30_45_correctas'] = p45NoDup
    dfFluidez.at[index,'fluency_s_45_60_correctas'] = p60NoDup
    dfFluidez.at[index,'fluency_a_0_15_correctas'] = animals15NoDup
    dfFluidez.at[index,'fluency_a_15_30_correctas'] = animals30NoDup
    dfFluidez.at[index,'fluency_a_30_45_correctas'] = animals45NoDup
    dfFluidez.at[index,'fluency_a_45_60_correctas'] = animals60NoDup
#%% Guardo el dataframe

dfFluidez.to_csv(cwd+"//transcripciones_fluidez_corregidas.csv")
  
# %% Agrego palabras a mi lematizador

# Load word_dict and composite_dict
import torch
model = torch.load('C:/Users/franc/stanza_resources/es/lemma/ancora.pt', map_location='cpu')
word_dict, composite_dict = model['dicts']

# %% Lemmatizo

import stanza
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

pacLemmaDupMisma = []
pacLemmaDupDistinta = []
pacLemmaanimalsDupMisma = []
pacLemmaanimalsDupDistinta = []
for index, row in dfFluidez.iterrows():

    p15NoDup = row['fluency_s_0_15_correctas']
    p30NoDup = row['fluency_s_15_30_correctas']
    p45NoDup = row['fluency_s_30_45_correctas']
    p60NoDup = row['fluency_s_45_60_correctas']
    animals15NoDup = row['fluency_a_0_15_correctas']
    animals30NoDup = row['fluency_a_15_30_correctas']
    animals45NoDup = row['fluency_a_30_45_correctas']
    animals60NoDup = row['fluency_a_45_60_correctas']
    
    words15,lemmas15 = getWordsAndLemmas(p15NoDup)
    words30,lemmas30 = getWordsAndLemmas(p30NoDup)
    words45,lemmas45 = getWordsAndLemmas(p45NoDup)
    words60,lemmas60 = getWordsAndLemmas(p60NoDup)
    wordsanimals15,lemmasanimals15 = getWordsAndLemmas(animals15NoDup)
    wordsanimals30,lemmasanimals30 = getWordsAndLemmas(animals30NoDup)
    wordsanimals45,lemmasanimals45 = getWordsAndLemmas(animals45NoDup)
    wordsanimals60,lemmasanimals60 = getWordsAndLemmas(animals60NoDup)

    contDupp15=0
    contDupp30=0
    contDupp45=0
    contDupp60=0
    contDupanimals15=0
    contDupanimals30=0
    contDupanimals45=0
    contDupanimals60=0
    
    # Busco duplicados en la misma columna
    p15NoDupLemma,contDupp15 = duplicadosMismaLista(lemmas15)
    p30NoDupLemma,contDupp30 = duplicadosMismaLista(lemmas30)
    p45NoDupLemma,contDupp45 = duplicadosMismaLista(lemmas45)
    p60NoDupLemma,contDupp60 = duplicadosMismaLista(lemmas60)
    animals15NoDupLemma,contDupanimals15 = duplicadosMismaLista(lemmasanimals15)
    animals30NoDupLemma,contDupanimals30 = duplicadosMismaLista(lemmasanimals30)
    animals45NoDupLemma,contDupanimals45 = duplicadosMismaLista(lemmasanimals45)
    animals60NoDupLemma,contDupanimals60 = duplicadosMismaLista(lemmasanimals60)
    
    if(contDupp15>0 or contDupp30>0 or contDupp45>0 or contDupp60>0):
        pacLemmaDupMisma.append(row.Codigo)
    if(contDupanimals15>0 or contDupanimals30>0 or contDupanimals45>0 or contDupanimals60>0):
        pacLemmaanimalsDupMisma.append(row.Codigo)
    contDupp15_30=0
    contDupp15_45=0
    contDupp15_60=0
    contDupp30_45=0
    contDupp30_60=0
    contDupp45_60=0
    contDupanimals15_30=0
    contDupanimals15_45=0
    contDupanimals15_60=0
    contDupanimals30_45=0
    contDupanimals30_60=0
    contDupanimals45_60=0
    # Busco duplicados entre pares de columnas, cuento y elimino
    p30NoDup,contDupp15_30 = duplicadosDistintasListas(lemmas15,lemmas30,contDupp15_30)
    p45NoDup,contDupp15_45 = duplicadosDistintasListas(lemmas15,lemmas45,contDupp15_45)
    p60NoDup,contDupp15_60 = duplicadosDistintasListas(lemmas15,lemmas60,contDupp15_60)
    p45NoDup,contDupp30_45 = duplicadosDistintasListas(lemmas30,lemmas45,contDupp30_45)
    p60NoDup,contDupp30_60 = duplicadosDistintasListas(lemmas30,lemmas60,contDupp30_60)
    p60NoDup,contDupp45_60 = duplicadosDistintasListas(lemmas45,lemmas60,contDupp45_60)    
    animals30NoDup,contDupanimals15_30 = duplicadosDistintasListas(lemmasanimals15,lemmasanimals30,contDupanimals15_30)
    animals45NoDup,contDupanimals15_45 = duplicadosDistintasListas(lemmasanimals15,lemmasanimals45,contDupanimals15_45)
    animals60NoDup,contDupanimals15_60 = duplicadosDistintasListas(lemmasanimals15,lemmasanimals60,contDupanimals15_60)
    animals45NoDup,contDupanimals30_45 = duplicadosDistintasListas(lemmasanimals30,lemmasanimals45,contDupanimals30_45)
    animals60NoDup,contDupanimals30_60 = duplicadosDistintasListas(lemmasanimals30,lemmasanimals60,contDupanimals30_60)
    animals60NoDup,contDupanimals45_60 = duplicadosDistintasListas(lemmasanimals45,lemmasanimals60,contDupanimals45_60)    
    
    if(contDupp15_30>0):
        pacLemmaDupDistinta.append(row.Codigo + " " + "15-30")
    if(contDupp15_45>0):
        pacLemmaDupDistinta.append(row.Codigo + " " + "15-45")
    if(contDupp15_60>0):
        pacLemmaDupDistinta.append(row.Codigo + " " + "15-60")
    if(contDupp30_45>0):
        pacLemmaDupDistinta.append(row.Codigo + " " + "30-45")
    if(contDupp30_60>0):
        pacLemmaDupDistinta.append(row.Codigo + " " + "30-60")
    if(contDupp45_60>0):
        pacLemmaDupDistinta.append(row.Codigo + " " + "45-60")  
        
        
    if(contDupanimals15_30>0):
        pacLemmaanimalsDupDistinta.append(row.Codigo + " " + "15-30")
    if(contDupanimals15_45>0):
        pacLemmaanimalsDupDistinta.append(row.Codigo + " " + "15-45")
    if(contDupanimals15_60>0):
        pacLemmaanimalsDupDistinta.append(row.Codigo + " " + "15-60")
    if(contDupanimals30_45>0):
        pacLemmaanimalsDupDistinta.append(row.Codigo + " " + "30-45")
    if(contDupanimals30_60>0):
        pacLemmaanimalsDupDistinta.append(row.Codigo + " " + "30-60")
    if(contDupanimals45_60>0):
        pacLemmaanimalsDupDistinta.append(row.Codigo + " " + "45-60")  


# %% Con nltk

from nltk import word_tokenize
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

pacStemDupMisma = []
pacStemDupDistinta = []
pacStemanimalsDupMisma = []
pacStemanimalsDupDistinta = []
for index, row in dfFluidez.iterrows():

    p15NoDup = row['fluency_s_0_15_correctas']
    p30NoDup = row['fluency_s_15_30_correctas']
    p45NoDup = row['fluency_s_30_45_correctas']
    p60NoDup = row['fluency_s_45_60_correctas']
    animals15NoDup = row['fluency_a_0_15_correctas']
    animals30NoDup = row['fluency_a_15_30_correctas']
    animals45NoDup = row['fluency_a_30_45_correctas']
    animals60NoDup = row['fluency_a_45_60_correctas']
    
    stem15 = [stemmer.stem(i) for i in word_tokenize(p15NoDup)]
    stem30 = [stemmer.stem(i) for i in word_tokenize(p30NoDup)]
    stem45 = [stemmer.stem(i) for i in word_tokenize(p45NoDup)]
    stem60 = [stemmer.stem(i) for i in word_tokenize(p60NoDup)]
    stemanimals15 = [stemmer.stem(i) for i in word_tokenize(animals15NoDup)]
    stemanimals30 = [stemmer.stem(i) for i in word_tokenize(animals30NoDup)]
    stemanimals45 = [stemmer.stem(i) for i in word_tokenize(animals45NoDup)]
    stemanimals60 = [stemmer.stem(i) for i in word_tokenize(animals60NoDup)]

    contDupp15=0
    contDupp30=0
    contDupp45=0
    contDupp60=0
    contDupanimals15=0
    contDupanimals30=0
    contDupanimals45=0
    contDupanimals60=0
    
    # Busco duplicados en la misma columna
    p15NoDupStem,contDupp15 = duplicadosMismaLista(stem15)
    p30NoDupStem,contDupp30 = duplicadosMismaLista(stem30)
    p45NoDupStem,contDupp45 = duplicadosMismaLista(stem45)
    p60NoDupStem,contDupp60 = duplicadosMismaLista(stem60)
    animals15NoDupStem,contDupanimals15 = duplicadosMismaLista(stemanimals15)
    animals30NoDupStem,contDupanimals30 = duplicadosMismaLista(stemanimals30)
    animals45NoDupStem,contDupanimals45 = duplicadosMismaLista(stemanimals45)
    animals60NoDupStem,contDupanimals60 = duplicadosMismaLista(stemanimals60)
    
    if(contDupp15>0 or contDupp30>0 or contDupp45>0 or contDupp60>0):
        pacStemDupMisma.append(row.Codigo)
    if(contDupanimals15>0 or contDupanimals30>0 or contDupanimals45>0 or contDupanimals60>0):
        pacStemanimalsDupMisma.append(row.Codigo)
        
    contDupp15_30=0
    contDupp15_45=0
    contDupp15_60=0
    contDupp30_45=0
    contDupp30_60=0
    contDupp45_60=0
    contDupanimals15_30=0
    contDupanimals15_45=0
    contDupanimals15_60=0
    contDupanimals30_45=0
    contDupanimals30_60=0
    contDupanimals45_60=0
    # Busco duplicados entre pares de columnas, cuento y elimino
    p30NoDupStem,contDupp15_30 = duplicadosDistintasListas(stem15,stem30,contDupp15_30)
    p45NoDupStem,contDupp15_45 = duplicadosDistintasListas(stem15,stem45,contDupp15_45)
    p60NoDupStem,contDupp15_60 = duplicadosDistintasListas(stem15,stem60,contDupp15_60)
    p45NoDupStem,contDupp30_45 = duplicadosDistintasListas(stem30,stem45,contDupp30_45)
    p60NoDupStem,contDupp30_60 = duplicadosDistintasListas(stem30,stem60,contDupp30_60)
    p60NoDupStem,contDupp45_60 = duplicadosDistintasListas(stem45,stem60,contDupp45_60)    
    animals30NoDupStem,contDupanimals15_30 = duplicadosDistintasListas(stemanimals15,stemanimals30,contDupanimals15_30)
    animals45NoDupStem,contDupanimals15_45 = duplicadosDistintasListas(stemanimals15,stemanimals45,contDupanimals15_45)
    animals60NoDupStem,contDupanimals15_60 = duplicadosDistintasListas(stemanimals15,stemanimals60,contDupanimals15_60)
    animals45NoDupStem,contDupanimals30_45 = duplicadosDistintasListas(stemanimals30,stemanimals45,contDupanimals30_45)
    animals60NoDupStem,contDupanimals30_60 = duplicadosDistintasListas(stemanimals30,stemanimals60,contDupanimals30_60)
    animals60NoDupStem,contDupanimals45_60 = duplicadosDistintasListas(stemanimals45,stemanimals60,contDupanimals45_60)
    
    if(contDupp15_30>0):
        pacStemDupDistinta.append(row.Codigo + " " + "15-30")
    if(contDupp15_45>0):
        pacStemDupDistinta.append(row.Codigo + " " + "15-45")
    if(contDupp15_60>0):
        pacStemDupDistinta.append(row.Codigo + " " + "15-60")
    if(contDupp30_45>0):
        pacStemDupDistinta.append(row.Codigo + " " + "30-45")
    if(contDupp30_60>0):
        pacStemDupDistinta.append(row.Codigo + " " + "30-60")
    if(contDupp45_60>0):
        pacStemDupDistinta.append(row.Codigo + " " + "45-60")    
    if(contDupanimals15_30>0):
        pacStemanimalsDupDistinta.append(row.Codigo + " " + "15-30")
    if(contDupanimals15_45>0):
        pacStemanimalsDupDistinta.append(row.Codigo + " " + "15-45")
    if(contDupanimals15_60>0):
        pacStemanimalsDupDistinta.append(row.Codigo + " " + "15-60")
    if(contDupanimals30_45>0):
        pacStemanimalsDupDistinta.append(row.Codigo + " " + "30-45")
    if(contDupanimals30_60>0):
        pacStemanimalsDupDistinta.append(row.Codigo + " " + "30-60")
    if(contDupanimals45_60>0):
        pacStemanimalsDupDistinta.append(row.Codigo + " " + "45-60")  




















