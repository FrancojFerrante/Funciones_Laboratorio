# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:33:17 2021

@author: franc
"""

# Combinatoria.py>

import itertools


def combinatoria_texto_no_orden(columnas_a_combinar, cantidad = -1):
    """
    Crea las posibles combinaciones que pueden darse entre las columnas de interés.
    No tiene en cuenta el orden. Ej: ['Col1','Col2'] es igual a ['Col2','Col1']


    Parameters
    ----------
    columnas_a_combinar : list
        lista de columnas que se quieren combinar.

    Returns
    -------
    all_combinations : lista de listas
        Una lista donde en cada elemento contiene otra lista con la combinación
        de las columnas_a_combinar.

    """
    all_combinations = []
    if (cantidad == -1):
        for r in range(len(columnas_a_combinar) + 1):
            all_combinations += list(itertools.combinations(columnas_a_combinar, r))
        all_combinations.pop(0) # Elimino el primero porque está vacío
        return all_combinations
    else:
        for r in range(cantidad + 1):
            all_combinations += list(itertools.combinations(columnas_a_combinar, r))
        all_combinations.pop(0) # Elimino el primero porque está vacío
        return all_combinations