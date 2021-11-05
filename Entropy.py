# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Norbey Muñoz, Luis Murillo'

import pandas as pd
import numpy as np
from math import log

def entropy(data_classes, base=2):
    '''
    Calcula la entropía de un conjunto de labels (instanciaciones de clases)
    :param base: base logarítmica para el cálculo
    :param data_classes: series con labels de los ejemplos en un dataset
    :return: valor de la entropía
    '''
    classes = np.unique(data_classes)
    N = len(data_classes)
    ent = 0  # inicialización de la entropía

    # iteración sobre las clases
    for c in classes:
        partition = data_classes[data_classes == c]  # data con class = c
        proportion = len(partition)/N
        # calculamos entropía
        ent -= proportion * log(proportion, base)

    return ent

def cut_point_information_gain(X, y, cut_point):
    '''
    Retorna la ganancia de la información obtenida al dividir un atributo numérico 
    en dos subconjuntos según el cut_point
    :param dataset(X): pandas dataframe con una columna para los valores del atributo y una columna para la clase
    :param cut_point: umbral en el que dividir el atributo numérico
    :param feature_label: label de la columna de los valores de atributo numérico en los datos
    :param class_label(y): arreglo con label de las instancias de clase
    :return: ganancia de información de la partición obtenida por umbral cut_point
    '''
    entropy_full = entropy(y)  # calcular la entropía del conjunto de datos completo (sin división) 

    # dividir los datos según cut_point
    data_left_mask = X <= cut_point # dataset[dataset[feature_label] <= cut_point]
    data_right_mask = X > cut_point # dataset[dataset[feature_label] > cut_point]
    (N, N_left, N_right) = (len(X), data_left_mask.sum(), data_right_mask.sum())

    gain = entropy_full - (N_left / N) * entropy(y[data_left_mask]) - \
        (N_right / N) * entropy(y[data_right_mask])

    return gain