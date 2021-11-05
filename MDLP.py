# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Norbey Muñoz, Luis Murillo'

import numpy as np
from Entropy import entropy, cut_point_information_gain
from math import log
from sklearn.base import TransformerMixin
from sklearn import datasets
from sklearn.model_selection import train_test_split

def previous_item(a, val):
    idx = np.where(a == val)[0][0] - 1
    return a[idx]

class MDLP_Discretizer(TransformerMixin):
    #01#
    def __init__(self, features=None, raw_data_shape=None):
        '''
        Inicializa el objeto discretizador:
            guarda una copia sin procesar de los datos y crea self._data con solo características para discretizar y clases
            calcula la entropía inicial (antes de cualquier división)
            self._features = características a discretizar
            self._classes = clases únicas
            self._class_name = label de la clase en pandas dataframe
            self._data = partición de datos con solo características de interés y clase
            self._cuts = diccionario con cut points para cada característica
        :param X: pandas dataframe con datos para discretizar
        :param class_label: nombre de la columna que contiene la clase en el dataframe de entrada
        :param features: Si !None, característica que el usuario desea discretizar específicamente
        :return:
        '''
        # Inicializar descripciones de bins de discretización
        self._bin_descriptions = {}

        # Crear un arreglo con índices de atributos para discretizar
        # col_idx características
        if features is None:  # supone que todas las columnas son numéricas y deben discretizarse
            if raw_data_shape is None:
                raise Exception("Si features=None, raw_data_shape debe ser una tupla no vacía")
            self._col_idx = range(raw_data_shape[1])
        else:
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            if np.issubdtype(features.dtype, np.integer):
                self._col_idx = features
            elif np.issubdtype(features.dtype, np.bool):  # características pasadas como mask
                if raw_data_shape is None:
                    raise Exception('Si las características son un array boolean, raw_data_shape debe ser != None')
                if len(features) != self._data_raw.shape[1]:
                    raise Exception('Boolean mask de la columna debe tener dimensiones (NColumns,)')
                self._col_idx = np.where(features)
            else:
                raise Exception('El argumento de características debe ser un np.array de índices de columna o una boolean mask')
    
    #02#
    def fit(self, X, y):
        self._data_raw = X  # copia de los datos de entrada originales
        self._class_labels = y.reshape(-1, 1)  # asegurarse que las etiquetas de clase sean un vector columna
        self._classes = np.unique(self._class_labels)

        if len(self._col_idx) != self._data_raw.shape[1]:  # algunas columnas no estarán discretizadas
            self._ignore_col_idx = np.array([var for var in range(self._data_raw.shape[1]) if var not in self._col_idx])

        # inicializar los puntos de corte
        self._cuts = {f: [] for f in self._col_idx}

        # precalcular todos los puntos límite en el conjunto de datos 3,4
        self._boundaries = self.compute_boundary_points_all_features()

        # obtener cortes para todas las caracteristicas 5, 6, 7, 8, 9
        self.all_features_accepted_cutpoints()

        # generar descriptions "string" para los bins 10
        self.generate_bin_descriptions()

        return self
    
    #03#
    def compute_boundary_points_all_features(self):
        '''
        calcula todos los posibles puntos límite para cada atributo en self._features (características por discretizar)
        '''
        def padded_cutpoints_array(arr, N):
            cutpoints = self.feature_boundary_points(arr)
            padding = np.array([np.nan] * (N - len(cutpoints)))
            return np.concatenate([cutpoints, padding])

        boundaries = np.empty(self._data_raw.shape)
        boundaries[:, self._col_idx] = np.apply_along_axis(padded_cutpoints_array, 0, self._data_raw[:, self._col_idx], self._data_raw.shape[0])
        mask = np.all(np.isnan(boundaries), axis=1)
        return boundaries[~mask]
    
    #04#
    def feature_boundary_points(self, values):
        '''
        Dado un atributo, encontrar todos los potenciales cut_points (puntos límite)
        :param values: índices de las filas para las que el valor de la característica cae dentro del intervalo de interés
        :return: array con potenciales cut_points
        '''

        missing_mask = np.isnan(values)
        data_partition = np.concatenate([values[:, np.newaxis], self._class_labels], axis=1)
        data_partition = data_partition[~missing_mask]
        # ordenar datos por valores
        data_partition = data_partition[data_partition[:, 0].argsort()]

        # obtención de valores únicos en la columna
        unique_vals = np.unique(data_partition[:, 0])  # cada uno podría ser un límite del bin
        # encuentra si cuando cambia la característica hay diferentes valores de clase
        boundaries = []
        for i in range(1, unique_vals.size):  # por definición, el primer valor único no puede ser un límite
            previous_val_idx = np.where(data_partition[:, 0] == unique_vals[i-1])[0]
            current_val_idx = np.where(data_partition[:, 0] == unique_vals[i])[0]
            merged_classes = np.union1d(data_partition[previous_val_idx, 1], data_partition[current_val_idx, 1])
            if merged_classes.size > 1:
                boundaries += [unique_vals[i]]
        boundaries_offset = np.array([previous_item(unique_vals, var) for var in boundaries])
        return (np.array(boundaries) + boundaries_offset) / 2
    
    #05#    
    def all_features_accepted_cutpoints(self):
        '''
        Calcula puntos de corte para todas las características numéricas (las que están en self._features)
        :return:
        '''
        for attr in self._col_idx:
            self.single_feature_accepted_cutpoints(X=self._data_raw[:, attr], y=self._class_labels, feature_idx=attr)
        return
    
    #06#
    def single_feature_accepted_cutpoints(self, X, y, feature_idx):
        '''
        Calcula los cortes para agrupar una característica de acuerdo con el criterio MDLP
        :param feature_idx: atributo de interés
        :return: lista de cortes para la característica de agrupamiento en la partición cubierta por el índice de partición
        '''

        # eliminación missing data
        mask = np.isnan(X)
        X = X[~mask]
        y = y[~mask]

        # detener si valores de característica constantes o nulos
        if len(np.unique(X)) < 2:
            return
        # determinar si cortar y dónde
        cut_candidate = self.best_cut_point(X, y, feature_idx)
        if cut_candidate == None:
            return
        decision = self.MDLPC_criterion(X, y, feature_idx, cut_candidate)

        # máscaras de partición
        left_mask = X <= cut_candidate
        right_mask = X > cut_candidate

        # aplicar decisión
        if not decision:
            return  # si la partición no fue aceptada, no hay nada más que hacer
        if decision:
            # ahora tenemos dos nuevas particiones que deben examinarse
            left_partition = X[left_mask]
            right_partition = X[right_mask]
            if (left_partition.size == 0) or (right_partition.size == 0):
                return # punto extremo seleccionado, no particionar
            self._cuts[feature_idx] += [cut_candidate]  # aceptar partición
            self.single_feature_accepted_cutpoints(left_partition, y[left_mask], feature_idx)
            self.single_feature_accepted_cutpoints(right_partition, y[right_mask], feature_idx)
            # ordenar puntos de corte en orden ascendente
            self._cuts[feature_idx] = sorted(self._cuts[feature_idx])
            return
    
    #07#
    def best_cut_point(self, X, y, feature_idx):
        '''
        Selecciona el mejor punto de corte para una característica en una partición de datos según la ganancia de información
        :param data(X): partición de los datos (pandas dataframe)
        :param feature_idx: atributo objetivo
        :return: valor del punto de corte con mayor ganancia de información (si hay muchos, elige el primero). None si no hay candidatos
        '''
        candidates = self.boundaries_in_partition(X, feature_idx=feature_idx)
        if candidates.size == 0:
            return None
        gains = [(cut, cut_point_information_gain(X, y, cut_point=cut)) for cut in candidates]
        gains = sorted(gains, key=lambda x: x[1], reverse=True)

        return gains[0][0] # retorna el punto de corte
    
    #08#
    def boundaries_in_partition(self, X, feature_idx):
        '''
        De la colección de todos los puntos de corte para todas las características, busque puntos de corte que se encuentren 
        dentro del rango de valores de atributo de las particiones de la característica
        :param data(X): partición de los datos (pandas dataframe)
        :param feature_idx: atributo de interés
        :return: puntos dentro del rango de la característica
        '''
        range_min, range_max = (X.min(), X.max())
        mask = np.logical_and((self._boundaries[:, feature_idx] > range_min), (self._boundaries[:, feature_idx] < range_max))
        return np.unique(self._boundaries[:, feature_idx][mask])
    
    #09#
    def MDLPC_criterion(self, X, y, feature_idx, cut_point):
        '''
        Determina si se acepta una partición de acuerdo con el criterio MDLP
        :param feature: característica de interés
        :param cut_point: cut_point propuesto
        :param feature_index: índice de la muestra (partición de dataframe) en el intervalo de interés
        :return: True/False, si se acepta la partición
        '''
        # obtener el dataframe solo con el atributo deseado y las columas de clase, y dividir por cut_point
        left_mask = X <= cut_point
        right_mask = X > cut_point

        # calcular la ganancia de información obtenida al dividir los datos en cut_point
        cut_point_gain = cut_point_information_gain(X, y, cut_point)
        # calculo del término delta en el criterio MDLP
        N = len(X) # número de ejemplos en la partición actual 
        partition_entropy = entropy(y)
        k = len(np.unique(y))
        k_left = len(np.unique(y[left_mask]))
        k_right = len(np.unique(y[right_mask]))
        entropy_left = entropy(y[left_mask])  # entropía de la partición
        entropy_right = entropy(y[right_mask])
        delta = log(3 ** k, 2) - (k * partition_entropy) + (k_left * entropy_left) + (k_right * entropy_right)

        # dividir o no dividir
        gain_threshold = (log(N - 1, 2) + delta) / N

        if cut_point_gain > gain_threshold:
            return True
        else:
            return False
     
    #10#
    def generate_bin_descriptions(self):
        '''
        Discretiza los datos aplicando bins según self._cuts. Se obtiene una descripción de los bins
        :return:
        '''
        bin_label_collection = {}
        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                bin_label_collection[attr] = ['All']
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                start_bin_indices = range(0, len(cuts) - 1)
                bin_labels = ['%s_to_%s' % (str(cuts[i]), str(cuts[i+1])) for i in start_bin_indices]
                bin_label_collection[attr] = bin_labels
                self._bin_descriptions[attr] = {i: bin_labels[i] for i in range(len(bin_labels))}    
    
    #11#
    def transform(self, X, inplace=False):
        if inplace:
            discretized = X
        else:
            discretized = X.copy()
        discretized = self.apply_cutpoints(discretized)
        return discretized
    
    #12#            
    def apply_cutpoints(self, data):
        '''
        Discretiza los datos aplicando bins según self._cuts. Se obtiene el nuevo discretizado y una descripción de los bins
        :return:
        '''
        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                # data[:, attr] = 'All'
                data[:, attr] = 0
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                discretized_col = np.digitize(x=data[:, attr], bins=cuts, right=False).astype('float') - 1
                discretized_col[np.isnan(data[:, attr])] = np.nan
                data[:, attr] = discretized_col
        return data