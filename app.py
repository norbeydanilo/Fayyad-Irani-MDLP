# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from MDLP import MDLP_Discretizer

def main():
    ######### EJEMPLO CASO DE USO #############

    # lectura del dataset
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    feature_names, class_names = dataset['feature_names'], dataset['target_names']
    numeric_features = np.arange(X.shape[1])  # todas las características son numéricas. Serán discretizadas

    # dividir entre entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # inicializar el objeto del discretizador y ajustarlo a los datos de entrenamiento
    discretizer = MDLP_Discretizer(features=numeric_features)
    discretizer.fit(X_train, y_train)
    X_train_discretized = discretizer.transform(X_train)

    # aplicar la misma discretización al conjunto de prueba
    X_test_discretized = discretizer.transform(X_test)

    # muestra de una porción de datos originales y discretizados
    print ('Dataset original:\n%s' % str(X_train[0:5]))
    print ('Dataset discretizado:\n%s' % str(X_train_discretized[0:5]))

    # vea cómo se discretizó la característica 0
    print ('Característica: %s' % feature_names[0])
    print ('Intervalo cut-points: %s' % str(discretizer._cuts[0]))
    print ('Descripción del bin: %s' % str(discretizer._bin_descriptions[0]))
    
    # vea cómo se discretizó la característica 1
    print ('Característica: %s' % feature_names[1])
    print ('Intervalo cut-points: %s' % str(discretizer._cuts[1]))
    print ('Descripción del bin: %s' % str(discretizer._bin_descriptions[1]))
    
    # vea cómo se discretizó la característica 2
    print ('Característica: %s' % feature_names[2])
    print ('Intervalo cut-points: %s' % str(discretizer._cuts[2]))
    print ('Descripción del bin: %s' % str(discretizer._bin_descriptions[2]))
    
    # vea cómo se discretizó la característica 3
    print ('Característica: %s' % feature_names[3])
    print ('Intervalo cut-points: %s' % str(discretizer._cuts[3]))
    print ('Descripción del bin: %s' % str(discretizer._bin_descriptions[3]))

if __name__ == '__main__':
    main()