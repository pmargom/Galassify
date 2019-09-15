# SDSS_clusters.py, v0.3
#
# Script para la determinación de outliers :
#
# ## Argumentos de entrada:
#
# --input - Nombre del archivo CSV con el dataset a analizar
# --output - Nombre del CSV de salida (por defecto "output.csv")
# --lat - Número de variables del espacio latente (250 por defecto)
# --num_clusters - Número de clusters para segmentación (7 por defecto) 
# --representacion - 0, 2 o 3. Número de variables para representación gráfica de clusters (2 por defecto)
# --epochs - Número de ciclos de entrenamiento del autoencoder (40 por defecto)
# --eps - Valor eps para algoritmo DBSCAN (1.8 por defecto)
#
# 
# ## Salida:
# 
# - Archivo csv con las columnas Plate, MJD,Fiber, redshift y cluster. Esta última contiene
#   el cluster asignado a cada espectro, siendo -1 en el caso de considerarse un outlier.
#    


import numpy as np
import pandas as pd
import argparse

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN 
from sklearn.mixture import GaussianMixture

from keras.layers import Input, Dense
from keras.models import Model


## FUNCIONES
# Función recogida en StackOverFlow. Devuelve el valor de posición en el array más cercano al "target"
# Entradas: A, el array a considerar, y el target o valor a buscar.
# En este caso, hay que tener en cuenta que la frecuencia a buscar es el índice en el dataset
def find_closest(A, target): 
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


# En esta función se calcula la media móvil con un número variable de peridos. Los parámetros de entrada son:
# - espectro, la pd.Serie con el espectro completo a corregir. Tiene como índice la frecuencia.
# - frecuencia, el valor de la frecucncia cuyo flux es negativo. Se corresponde con el mínimo de la "función
#   espectro". Debe llegar como float.
# - periodo, el de la media móvila considerar. Se tiene 20 por defecto (se ajusta bien al espectro medio)
def media_movil(espectro, frecuencia, periodo = 20):
    sma = espectro.rolling(periodo).mean() # sma es la media móvil de todo el espectro
    # Calculamos el valor de sma en la frecuencia dada
    if frecuencia < float(sma.index[periodo]):   # La Serie de sma no tiene valores para los primeros 20 valores
        frecuencia = float(sma.index[periodo])
    indice_sma = pd.to_numeric(sma.index)
    
    # Por alguna razón al tratar los índices y convertirlos de str a float, se producen errores al reconocer valores
    # en casos contados. Para evitarlo, tomamos el valor de frecuencia más cercano
    if frecuencia in indice_sma:
        # Tomamos como valor del flux el de la media móvil en la frecuencia problemática 
        return sma[indice_sma==frecuencia]
    else:
        # Es en estos casos cuando surge el error: tomamos el valor de la siguiente frecuencia a la problemática
        print("¡No esta!")
        frecuencia = float(sma.index[find_closest(indice_sma, frecuencia)+1])
        print(frecuencia)
        return sma[indice_sma==frecuencia]
    
# Función para el entrenamiento del autoencoder y extracción del espacio latente resultante
# Dentro se estandariza el dataset de entrada y se divide el dataset en entrenamiento y validación.
# Sale el espacio latente correspondiente
def autoencoder(dataset_origen, dim_latente, epochs):
    # Para luego entrenar los modelos y tener un set de test para las medidas de accuracy, dividimos el dataset en train y test (nos vale un 15% para el test). Recordemos que estamos usando un aprendizaje no supervisado y no tenemos etiquetas.
    espectros_train, espectros_test, _, _ = train_test_split(dataset_origen, dataset_origen, test_size=0.15, random_state=21)

    # Ahora podemos seguir con el proceso, primero estandarizando el dataset
    scaler = MinMaxScaler() # = np.array(data.apply(lambda x: (x-x.min()) / (x.max()-x.min())))
    scaler.fit(espectros_train)
    espectros_train_scaled = pd.DataFrame(scaler.transform(espectros_train), 
                               columns=espectros_train.columns,
                               index=espectros_train.index)
    espectros_test_scaled = pd.DataFrame(scaler.transform(espectros_test),
                             columns=espectros_test.columns,
                             index=espectros_test.index)

    # Esta es la versiós estandarizada de todo el dataset para comparaciones posteriores
    espectros_scaled = pd.DataFrame(scaler.transform(dataset_origen), 
                               columns=dataset_origen.columns,
                               index=dataset_origen.index)
    
    dim_input = espectros_train_scaled.shape[1]
    
    input = Input(shape=((dim_input, )))
    encoded = Dense(1000, activation='relu')(input)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(dim_latente, activation='relu')(encoded)
    decoded = Dense(500, activation='relu')(encoded)
    decoded = Dense(1000, activation='relu')(decoded)
    decoded = Dense(dim_input, activation='sigmoid')(decoded)

    autoencoder_deep = Model(input, decoded)
    encoder = Model(input, encoded)
    
    autoencoder_deep.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy"])
    autoencoder_deep.fit(espectros_train_scaled, espectros_train_scaled,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(espectros_test_scaled, espectros_test_scaled))  
    
    return encoder.predict(espectros_scaled)



## Tratamiento de argumentos
# --espectro - Nombre del archivo CSV con el dataset a analizar
# --output - Nombre del CSV de salida (por defecto "output.csv")
# --lat - Número de variables del espacio latente (250 por defecto)
# --num_clusters - Número de clusters para segmentación (7 por defecto) 
# --representacion - 0, 2 o 3. Número de variables para representación gráfica de clusters (2 por defecto)
# --epochs - Número de ciclos de entrenamiento del autoencoder (40 por defecto)
# --eps - Valor eps para algoritmo DBSCAN (1.8 por defecto)

ap = argparse.ArgumentParser()
ap.add_argument("--input", required = False,
                help = "Ruta de acceso completa al archivo CSV donde está el espectro a analizar",
                default="./data/datasetV3_3KRandom.csv")
ap.add_argument("--output", required = False,
                help = "Ruta de acceso completa al archivo CSV donde está el dataset de salida (output.csv por defecto)",
                default="./output.csv")
ap.add_argument("--lat", required = False,
                help = "Número de variables del espacio latente (250 por defecto)",
                default = 250)
ap.add_argument("--num_clusters", required = False,
                help = "Número de clasters a extraer vía GMM (7 por defecto)",
                default = 7)
ap.add_argument("--representacion", required = False,
                help = "0, 2 o 3 - Extracción del espacio latente para representarlo (2 por defecto)",
                default = 2)
ap.add_argument("--epochs", required = False,
                help = "Número de epochs para entrenamiento del autoencoder (40 por defecto)",
                default = 40)
ap.add_argument("--eps", required = False,
                help = "Valor eps para DBSCAN (1.8 por defecto)",
                default = 1.8)
args = vars(ap.parse_args())

# Cargamos el dataset
data_origen = pd.read_csv(args["input"], sep=";")

# ¡Ojo! Para el entrenamiento, no necesitamos trazabilidad de los espectros (y estos campos nos estorban para meterlos en la red neuronal), pero después habrá que recuperar los valores de plate, MJD, Fiber y Z para poder saber de qué galaxia estamos hablando.
data = data_origen[data_origen.columns[4:]]

# Se detecta que existen espectros donde el valor del flux es negativo, lo que no es posible. Se debe a un error en la lectura. Vamos a ver el tamaño de este problema.
# Vamos a detectar espectros en los que uno o más valores son negativos (no podemos asegurar que habrá un único
# valor negativo)
print("****\n**** Tratamiento de espectros con valores negativos\n****")
negativos = []
for i in range(len(data)):
        if any(data.iloc[i] < 0):
            negativos.append((i, sum(n < 0 for n in data.iloc[i])))

# Un 0,28% de todos los valores de frecuencias son negativos, es decir, erróneos (no puede haber cantidades de flux negativas, es decir "luz" negativa). Esto podría disminuir la confianza en la fiabilidad de los datos del dataset.
# Además de haber bastantes valores erróneos, disminuyen mucho el valor mínimo de muchos espectros, haciendo que la estandarización que vamos a hacer después comprima demasiado los valores correctos (lo que hace que casi todos los espectros estandarizados "parezcan iguales".
# Por esto, vamos a corregir la situación con la siguiente estrategia:
# 
# * Se eliminarán aquellos espectros que tengan dos o más valores negativos (se considerará que ha existido algún error de lectura que puede comprometer la fiabilidad de todo el espectro).
# * En los restantes, se cambiará el único valor negativo por el valor correspondiente al continuum, tomando como tal el valor de la media móvil de 20 periodos en ese punto.

# Eliminamos los espectros con dos o más valores negativos
# Hay que eliminarlos también del dataset de origen, para así mantener el índice igual
dim_orig = data.shape[0]
for i in negativos:
        if i[1]>1:
            data.drop(i[0], axis=0, inplace=True)
            data_origen.drop(i[0], axis=0, inplace=True)
            num_negativos = dim_orig - data.shape[0]
print ("Hay " + str(num_negativos) + " espectros con 2 o más valores negativos. Se han eliminado")

# Y ahora tratamos los espectros con un solo valor negativo. En estos casos, esa frecuencia tendrá el valor de
# flux mínimo del espectro

# Aquí corregimos los valores de flux negativos por los de la media móvil, con la ayuda de la función definida
# Nos ayudamos del hecho de que, habiendo un solo negativo, éste será el valor mínimo
for i in range(len(data)):
        if any(data.iloc[i] < 0):
            data.iloc[i][data.iloc[i].index==data.iloc[i].idxmin()] = media_movil(data.iloc[i], float(data.iloc[i].idxmin()))[0]

# Entrenamos el autoencoder, de momento para las variables del espacio latente definidas en el
# argumento de entrada. Más tarde lo repetiremos para las 2 o 3 variables para representación
print("****\n**** Entrenando autoencoder\n****")
espectros_latentes = autoencoder(data, int(args["lat"]), int(args["epochs"]))

# ## Aplicación de algoritmos de clustering
# ## 1.- DBSCAN
print("****\n**** Clustering DBSCAN\n****")
clustering = DBSCAN(eps=args["eps"], min_samples=2, n_jobs=-1).fit(espectros_latentes)
labels_DBSCAN = np.unique(clustering.labels_, return_counts=True)
clusters_DBSCAN = []
for i in range(-1,len(labels_DBSCAN[0]), 1): # Empezamos en -1 para coger también los outliers, en el índice 0 del array labels
        clusters_DBSCAN.append(np.where(clustering.labels_==i))

# ## 2.- Gaussian Mixture Model (GMM)
print("****\n**** Clustering GMM\n****")
clusters_GMM = []
num_clusters = int(args["num_clusters"])
repres = int(args["representacion"])  # Si vamos o no a representar el espacio latente y en cuantas dimensiones

clustering = GaussianMixture(n_components=num_clusters).fit(espectros_latentes)
labels = clustering.predict(espectros_latentes)
for i in range(num_clusters):
        clusters_GMM.append([i, np.where(labels==i)])

print("****\n**** Creando CSV de salida\n****")
columnas = ['PLATE', 'MJD', 'FIBER', 'Z', 'cluster']
if repres > 0:
        columnas.extend(["x_lat", "y_lat"])
        if repres == 3:
            columnas.append("z_lat")
data_output = pd.DataFrame(data_origen, columns=columnas) # Copiamos los datos de identificación en el nuevo dataset

# Recorremos todo el dataset 
for item in range(len(data_origen)):
        # En cada espectro, vemos cual es el cluster que se le ha asignado. Primero vemos si es un outlier o no
        if item in np.array(clusters_DBSCAN[0]):
            data_output["cluster"].iloc[item] = -1
        else: # Si no lo es, buscamos el cluster asignado en el intento 0
            for k in range(num_clusters):
                if item in np.array(clusters_GMM[k][1]):
                    data_output["cluster"].iloc[item] = k

# Por último, sacamos, si procede, el espacio latente en 2  3 dimensiones para que se pueda representar

if repres > 0:
        print("****\n**** Creando la representación del espacio latente\n****")
        data_repres = autoencoder(data, repres, int(args["epochs"]))
        data_output["x_lat"] = pd.Series([data_repres[i][0] for i in range(len(data_repres))]) 
        data_output["y_lat"] = pd.Series([data_repres[i][1] for i in range(len(data_repres))])
        if repres==3:
            data_output["z_lat"] = pd.Series([data_repres[i][2] for i in range(len(data_repres))])

# Y exportamos todo a un CSV
data_output.to_csv(args["output"], columns=data_output.columns, index=False, header=True)
print("****\n**** ¡Terminado!\n****")
exit()




