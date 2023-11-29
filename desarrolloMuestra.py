import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# for visualization
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


###########################################################################################################################################################################
#########################################################           IMPORTAR EL CONJUNTO DE DATOS           ###############################################################
###########################################################################################################################################################################
# the dataset have already been split into train, test set
test_set = "mnist_test.csv"
train_set = "mnist_train.csv"

# dump the train dataset into pandas dataframe for easy manipulation
# and plotting (don't touch the test set until when you are finally going to
# test in on the final model)
X_train = pd.read_csv(train_set)
X_test = pd.read_csv(test_set)
print(X_train.head())

# to check out what we are going to be working with
X_train.info()

# re-arrange:  y:class and x:features
y_train = X_train['label'].copy()
X_train.drop('label', axis=1, inplace=True)

# re-arrange:  y:class and x:features
y_test = X_test['label'].copy()
X_test.drop('label', axis=1, inplace=True)


###########################################################################################################################################################################
#########################################################           GRAFICAR Y UTILIZAR LOS CONJUNTOS PARA DESCRIPCIÓN INCIAL          ###############################################################
###########################################################################################################################################################################

########################################################################## Barplot del conjunto test
# num samples per class
value_counts = y_test.value_counts()

# Ordenar los valores y etiquetas por valor descendente
sorted_counts = value_counts.sort_index(ascending=True)
sorted_labels = sorted_counts.index

max_count = sorted_counts.max()  # Recuento máximo

# Crear el gráfico de barras
plt.bar(sorted_labels, sorted_counts)

# Establecer una altura fija para todas las etiquetas de texto
label_height = max_count + 10  # Puedes ajustar este valor según tus preferencias

# Agregar etiquetas de texto con el recuento en la altura fija
for index, value in enumerate(sorted_counts):
    plt.text(index, label_height, str(value), ha='center', va='bottom')

# Ajustar el eje x para mostrar cada valor individualmente
plt.xticks(range(len(sorted_labels)), sorted_labels)

# Agregar etiquetas en los ejes
plt.xlabel('Clase')
plt.ylabel('Número de Instancias')

# Añadir título
plt.title('Barplot del conjunto test')

plt.savefig('instanciasPorClaseTest.png')
plt.show()


########################################################################### Barplot del conjunto train
# num samples per class
value_counts = y_train.value_counts()

# Ordenar los valores y etiquetas por valor descendente
sorted_counts = value_counts.sort_index(ascending=True)
sorted_labels = sorted_counts.index

max_count = sorted_counts.max()  # Recuento máximo

# Crear el gráfico de barras
plt.bar(sorted_labels, sorted_counts)

# Establecer una altura fija para todas las etiquetas de texto
label_height = max_count + 10  # Puedes ajustar este valor según tus preferencias

# Agregar etiquetas de texto con el recuento en la altura fija
for index, value in enumerate(sorted_counts):
    plt.text(index, label_height, str(value), ha='center', va='bottom')

# Ajustar el eje x para mostrar cada valor individualmente
plt.xticks(range(len(sorted_labels)), sorted_labels)

# Agregar etiquetas en los ejes
plt.xlabel('Clase')
plt.ylabel('Número de Instancias')

# Añadir título
plt.title('Barplot del conjunto train')

plt.savefig('instanciasPorClase.png')
plt.show()



# you can change the random_image value to visualize any other image
def visualize(i):
    some_digit = X_train.iloc[i] # select any number, change to select any number
    plt.imshow(some_digit.values.reshape(28, 28))
    plt.show()
    print('label', y_train.iloc[i])


###########################################################################################################################################################################
#########################################################           UTILIZAR KMEANS PARA CLUSTERING           ###############################################################
###########################################################################################################################################################################
'''
######################################################Redimensionar con PCA
print('Dim originally: ',X_train.shape)
# Reducir las dimensiones para visualizarlas: PCA
pca = PCA(n_components=2)
pca.fit(X_train)
# Cambio de base a dos dimensiones PCA
X_train_PCAspace = pca.transform(X_train)
print('Dim after PCA: ',X_train_PCAspace.shape)

# Aplicar KMeans al conjunto reducido
n_clusters = 36
kmeans = KMeans(n_clusters=n_clusters, n_init=36)
kmeans.fit(X_train)
kmeansLabels = kmeans.predict(X_train)
'''

#####################################################REDIMENSIONAR CON t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_train_TSNE = tsne.fit_transform(X_train)

# Aplicar KMeans al conjunto reducido
n_clusters = 21
kmeans = KMeans(n_clusters=n_clusters, n_init=21)
kmeans.fit(X_train_TSNE)
kmeansLabels = kmeans.predict(X_train_TSNE)


###########################################################################################################################################################################
#########################################################           GRAFICAR VALOR REAL VS PREDECIDO           ###############################################################
###########################################################################################################################################################################

#############################################################################Graficar 300 instancias con su clase real y predecida
'''
# Dibujar sólo unas pocas instancias
samples = 300
# Dibujar los puntos en el espacio, color: cluster, etiqueta-numérica: clase
# Color del punto: cluster
sc = plt.scatter(X_train_PCAspace[:samples, 0], X_train_PCAspace[:samples, 1],
                 cmap=plt.get_cmap('nipy_spectral', 10), c=kmeansLabels[:samples])
plt.colorbar()
# Etiqueta numérica: clase
for i in range(samples):
    plt.text(X_train_PCAspace[i, 0], X_train_PCAspace[i, 1], y_train[i])
plt.savefig('mapaDeColores.png')
plt.title('Clustering en el espacio de PCA')
plt.show()
'''

# Dibujar sólo unas pocas instancias (las primeras 300)
plt.scatter(X_train_TSNE[:300, 0], X_train_TSNE[:300, 1], c=kmeansLabels[:300], cmap='nipy_spectral')
plt.colorbar()
plt.title('Clustering en espacio t-SNE')
plt.savefig('cluster_tsne.png')
plt.show()


######################################## Calcular la matriz de confusión##########################################################3
confusion = confusion_matrix(y_train, kmeansLabels)

# Visualizar la matriz de confusión con el número de instancias
# El atributo generado por K-means es int, hay que pasarlos a string
to_string = lambda x : str(x)
# Obtener matriz de confusión Class to clustering eval
cm = confusion_matrix(np.vectorize(to_string)(kmeansLabels), np.vectorize(to_string)(y_train))
# Mapa de calor a partir de la matriz de confusion sin números
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Etiqueta Real')
plt.ylabel('Cluster')
plt.title('Matriz de Confusión sin Números')
plt.savefig('matriz_confusionSinLabels.png')
plt.show()


####################################################################################33Calcular las inercias para obtener el codo
'''
inertia = []
for n_clusters in range(1, 37):  # Prueba números de clusters de 1 a 36
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)
    print("Calculando la inercia del cluster " + str(n_clusters))
    print(kmeans.inertia_)

# Graficar la inercia en función del número de clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 37), inertia, marker='o', linestyle='--')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.savefig('codoPorInercia_36_clusters.png')  # Cambiar el nombre del archivo
plt.show()
'''



###########################################################################################################################################################################
#########################################################           RENOMBRAR LABELS PARA CONSEGUIR MATRIZ DIAGONAL           ###############################################################
###########################################################################################################################################################################

# Reassign cluster labels based on the most common true labels within each cluster
cluster_to_class = {}
# Reasignar etiquetas de clúster basadas en las etiquetas verdaderas más comunes dentro de cada clúster
cluster_to_class = {}
for cluster_id in range(n_clusters):
    cluster_indices = np.where(kmeansLabels == cluster_id)[0]
    true_labels = y_train.iloc[cluster_indices].values
    most_common_label = np.bincount(true_labels).argmax()
    cluster_to_class[cluster_id] = most_common_label

# Mapear las etiquetas de clúster a etiquetas de clase
reassigned_labels = np.vectorize(cluster_to_class.get)(kmeansLabels)


# Calculate the confusion matrix with the reassigned labels
cm = confusion_matrix(y_train, reassigned_labels)
total_correct = np.trace(cm)  # Suma de valores en la diagonal principal
total_samples = np.sum(cm)    # Suma de todos los valores en la matriz de confusión

# Calcular el número total de clasificaciones incorrectas
total_incorrect = total_samples - total_correct

# Calcular la tasa de error
error_rate = total_incorrect / total_samples

print("Tasa de Error:", error_rate)


# Create a heatmap
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de confusión con las etiquetas ajustadas")
plt.xlabel('Etiqueta Real')
plt.ylabel('Cluster')
plt.savefig('matriz_confusionConLabels.png')
plt.show()

from sklearn.metrics import silhouette_score, calinski_harabasz_score
n_components_list = [2, 3, 5, 10, 15, 20]
silhouette_scores = []
calinski_harabasz_scores = []

for n_components in n_components_list:
    # Realiza PCA con el número de componentes especificado
    pca = PCA(n_components=n_components)
    X_train_PCAspace = pca.fit_transform(X_train)

    # Realiza KMeans en el conjunto reducido
    n_clusters = 36  # Número de clusters para KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=36)
    kmeans.fit(X_train_PCAspace)  # X_train_PCAspace es la representación PCA

    # Calcula las métricas de calidad
    silhouette_avg = silhouette_score(X_train_PCAspace, kmeans.labels_)
    calinski_harabasz_avg = calinski_harabasz_score(X_train_PCAspace, kmeans.labels_)

    # Almacena las métricas en las listas
    silhouette_scores.append(silhouette_avg)
    calinski_harabasz_scores.append(calinski_harabasz_avg)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(n_components_list, silhouette_scores, marker='o', label='Silueta')
plt.plot(n_components_list, calinski_harabasz_scores, marker='o', label='Calinski-Harabasz')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Puntuación')
plt.title('Puntuación de Silueta y Calinski-Harabasz vs. Número de Componentes Principales')
plt.legend()
plt.grid(True)
plt.show()
