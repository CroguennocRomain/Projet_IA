#install panadas, scikit-learn, sklearn

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("plotly")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
import plotly




data = pd.read_csv('Data_Arbre.csv')

data.info(max)

#1. Préparation des données

data = data.drop(['clc_nbr_diag'],axis=1)
data.info(max)

# Afficher les types de colonnes avant la conversion
print("Types de colonnes avant la conversion :")
print(data.dtypes)
print(data.value_counts('feuillage'))

# Instancier le LabelEncoder
le = LabelEncoder()

# Liste des colonnes à convertir
colonnes_a_convertir = [
    'clc_quartier', 'clc_secteur', 'fk_arb_etat', 'fk_stadedev',
    'fk_port', 'fk_pied', 'fk_situation', 'fk_revetement',
    'fk_nomtech', 'villeca', 'feuillage', 'remarquable'
]

# Appliquer le LabelEncoder à chaque colonne et convertir en float64
for colonne in colonnes_a_convertir:
    if colonne in data.columns:
        data[colonne] = le.fit_transform(data[colonne].astype(str)).astype('float64')

# Afficher les types de colonnes après la conversion
print("\nTypes de colonnes après la conversion :")
print(data.dtypes)

# Afficher les premières lignes du DataFrame mis à jour
print("\nDataFrame mis à jour :")
print(data.head())
print(data.value_counts('feuillage'))
print(data.value_counts('haut_tot'))




#2. Apprentissage non supervisé

# Extraire les données de la colonne 'haut_tot'
X = data[['haut_tot']].values


# Fonction pour appliquer K-means et afficher les résultats
def apply_kmeans(data, n_clusters):
    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

    # Convertir le tableau NumPy en DataFrame
    data_with_clusters = pd.DataFrame(data, columns=['haut_tot'])

    # Ajouter les labels des clusters au DataFrame
    data_with_clusters['cluster'] = kmeans.labels_

    return data_with_clusters, kmeans

# Spécifier le nombre de clusters
n_clusters = 4  # Par exemple, 3 clusters

# Appliquer K-means
data_with_clusters, kmeans_model = apply_kmeans(X, n_clusters)

# Afficher les résultats
print(data_with_clusters)
print(data_with_clusters.value_counts('cluster'))



# test nombre de cluster


# Déterminer l'inertie pour différents nombres de clusters
inertia = []
range_n_clusters = range(2, 12)  # Essayer de 1 à 10 clusters

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    inertia.append(kmeans.inertia_)

# Tracer la courbe du coude
plt.figure(figsize=(8, 6))
plt.plot(range_n_clusters, inertia, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.show()


#3. Métrique pour apprentissage non supervisé

# Définir la plage de nombre de clusters à tester
range_n_clusters = range(2, 11)  # On commence à 2 clusters car 1 cluster n'a pas de sens pour les métriques

# Initialiser les listes pour stocker les scores des différentes métriques
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []

# Appliquer K-means et calculer les métriques pour chaque nombre de clusters
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_

    silhouette_scores.append(silhouette_score(X, labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
    davies_bouldin_scores.append(davies_bouldin_score(X, labels))

# Tracer les scores des différentes métriques
plt.figure(figsize=(16, 5))

# Tracé du coefficient de silhouette
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Coefficient de silhouette')
plt.title('Coefficient de silhouette')

# Tracé de l'indice de Calinski-Harabasz
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, calinski_harabasz_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Indice de Calinski-Harabasz')
plt.title('Indice de Calinski-Harabasz')

# Tracé de l'indice de Davies-Bouldin
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, davies_bouldin_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Indice de Davies-Bouldin')
plt.title('Indice de Davies-Bouldin')

plt.tight_layout()
plt.show()

# Afficher les scores dans un tableau comparatif
metrics_table = pd.DataFrame({
    'Nombre de clusters': range_n_clusters,
    'Coefficient de silhouette': silhouette_scores,
    'Indice de Calinski-Harabasz': calinski_harabasz_scores,
    'Indice de Davies-Bouldin': davies_bouldin_scores
})

print(metrics_table)

#4. Visualisation sur carte

plt.scatter(data_with_clusters['haut_tot'], [0] * len(data_with_clusters), c=data_with_clusters['cluster'], cmap='viridis')
plt.xlabel('haut_tot')
plt.title('Clustering K-means (n_clusters={})'.format(n_clusters))
plt.show()

#5. Préparation de script


from sklearn import preprocessing
