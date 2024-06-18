# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                 IMPORTATIONS                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

'''
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("plotly")
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.image as mpimg

data = pd.read_csv('Data_Arbre.csv')

'''
# Assurer que les colonnes 'latitude' et 'longitude' sont de type float et ne contiennent pas de valeurs manquantes
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
data = data.dropna(subset=['latitude', 'longitude'])
'''

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃            PREPARATION DES DONNEES            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Enlever les colonnes inutiles
data = data.drop(['clc_nbr_diag'],axis=1)

# Créer le LabelEncoder
labE = LabelEncoder()

# Appliquer le LabelEncoder à chaque colonne à convertir et convertir object en float64
for colonne in data:
    if data[colonne].dtype.name == 'object':
        data[colonne] = labE.fit_transform(data[colonne].astype(str)).astype('float64')


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃          APPRENTISSAGE NON SUPERVISE          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Extraire les données d'intérêt pour K-means (toutes les colonnes sauf latitude, longitude et haut_tot)
X = data.drop(['latitude', 'longitude', 'haut_tot'], axis=1).values
#X = data[['latitude', 'longitude']].values  # Latitude et longitude pour les coordonnées

# Extraire les données de la colonne 'haut_tot'
Y = data['haut_tot'].copy().values  # Copie de la colonne haut_tot pour les couleurs


# Fonction pour appliquer K-means et afficher les résultats
def apply_kmeans(data, X, n_clusters):
    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    y_pred = kmeans.fit_predict(X)

    # Ajouter les labels des clusters au DataFrame original
    data['cluster'] = y_pred
    data["cluster"] = data["cluster"]+1

    # Calcul des métriques de performance
    silhouette_avg = silhouette_score(X, y_pred)
    calinski_harabasz = calinski_harabasz_score(X, y_pred)
    davies_bouldin = davies_bouldin_score(X, y_pred)
    NMI = normalized_mutual_info_score(Y.astype(np.uint8), y_pred)

    print(f"Silhouette Score: {silhouette_avg}"," pire cas = -1 et meilleur cas = 1")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}"," pire cas = score faible et meilleur cas = score élevé")
    print(f"Davies-Bouldin Index: {davies_bouldin}"," pire cas = score élevé et meilleur cas = score faible")
    print(f"NMI (Normalized Mutual Index): {NMI}"," pire cas = 0 et meilleur cas = 1")

    return data, kmeans

# Spécifier le nombre de clusters
def demander_nombre_clusters():
    while True:
        try:
            n_clusters = int(input("Veuillez entrer un nombre de clusters entre 2 et 10: "))
            if 2 <= n_clusters <= 10:
                return n_clusters
            else:
                print("Le nombre de clusters doit être entre 2 et 10.")
        except ValueError:
            print("Veuillez entrer un nombre entier valide.")

# Utiliser la fonction pour obtenir le nombre de clusters
n_clusters = demander_nombre_clusters()
print(f"Nombre de clusters sélectionné: {n_clusters}")

# Appliquer K-means
data_with_clusters, kmeans_model = apply_kmeans(data.copy(), X, n_clusters)

# Afficher les résultats
print(data_with_clusters)
print(data_with_clusters['cluster'].value_counts())

# test nombre de cluster


# Déterminer l'inertie pour différents nombres de clusters
inertia = []
range_n_clusters = range(2, 11)  # Essayer de 1 à 10 clusters

for iter_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=iter_clusters, random_state=0).fit(X)
    inertia.append(kmeans.inertia_)

# Tracer la courbe du coude
plt.figure(figsize=(8, 6))
plt.plot(range_n_clusters, inertia, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.show()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   METRIQUE POUR APPRENTISSAGE NON SUPERVISE   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Définir la plage de nombre de clusters à tester
range_n_clusters = range(2, 11)  # On commence à 2 clusters car 1 cluster n'a pas de sens pour les métriques

# Initialiser les listes pour stocker les scores des différentes métriques
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []

# Appliquer K-means et calculer les métriques pour chaque nombre de clusters
for iter_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=iter_clusters, random_state=0).fit(X)
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


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃            VISUALISATION SUR CARTE            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Définir la palette de couleurs personnalisée avec les couleurs spécifiées
custom_colors = ['red', 'yellow', 'green', 'navy', 'black', 'purple', 'orange', 'cyan', 'brown', 'lightblue']

# Utilisation de Plotly Express pour tracer les arbres sur une carte avec Mapbox (OpenStreetMap)
fig = px.scatter_mapbox(data_with_clusters,
                        lat='latitude',
                        lon='longitude',
                        color='cluster',  # Utiliser la colonne 'cluster' pour la couleur
                        hover_name='fk_nomtech',  # Nom à afficher au survol
                        hover_data=['haut_tot'],  # Données supplémentaires au survol
                        zoom=10,  # Niveau de zoom initial de la carte
                        mapbox_style='open-street-map',  # Utiliser le style de carte OpenStreetMap
                        color_discrete_sequence=custom_colors,  # Utiliser la palette de couleurs personnalisée
                        opacity=0.8,  # Opacité des points
                        title='Représentation des arbres par clusters')  # Titre de la carte

# Affichage de la carte interactive
fig.show()


'''

mapbox_token = 'your_mapbox_token'

# Utilisation de Plotly Express pour tracer les arbres sur une carte avec Mapbox
fig = px.scatter_mapbox(data_with_clusters,
                        lat='latitude',
                        lon='longitude',
                        color='cluster',  # Utiliser la colonne 'cluster' pour la couleur
                        hover_name='nom',  # Nom à afficher au survol
                        hover_data=['haut_tot'],  # Données supplémentaires au survol
                        zoom=10,  # Niveau de zoom initial de la carte
                        mapbox_style='open-street-map',  # Style de la carte Mapbox (open-street-map, carto-positron, etc.)
                        color_continuous_scale=px.colors.qualitative.Vivid,  # Échelle de couleurs
                        opacity=0.8,  # Opacité des points
                        title='Représentation des arbres par clusters')  # Titre de la carte

# Mise à jour du token Mapbox pour afficher la carte
fig.update_layout(mapbox={'accesstoken': mapbox_token})

# Affichage de la carte interactive
fig.show()
'''

'''

plt.scatter(data_with_clusters['haut_tot'], [0] * len(data_with_clusters), c=data_with_clusters['cluster'], cmap='viridis')
plt.xlabel('haut_tot')
plt.title('Clustering K-means (n_clusters={})'.format(n_clusters))
plt.show()

data["cluster"] = data["cluster"]+1

'''
# Charger la carte de Saint Quentin
map_img = mpimg.imread('saint_quentin_map.png')

# Définir les limites de la carte (ajuster en fonction de votre image et données)
min_lat, max_lat = 49.82, 49.871
min_lon, max_lon = 3.2375, 3.325

# Tracer les points des arbres sur la carte avec des couleurs différentes pour chaque cluster
plt.figure(figsize=(10, 10))
plt.imshow(map_img, extent=[min_lon, max_lon, min_lat, max_lat])

# Tracer chaque cluster avec une couleur différente
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
for cluster in range(1,n_clusters+1):
    clustered_data = data_with_clusters[data_with_clusters['cluster'] == cluster]
    plt.scatter(clustered_data['longitude'], clustered_data['latitude'], color=colors[cluster % len(colors)], label=f'Cluster {cluster}', alpha=0.6)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustering des arbres à Saint Quentin selon la hauteur totale')
plt.legend()
plt.show()



# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃             PREPARATION DE SCRIPT             ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛






# Transformer une image jpeg en png
#from PIL import Image

# Charger l'image JPEG
#img = Image.open('saint-quentin.jpg')

# Sauvegarder en PNG
#img.save('saint_quentin_map.png')
