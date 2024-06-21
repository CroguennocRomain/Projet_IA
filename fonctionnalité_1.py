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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.image as mpimg
import pickle

data = pd.read_csv('Data_Arbre.csv')
data_original = data.copy()

'''
# Assurer que les colonnes 'latitude' et 'longitude' sont de type float et ne contiennent pas de valeurs manquantes
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
data = data.dropna(subset=['latitude', 'longitude'])
'''

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃            PREPARATION DES DONNEES            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''
# Instancier l'OrdinalEncoder
encoder = OrdinalEncoder()

for colonne in data:
    if data[colonne].dtype.name == 'object':
        data[colonne] = encoder.fit_transform(data[[colonne]])
'''

# Sélectionner les colonnes
categorical_columns = [colonne for colonne in data if data[colonne].dtype.name == 'object']

# Créer et utiliser l'OrdinalEncoder
encoder = OrdinalEncoder()
data[categorical_columns] = encoder.fit_transform(data[categorical_columns])

# Enregistrer l'encodeur dans un fichier
with open('OrdinalEncoder/ordinal_encoder1.pkl', 'wb') as file:
    pickle.dump(encoder, file)

# numériser les données

scaler = StandardScaler()
data_norm = scaler.fit_transform(data)
data_norm = pd.DataFrame(data_norm, columns=data.columns)

with open('Scaler/scaler1.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃          APPRENTISSAGE NON SUPERVISE          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Extraire les données d'intérêt pour K-means
X = data_norm[['haut_tot', 'haut_tronc', 'fk_stadedev', 'fk_nomtech', 'feuillage']]


# ======================Fonction pour appliquer K-means et afficher les résultats=========================
def apply_kmeans(data_norm, X, n_clusters):
    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    y_pred = kmeans.predict(X)

    # Ajouter les labels des clusters au DataFrame original
    data_norm['cluster'] = y_pred

    # Calcul des métriques de performance
    silhouette_avg = silhouette_score(X, y_pred)
    calinski_harabasz = calinski_harabasz_score(X, y_pred)
    davies_bouldin = davies_bouldin_score(X, y_pred)

    print("K-mean method pour n_clusters = ", n_clusters)
    print(f"Silhouette Score: {silhouette_avg}"," pire cas = -1 et meilleur cas = 1")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}"," pire cas = score faible et meilleur cas = score élevé")
    print(f"Davies-Bouldin Index: {davies_bouldin}"," pire cas = score élevé et meilleur cas = score faible")

    return data_norm, kmeans

# ======================Fonction pour appliquer agglomerative_clustering et afficher les résultats=========================
def apply_agglomerative_clustering(data_norm, X, n_clusters):
    # Appliquer l'Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    y_pred = agg_clustering.labels_

    # Ajouter les labels des clusters au DataFrame original
    data_norm['cluster'] = y_pred

    # Calcul des métriques de performance
    silhouette_avg = silhouette_score(X, y_pred)
    calinski_harabasz = calinski_harabasz_score(X, y_pred)
    davies_bouldin = davies_bouldin_score(X, y_pred)

    print(" Agglomerative_clustering method pour n_clusters = ", n_clusters)
    print(f"Silhouette Score: {silhouette_avg} (pire cas = -1 et meilleur cas = 1)")
    print(f"Calinski-Harabasz Index: {calinski_harabasz} (pire cas = score faible et meilleur cas = score élevé)")
    print(f"Davies-Bouldin Index: {davies_bouldin} (pire cas = score élevé et meilleur cas = score faible)")

    return data_norm, agg_clustering

# Spécifier le nombre de clusters
def demander_nombre_clusters():
    while True:
        try:
            n_clusters = int(input("Veuillez entrer un nombre de clusters entre 2 et 15: "))
            if 2 <= n_clusters <= 15:
                return n_clusters
            else:
                print("Le nombre de clusters doit être entre 2 et 15.")
        except ValueError:
            print("Veuillez entrer un nombre entier valide.")

# Utiliser la fonction pour obtenir le nombre de clusters
n_clusters = demander_nombre_clusters()
print(f"Nombre de clusters sélectionné: {n_clusters}")

# Appliquer K-means
data_with_clusters, kmeans_model = apply_kmeans(data_norm.copy(), X, n_clusters)
# Appliquer Agglomerative Clustering
data_agglo_cluster_method, agg_clustering = apply_agglomerative_clustering(data_norm.copy(), X, n_clusters)



# Afficher les résultats
print(data_with_clusters)
print(data_with_clusters['cluster'].value_counts())
print(data_agglo_cluster_method)
print(data_agglo_cluster_method['cluster'].value_counts())

# =====================test nombre de cluster==========================
# Déterminer l'inertie pour différents nombres de clusters
inertia = []
range_n_clusters = range(2, 16)  # Essayer de 2 à 15 clusters

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

#====================== créer fichier centroide cluster ================================

# obtenir les centroids.
centroids = kmeans_model.cluster_centers_
print(centroids)

centroids_df = pd.DataFrame(centroids, columns=[f'feature_{i}' for i in range(centroids.shape[1])])


#swap les centroids et les valeur de cluster associé pour que le cluster 0 ait le plus petit centroids pour 'haut_tot' et que le cluster 6 ait le plus grand
# Boucle pour échanger les valeurs entre l'index 0 et les autres indices
for i in range(len(centroids_df['feature_0']) - 1):
    # Trouver la valeur maximale de la colonne entre l'index 0 et l'index len(centroids_df['feature_0'])-1 - i
    max_value = centroids_df['feature_0'][:len(centroids_df['feature_0']) - 1 - i].max()

    # Trouver la dernière valeur de la colonne pour l'index len(centroids_df['feature_0'])-1 - i
    last_value = centroids_df['feature_0'].iloc[-1 - i]
    print(last_value, max_value)

    # Vérifier si la valeur maximale est supérieure à la dernière valeur
    if max_value > last_value:
        # Trouver l'index de la valeur maximale
        max_index = centroids_df['feature_0'][:len(centroids_df['feature_0']) - 1 - i].idxmax()
        print(centroids_df.index[-1 - i], max_index)

        # Échanger les lignes complètes dans centroids_df
        max_row = centroids_df.iloc[max_index].copy()
        last_row = centroids_df.iloc[centroids_df.index[-1 - i]].copy()

        centroids_df.iloc[max_index] = last_row
        centroids_df.iloc[centroids_df.index[-1 - i]] = max_row

        # Échanger les valeurs de cluster dans data_with_clusters
        data_with_clusters['cluster'] = data_with_clusters['cluster'].replace(
            {max_index: centroids_df.index[-1 - i], centroids_df.index[-1 - i]: max_index})

# Afficher les DataFrames modifiés
print("Centroids DataFrame après échange :")
print(centroids_df)

print("\nData_with_clusters DataFrame après échange des clusters :")
print(data_with_clusters)


centroids_df.to_csv('centroids.csv', index=False)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   METRIQUE POUR APPRENTISSAGE NON SUPERVISE   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

#========================K-means methode============================
# Définir la plage de nombre de clusters à tester
range_n_clusters = range(2, 16)  # On commence à 2 clusters car 1 cluster n'a pas de sens pour les métriques

# Initialiser les listes pour stocker les scores des différentes métriques
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []

# Appliquer K-means et calculer les métriques pour chaque nombre de clusters
for iter_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=iter_clusters, random_state=0).fit(X)
    labels = kmeans.predict(X)

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


#========================Agglomerative Clustering methode============================
# Initialiser les listes pour stocker les scores des différentes métriques
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []

# Appliquer Agglomerative_Clustering et calculer les métriques pour chaque nombre de clusters
for iter_clusters in range_n_clusters:
    agg_clustering = AgglomerativeClustering(n_clusters=iter_clusters).fit(X)
    labels = agg_clustering.labels_

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

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃            VISUALISATION SUR CARTE            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

#mettre les clusters sous forme textuel
data_with_clusters['cluster'] = data_with_clusters['cluster'].astype(str)
data_with_clusters_non_norm = data_original
data_with_clusters_non_norm['cluster'] = data_with_clusters['cluster']

# Définir la palette de couleurs personnalisée avec les couleurs spécifiées
custom_colors = ['red', 'yellow', 'green', 'navy', 'black', 'purple', 'orange', 'cyan', 'brown', 'lightblue']

# Utilisation de Plotly Express pour tracer les arbres sur une carte avec Mapbox (OpenStreetMap)
fig = px.scatter_mapbox(data_with_clusters_non_norm,
                        lat='latitude',
                        lon='longitude',
                        color='cluster',  # Utiliser la colonne 'cluster' pour la couleur
                        hover_name='fk_nomtech',  # Nom à afficher au survol
                        hover_data=['haut_tot','haut_tronc','fk_stadedev', 'feuillage'],  # Données supplémentaires au survol
                        zoom=10,  # Niveau de zoom initial de la carte
                        mapbox_style='open-street-map',  # Utiliser le style de carte OpenStreetMap
                        color_discrete_sequence=custom_colors,  # Utiliser la palette de couleurs personnalisée
                        opacity=0.8,  # Opacité des points
                        title='Représentation des arbres par clusters')  # Titre de la carte

# Affichage de la carte interactive
fig.show()



# Charger la carte de Saint Quentin
map_img = mpimg.imread('saint_quentin_map.png')

# Définir les limites de la carte (ajusté en fonction de l'image et des données de longitude et latitude)
min_lat, max_lat = 49.82, 49.871
min_lon, max_lon = 3.2375, 3.325

# Convertir les clusters en int
data_with_clusters_non_norm['cluster'] = data_with_clusters_non_norm['cluster'].astype(int)

# Définir le nombre de clusters
n_clusters = data_with_clusters_non_norm['cluster'].nunique()

# Tracer les points des arbres sur la carte avec des couleurs différentes pour chaque cluster
plt.figure(figsize=(10, 10))
plt.imshow(map_img, extent=[min_lon, max_lon, min_lat, max_lat])

# Tracer chaque cluster avec une couleur différente
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
for cluster in range(n_clusters):
    clustered_data = data_with_clusters_non_norm[data_with_clusters_non_norm['cluster'] == cluster]
    plt.scatter(clustered_data['longitude'], clustered_data['latitude'], color=colors[cluster % len(colors)], label=f'Cluster {cluster}', alpha=0.6)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustering des arbres à Saint Quentin selon la hauteur totale')
plt.legend()
plt.show()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃             PREPARATION DE SCRIPT             ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

#==============prédire pour une nouvelle ligne le cluster avec model kmean===================
# Charger l'encodeur depuis le fichier
with open('ordinal_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Nouvelle ligne de données à encoder
new_data = {
    'haut_tot': [5.1],
    'haut_tronc': [2.1],
    'fk_stadedev': ['Jeune'],
    'fk_nomtech': ['PINNIGnig'],
    'feuillage': ['Conifère']
}

# Convertir en DataFrame
new_data_df = pd.DataFrame(new_data)

# Charger les données originales pour obtenir la structure complète
data = pd.read_csv('Data_Arbre.csv')

# Ajouter les colonnes manquantes avec des valeurs par défaut
for colonne in data.columns:
    if colonne not in new_data_df.columns:
        new_data_df[colonne] = data[colonne][0]

# Réorganiser les colonnes pour correspondre à l'ordre des colonnes originales
new_data_df = new_data_df[data.columns]

# Sélectionner les colonnes de la nouvelle ligne de données
categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']

# Appliquer l'encodeur sur les colonnes sélectionnées de la nouvelle ligne de données
new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

# appliquer normalisation
new_data_df = scaler.transform(new_data_df)
new_data_df = pd.DataFrame(new_data_df, columns=data.columns)

# Sélectionner les colonnes utilisées pour les centroids
features = ['haut_tot', 'haut_tronc', 'fk_stadedev', 'fk_nomtech', 'feuillage']

# Extraire les colonnes de new_data_df qui correspondent aux features utilisées pour les centroids
new_data_renamed = new_data_df[features]

# Faire la prédiction pour la nouvelle ligne
predicted_cluster = kmeans_model.predict(new_data_renamed)
print(f'La nouvelle ligne appartient au cluster {predicted_cluster[0]}')


new_data = {
    'haut_tot': [15.1],
    'haut_tronc': [2.1],
    'fk_stadedev': ['Jeune'],
    'fk_nomtech': ['ACEPLA'],
    'feuillage': ['Feuillu']
}

# Convertir en DataFrame
new_data_df = pd.DataFrame(new_data)

# Charger les données originales pour obtenir la structure complète
data = pd.read_csv('Data_Arbre.csv')

# Ajouter les colonnes manquantes avec des valeurs par défaut
for colonne in data.columns:
    if colonne not in new_data_df.columns:
        new_data_df[colonne] = data[colonne][0]

# Réorganiser les colonnes pour correspondre à l'ordre des colonnes originales
new_data_df = new_data_df[data.columns]

# Sélectionner les colonnes de la nouvelle ligne de données
categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']

# Appliquer l'encodeur sur les colonnes sélectionnées de la nouvelle ligne de données
new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

# appliquer normalisation
new_data_df = scaler.transform(new_data_df)
new_data_df = pd.DataFrame(new_data_df, columns=data.columns)

# Sélectionner les colonnes utilisées pour les centroids
features = ['haut_tot', 'haut_tronc', 'fk_stadedev', 'fk_nomtech', 'feuillage']

# Extraire les colonnes de new_data_df qui correspondent aux features utilisées pour les centroids
new_data_renamed = new_data_df[features]

# Faire la prédiction pour la nouvelle ligne
predicted_cluster = kmeans_model.predict(new_data_renamed)
print(f'La nouvelle ligne appartient au cluster {predicted_cluster[0]}')


new_data = {
    'haut_tot': [35.1],
    'haut_tronc': [12.1],
    'fk_stadedev': ['senescent'],
    'fk_nomtech': ['ACEPLA'],
    'feuillage': ['Feuillu']
}

# Convertir en DataFrame
new_data_df = pd.DataFrame(new_data)

# Charger les données originales pour obtenir la structure complète
data = pd.read_csv('Data_Arbre.csv')

# Ajouter les colonnes manquantes avec des valeurs par défaut
for colonne in data.columns:
    if colonne not in new_data_df.columns:
        new_data_df[colonne] = data[colonne][0]

# Réorganiser les colonnes pour correspondre à l'ordre des colonnes originales
new_data_df = new_data_df[data.columns]

# Sélectionner les colonnes de la nouvelle ligne de données
categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']

# Appliquer l'encodeur sur les colonnes sélectionnées de la nouvelle ligne de données
new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

# appliquer normalisation
new_data_df = scaler.transform(new_data_df)
new_data_df = pd.DataFrame(new_data_df, columns=data.columns)

# Sélectionner les colonnes utilisées pour les centroids
features = ['haut_tot', 'haut_tronc', 'fk_stadedev', 'fk_nomtech', 'feuillage']

# Extraire les colonnes de new_data_df qui correspondent aux features utilisées pour les centroids
new_data_renamed = new_data_df[features]

# Faire la prédiction pour la nouvelle ligne
predicted_cluster = kmeans_model.predict(new_data_renamed)
print(f'La nouvelle ligne appartient au cluster {predicted_cluster[0]}')


'''
# Transformer une image jpeg en png
#from PIL import Image

# Charger l'image JPEG
#img = Image.open('saint-quentin.jpg')

# Sauvegarder en PNG
#img.save('saint_quentin_map.png')

'''
