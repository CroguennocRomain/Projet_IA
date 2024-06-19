import sys
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle

def load_data(data_file):
    data = pd.read_csv(data_file)
    return data


def load_centroids(centroids_file):
    centroids_data = pd.read_csv(centroids_file)
    return centroids_data

def main():
    if len(sys.argv) != 5:
        print('Usage: python script_fonc1 <haut_tot> <haut_tronc> <fk_stadedev> <fk_nomtech> ----> Exemple: python script_fonc1 11.4 7.3 "EN PLACE" "QUERUB"')
        sys.exit(1)

    # Load data and centroids
    data = load_data('Data_Arbre.csv')
    centroids = load_centroids('centroids.csv')

    with open('ordinal_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    new_data = {
        'haut_tot': [15],
        'haut_tronc': [2],
        'fk_stadedev': ['Jeune'],
        'fk_nomtech': ['ACEPLA']
    }

    # Convertir en DataFrame
    new_data_df = pd.DataFrame(new_data)

    print(new_data_df)
    # Sélectionner les colonnes catégorielles de la nouvelle ligne de données
    categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype.name == 'object']

    # Appliquer l'encodeur sur les colonnes catégorielles de la nouvelle ligne de données
    new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

    print(new_data_df)


'''
    # Select the specified features
    #X = data[features].values

    # Create a KMeans instance with the specified centroids
    kmeans = KMeans(init=centroids, n_clusters=len(centroids), n_init=1)

    # Fit the model and predict the clusters
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    # Add the cluster labels to the dataframe
    data['cluster'] = clusters

    # Convert the result to JSON
    result = data.to_json(orient='records')

    # Print the result
    print(result)
'''

#if __name__ == '__main__':
#    main()


# Charger l'encodeur depuis le fichier
with open('ordinal_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Nouvelle ligne de données à encoder
new_data = {
    'haut_tot': [15],
    'haut_tronc': [2],
    'fk_stadedev': ['Adulte'],
    'fk_nomtech': ['PINNIGnig']
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

# Sélectionner les colonnes catégorielles de la nouvelle ligne de données
categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']

# Appliquer l'encodeur sur les colonnes catégorielles de la nouvelle ligne de données
new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

