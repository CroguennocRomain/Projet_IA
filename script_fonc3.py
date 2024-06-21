import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import json

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Fonction principale pour prédire si un arbre survie à une tempête en utilisant la méthode spécifiée
def predire_tempete(method):
    # Configure pandas pour éviter les avertissements sur le downcasting futur
    pd.set_option('future.no_silent_downcasting', True)

    # Lecture des données depuis un fichier CSV
    data = pd.read_csv('Data_Arbre.csv')

    # Si la méthode est '0' et que le nombre d'arguments est correct
    if method == '0' and len(sys.argv) == 8:
        # Création d'un nouveau DataFrame avec les données fournies en argument
        new_data = {
            'haut_tronc': [float(sys.argv[1])],
            'latitude': [float(sys.argv[2])],
            'longitude': [float(sys.argv[3])],
            'fk_stadedev': [sys.argv[4]],
            'haut_tot': [float(sys.argv[5])],
            'clc_secteur': [sys.argv[6]]
        }
        new_data_df = pd.DataFrame(new_data)

        # Ajouter les colonnes manquantes du jeu de données original
        for colonne in data.columns:
            if colonne not in new_data_df.columns:
                new_data_df[colonne] = data[colonne][0]

        # Réordonner les colonnes pour correspondre à l'ordre du jeu de données original
        new_data_df = new_data_df[data.columns]

        # Encoder les colonnes catégorielles de la nouvelle ligne de données
        categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']
        with open('OrdinalEncoder/ordinal_encoder3.pkl', 'rb') as file:
            encoder = pickle.load(file)
        new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

        # Sélectionner les colonnes nécessaires pour le modèle
        X = new_data_df[["haut_tronc", "latitude", "longitude", 'fk_stadedev', 'haut_tot', 'clc_secteur']]

        # Charger le scaler et transformer les données
        with open('Scaler/scaler3.pkl', 'rb') as file:
            model = pickle.load(file)
        X = model.transform(X)
        print(X)
        model_filename = 'models/rf_model.pkl'

    # Si la méthode est '1' et que le nombre d'arguments est correct
    elif method == '1' and len(sys.argv) == 6:
        new_data = {
            'latitude': [float(sys.argv[1])],
            'longitude': [float(sys.argv[2])],
            'clc_secteur': [sys.argv[3]],
            'fk_port': [sys.argv[4]]
        }
        new_data_df = pd.DataFrame(new_data)
        for colonne in data.columns:
            if colonne not in new_data_df.columns:
                new_data_df[colonne] = data[colonne][0]
        new_data_df = new_data_df[data.columns]

        # Encoder les colonnes catégorielles de la nouvelle ligne de données
        categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']
        with open('OrdinalEncoder/ordinal_encoder3.pkl', 'rb') as file:
            encoder = pickle.load(file)
        new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

        # Sélectionner les colonnes nécessaires pour le modèle
        X = new_data_df[["latitude", "longitude", "clc_secteur", 'fk_port']]

        # Charger le scaler et transformer les données
        with open('Scaler/scaler3.pkl', 'rb') as file:
            model = pickle.load(file)
        X = model.transform(X)

        model_filename = 'models/knn_model.pkl'

    # Si la méthode est '2' et que le nombre d'arguments est correct
    elif method == '2' and len(sys.argv) == 4:
        new_data = {
            'age_estim': [float(sys.argv[1])]
        }
        new_data_df = pd.DataFrame(new_data)
        for colonne in data.columns:
            if colonne not in new_data_df.columns:
                new_data_df[colonne] = data[colonne][0]
        new_data_df = new_data_df[data.columns]

        # Encoder les colonnes catégorielles de la nouvelle ligne de données
        categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']
        with open('OrdinalEncoder/ordinal_encoder3.pkl', 'rb') as file:
            encoder = pickle.load(file)
        new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

        # Sélectionner les colonnes nécessaires pour le modèle
        X = new_data_df[['haut_tot','fk_revetement']]

        # Charger le scaler et transformer les données
        with open('Scaler/scaler3.pkl', 'rb') as file:
            model = pickle.load(file)
        X = model.transform(X)

        model_filename = 'models/svm_model.pkl'
    else:
        raise ValueError("Invalid method or number of arguments")

    # Charger le modèle approprié et faire des prédictions
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X)

    # Convertir les résultats en JSON
    res = y_pred.tolist()
    json_data = json.dumps(res)

    with open('JSON/script2_result.json', 'w') as f:
        json.dump(res, f)

    return json_data


# Fonction principale
def main():
    method = sys.argv[-1]
    tempete = predire_tempete(method)
    print(tempete)
    return tempete

if __name__ == "__main__":
    main()
