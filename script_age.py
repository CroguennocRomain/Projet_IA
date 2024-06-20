import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
import pickle


def predire_age(values, method):
    # Charger l'encodeur depuis le fichier
    with open('ordinal_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    # Colonnes utilisées lors de l'entraînement du OrdinalEncoder
    encoder_cols = encoder.feature_names_in_

    # Ajouter valeurs entrées dans un dictionnaire
    data_input = {}
    if len(values) == 5:
        data_input['haut_tot'] = values[0]
        data_input['haut_tronc'] = values[1]
        data_input['tronc_diam'] = values[2]
        data_input['fk_stadedev'] = values[3]
        data_input['fk_nomtech'] = values[4]
    else:
        print("Erreur : Il faut entrer 6 arguments")

    # Convertir le dictionnaire en DataFrame
    df = pd.DataFrame([data_input])

    data_arbre = pd.read_csv('Data_Arbre.csv')
    # Ajouter les colonnes manquantes avec des valeurs NaN
    for col in encoder_cols:
        if col not in df.columns:
            df[col] = data_arbre[col][0]

    # Réorganiser les colonnes selon l'ordre des colonnes utilisées lors de l'entraînement
    df = df[encoder_cols]

    # Remplir les valeurs manquantes avec une valeur par défaut
    df.fillna('missing_value', inplace=True)

    # Encoder les colonnes textuelles en numériques
    df[encoder_cols] = encoder.transform(df[encoder_cols])

    # Supprimer les colonnes qui ne nous intéressent pas
    df.drop(columns=['clc_quartier', 'clc_secteur', 'fk_arb_etat', 'fk_port',
                     'fk_pied', 'fk_situation', 'fk_revetement', 'villeca',
                     'feuillage', 'remarquable'], inplace=True)

    # Ajouter les colonnes non encodées
    for col in ['haut_tot', 'haut_tronc', 'tronc_diam']:
        df[col] = data_input[col]


    # Mettre notre instance à prédire sous le bon format
    arbre = np.array([[float(df['haut_tot'][0]), float(df['haut_tronc'][0]), float(df['tronc_diam'][0]), float(df['fk_stadedev'][0]), float(df['fk_nomtech'][0])]])
    print(arbre)

    # Normalisation
    """
    with open("normalisation.pkl", "rb") as f:
        scaler = pickle.load(f)
    df_norm = scaler.fit_transform(df)
    print(df_norm)
    """

    # Sélection du modèle d'apprentissage
    if method == '0':
        with open("models/age_SGD.pkl", "rb") as f:
            model = pickle.load(f)
    elif method == '1':
        with open("models/age_neigh.pkl", "rb") as f:
            model = pickle.load(f)
    elif method == '2':
        with open("models/age_SVM.pkl", "rb") as f:
            model = pickle.load(f)
    elif method == '3':
        with open("models/age_tree.pkl", "rb") as f:
            model = pickle.load(f)

    # Proba de chaque classe
    classes = model.predict_proba(arbre)

    # Créer structure json
    json_data = {}
    json_data['0-10'] = classes[0][0]
    json_data['11-20'] = classes[0][1]
    json_data['21-30'] = classes[0][2]
    json_data['31-40'] = classes[0][3]
    json_data['41-50'] = classes[0][4]
    json_data['51-100'] = classes[0][5]
    json_data['101-200'] = classes[0][6]
    
    # Renvoie les données en format json
    return json.dumps(json_data)


# On récupère les arguments noms de colonnes dans features
features = []
# On récupère l'indice du dernier argument correspondant au modèle
i_last_arg = len(sys.argv) - 1  # -1 car on ne compte pas argv[0] qui est le nom du script
for i in range(1, i_last_arg):
    features.append(sys.argv[i])


age = predire_age(features, sys.argv[i_last_arg])
print(age)
