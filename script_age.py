import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
import pickle


def predire_age(values, method):

    # associer noms de colonnes et valeurs dans une structure json
    data = {}
    if len(values) == 5:
        data['haut_tot'] = values[0]
        data['haut_tronc'] = values[1]
        data['tronc_diam'] = values[2]
        data['fk_stadedev'] = values[3]
        data['nomfrancais'] = values[4]
    else:
        print("Erreur : Il faut entrer 6 arguments")

    # Sélection du modèle d'apprentissage
    if method == '0':
        with open("age_SGD.pkl", "rb") as f:
            model = pickle.load(f)
    elif method == '1':
        with open("age_neigh.pkl", "rb") as f:
            model = pickle.load(f)
    elif method == '2':
        with open("age_SVM.pkl", "rb") as f:
            model = pickle.load(f)
    elif method == '3':
        with open("age_tree.pkl", "rb") as f:
            model = pickle.load(f)

    # Créer l'instance au bon format
    arbre = np.array([[int(data["haut_tot"]), int(data["haut_tronc"]), int(data["tronc_diam"]), int(data["fk_stadedev"]), int(data["nomfrancais"])]])

    # Normalisation
    scaler = StandardScaler()
    arbre_norm = scaler.fit_transform(arbre)

    # Proba de chaque classe
    classes = model.predict_proba(arbre_norm)

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
