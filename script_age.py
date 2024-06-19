import sys
import json
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree


def predire_age(features, method):
    data = pd.read_csv("export_IA.csv")

    # Définir des intervalles d'âge dans une nouvelle colonne 'age_group'
    bins = [0, 10, 20, 30, 40, 50, 100, 200]
    labels = [0, 1, 2, 3, 4, 5, 6]
    data['age_group'] = pd.cut(data['age_estim'], bins=bins, labels=labels, right=True)
    data = data.dropna()  # supprimer les lignes NaN

    # Séparation des features choisies X et des labels y
    X = data[features]
    y = data['age_group']

    # Répartition des données : 80% apprentissage, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Sélection du modèle d'apprentissage
    if method == '0':
        model = SGDClassifier()
    elif method == '1':
        model = KNeighborsClassifier(n_neighbors=3)
    elif method == '2':
        model = svm.SVC()
    elif method == '3':
        model = tree.DecisionTreeClassifier()
    # Application du modèle à nos données
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    json_data = json.dumps(y_pred.tolist())

    return json_data


def predire_age2(values, method):

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
        model = SGDClassifier()
    elif method == '1':
        model = KNeighborsClassifier(n_neighbors=3)
    elif method == '2':
        model = svm.SVC()
    elif method == '3':
        model = tree.DecisionTreeClassifier()

    print(data)
    print(model)

    return 0


# On récupère les arguments noms de colonnes dans features
features = []
# On récupère l'indice du dernier argument correspondant au modèle
i_last_arg = len(sys.argv) - 1  # -1 car on ne compte pas argv[0] qui est le nom du script
for i in range(1, i_last_arg):
    features.append(sys.argv[i])


predire_age2(features, sys.argv[i_last_arg])
#print(age)
