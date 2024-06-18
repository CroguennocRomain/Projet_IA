import sys
import json
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def predire_age(features, method):
    data = pd.read_csv("Data_Arbre.csv")

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
    else:
        model = KNeighborsClassifier(n_neighbors=3)
    # Application du modèle à nos données
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    json_data = json.dumps(y_pred.tolist())

    return json_data


# On récupère les arguments noms de colonnes dans features
features = []
# On récupère l'indice du dernier argument corresponndant au modèle
i_last_arg = len(sys.argv) - 1  # -1 car on ne compte pas argv[0] qui est le nom du script
for i in range(1, i_last_arg):
    features.append(sys.argv[i])


age = predire_age(features, sys.argv[i_last_arg])
print(age)
