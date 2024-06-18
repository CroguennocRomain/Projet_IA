import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json


def charger_donnees(chemin_fichier):
    return pd.read_csv(chemin_fichier)

def preparer_donnees(data, features):
    data = data[features].copy()

    # Exemple d'encodage des variables catégorielles si nécessaire
    # Exemple d'utilisation de LabelEncoder ou OneHotEncoder si nécessaire

    return data

def entrainer_modele(X_train, y_train, method='random_forest'):
    """
    Entraîne un modèle de classification supervisée.
    """
    if method == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Ajouter d'autres méthodes de classification comme SVM, réseaux de neurones, etc.

    model.fit(X_train, y_train)
    return model

def predire_et_generer_json(model, X_test, features):
    """
    Fait des prédictions avec le modèle et génère une sortie JSON.
    """
    y_pred = model.predict(X_test)

    arbres_susceptibles = []
    for i, prediction in enumerate(y_pred):
        if prediction == 1:  # Adapté selon votre modèle et vos données
            arbre = {
                'id': i + 1,
                'features': {feat: X_test.iloc[i][feat] for feat in features}
            }
            arbres_susceptibles.append(arbre)

    # Générer la sortie JSON
    output_json = json.dumps(arbres_susceptibles, indent=4)
    return output_json

def main():
    # Paramètres d'entrée
    chemin_fichier = "Data_Arbre.csv"
    features = sys.argv[1:-1]
    method = sys.argv[-1]

    # Charger les données
    data = charger_donnees(chemin_fichier)

    # Préparer les données
    X = preparer_donnees(data, features)
    y = data['target_column']  # Remplacez 'target_column' par votre colonne cible

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model = entrainer_modele(X_train, y_train, method)

    # Faire des prédictions sur l'ensemble de test
    y_pred = predire_et_generer_json(model, X_test, features)

    # Générer la sortie au format JSON
    output = {
        "predictions": y_pred.tolist()
    }

    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    main()