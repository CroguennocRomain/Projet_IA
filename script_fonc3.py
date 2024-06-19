import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import json

def predire_tempete(method, features, arbre):
    df = pd.read_csv("Data_Arbre.csv")
    pd.set_option('future.no_silent_downcasting', True)

    colonnes = ["latitude", "longitude", "haut_tot", "haut_tronc", "fk_stadedev", "fk_revetement","fk_arb_etat", "fk_situation"]
    data = df[colonnes].copy()

    label_encoder = LabelEncoder()
    data['fk_revetement_encoded'] = label_encoder.fit_transform(data['fk_revetement'])
    data['fk_situation_encoded'] = label_encoder.fit_transform(data['fk_situation'])
    data['fk_stadedev_encoded'] = label_encoder.fit_transform(data['fk_stadedev'])
    data['fk_arb_etat'] = data['fk_arb_etat'].replace({
        'Essouché': 1,
        'EN PLACE': 0,
        'SUPPRIMÉ': 0,
        'Non essouché': 1,
        'REMPLACÉ': 0,
        'ABATTU': 0
    }).infer_objects(copy=False)
    X = data[features]
    Y = data['fk_arb_etat']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Sélection du modèle d'apprentissage
    if method == '0':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_filename = 'random_forest_model.pkl'
    elif method == '1':
        model = KNeighborsClassifier(n_neighbors=5)
        model_filename = 'knn_model.pkl'
    elif method == '2':
        model = SVC(probability=True, random_state=42)
        model_filename = 'svm_model.pkl'

    # Entraînement et sauvegarde du modèle
    model.fit(X_train, y_train)
    with open(model_filename, 'rb') as file:
        #pickle.dump(model, file)
        model = pickle.load(file)

    y_pred = model.predict(X_test)

    res = y_pred.tolist()
    json_data = json.dumps(res[arbre])

    return json_data


def main():
    features = sys.argv[1:-2]
    method = sys.argv[-2]
    arbre = sys.argv[-1]
    arbre = int(arbre)


    tempete = predire_tempete(method, features,arbre)
    print(tempete)


if __name__ == "__main__":
    main()