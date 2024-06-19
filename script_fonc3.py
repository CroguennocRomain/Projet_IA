import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def predire_tempete(method, features, arbre):
    data = pd.read_csv("Data_Arbre.csv")
    pd.set_option('future.no_silent_downcasting', True)

    data['fk_arb_etat'] = data['fk_arb_etat'].replace({
        'Essouché': 1,
        'EN PLACE': 0,
        'SUPPRIMÉ': 0,
        'Non essouché': 1,
        'REMPLACÉ': 0,
        'ABATTU': 0
    })

    encoder = OrdinalEncoder()

    for colonne in data:
        if data[colonne].dtype.name == 'object':
            data[colonne] = encoder.fit_transform(data[[colonne]])

    X = data[features]
    Y = data['fk_arb_etat']


    with open('models/scaler.pkl', 'rb') as file:
        model = pickle.load(file)
        X = model.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Sélection du modèle d'apprentissage
    if method == '0':
        #model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_filename = 'models/rf_model.pkl'
    elif method == '1':
        #model = KNeighborsClassifier(n_neighbors=5)
        model_filename = 'models/knn_model.pkl'
    elif method == '2':
        #model = SVC(probability=True, random_state=42)
        model_filename = 'models/svm_model.pkl'

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