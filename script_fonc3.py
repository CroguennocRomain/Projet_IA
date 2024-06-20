import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def predire_tempete(method):
    pd.set_option('future.no_silent_downcasting', True)
    data = pd.read_csv('Data_Arbre.csv')

    if method == '0' and len(sys.argv) == 8:
        new_data = {
            'haut_tronc': [float(sys.argv[1])],
            'latitude': [float(sys.argv[2])],
            'longitude': [float(sys.argv[3])],
            'fk_stadedev': [sys.argv[4]],
            'haut_tot': [float(sys.argv[5])],
            'clc_secteur': [sys.argv[6]]
        }
        new_data_df = pd.DataFrame(new_data)


        for colonne in data.columns:
            if colonne not in new_data_df.columns:
                new_data_df[colonne] = data[colonne][0]

        new_data_df = new_data_df[data.columns]

        # Sélectionner les colonnes catégorielles de la nouvelle ligne de données
        categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']
        with open('OrdinalEncoder/ordinal_encoder3.pkl', 'rb') as file:
            encoder = pickle.load(file)

        # Appliquer l'encodeur sur les colonnes catégorielles de la nouvelle ligne de données
        new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

        data['fk_arb_etat'] = data['fk_arb_etat'].replace({
            'Essouché': 1,
            'EN PLACE': 0,
            'SUPPRIMÉ': 0,
            'Non essouché': 1,
            'REMPLACÉ': 0,
            'ABATTU': 0
        })



        X = new_data_df[["haut_tronc","latitude","longitude",'fk_stadedev','haut_tot','clc_secteur']]

        with open('Scaler/scaler3.pkl', 'rb') as file:
            model = pickle.load(file)
        X = model.fit_transform(X)
        print(X)
        model_filename = 'models/rf_model.pkl'
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

        # Sélectionner les colonnes catégorielles de la nouvelle ligne de données
        categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']

        with open('OrdinalEncoder/ordinal_encoder3.pkl', 'rb') as file:
            encoder = pickle.load(file)
        # Appliquer l'encodeur sur les colonnes catégorielles de la nouvelle ligne de données
        new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

        data['fk_arb_etat'] = data['fk_arb_etat'].replace({
            'Essouché': 1,
            'EN PLACE': 0,
            'SUPPRIMÉ': 0,
            'Non essouché': 1,
            'REMPLACÉ': 0,
            'ABATTU': 0
        })
        X = new_data_df[["latitude","longitude","clc_secteur",'fk_port']]

        with open('Scaler/scaler3.pkl', 'rb') as file:
            model = pickle.load(file)
        X = model.fit_transform(X)

        model_filename = 'models/knn_model.pkl'
    elif method == '2' and len(sys.argv) == 3:
        new_data = {
            'age_estim': [float(sys.argv[1])]
        }
        new_data_df = pd.DataFrame(new_data)
        for colonne in data.columns:
            if colonne not in new_data_df.columns:
                new_data_df[colonne] = data[colonne][0]
        new_data_df = new_data_df[data.columns]

        # Sélectionner les colonnes catégorielles de la nouvelle ligne de données
        categorical_columns = [colonne for colonne in new_data_df if new_data_df[colonne].dtype == 'object']

        with open('OrdinalEncoder/ordinal_encoder3.pkl', 'rb') as file:
            encoder = pickle.load(file)
        # Appliquer l'encodeur sur les colonnes catégorielles de la nouvelle ligne de données
        new_data_df[categorical_columns] = encoder.transform(new_data_df[categorical_columns])

        data['fk_arb_etat'] = data['fk_arb_etat'].replace({
            'Essouché': 1,
            'EN PLACE': 0,
            'SUPPRIMÉ': 0,
            'Non essouché': 1,
            'REMPLACÉ': 0,
            'ABATTU': 0
        })
        X = new_data_df[['age_estim']]
        with open('Scaler/scaler3.pkl', 'rb') as file:
            model = pickle.load(file)
        X = model.fit_transform(X)

        model_filename = 'models/svm_model.pkl'
    else:
        raise ValueError("Invalid method or number of arguments")

    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X)

    res = y_pred.tolist()
    json_data = json.dumps(res)

    return json_data


def main():
    method = sys.argv[-1]
    tempete = predire_tempete(method)
    print(tempete)


if __name__ == "__main__":
    main()