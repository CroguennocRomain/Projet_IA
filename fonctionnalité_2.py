import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Data_Arbre.csv")

#=================================================================
#=================== Préparation des données =====================
#=================================================================

# Suppression des colonnes sans intérêt
del data['longitude']
del data['latitude']
del data['clc_quartier']
del data['clc_secteur']
del data['clc_nbr_diag']
del data['villeca']
del data['fk_port']
del data['fk_pied']
del data['fk_revetement']
del data['fk_situation']
del data['fk_arb_etat']
del data['feuillage']
del data['remarquable']
del data['fk_prec_estim']



# Mettre les noms en majuscules
data['fk_nomtech'] = data['fk_nomtech'].str.upper()
# Mettre les fk_stadedev en minuscules
data['fk_stadedev'] = data['fk_stadedev'].str.lower()

# Définir différents intervalles d'âge dans une nouvelle colonne
bins = [0, 10, 30, 70, 100, 150, 200]
labels = [0, 1, 2, 3, 4, 5]
data['age_group'] = pd.cut(data['age_estim'], bins=bins, labels=labels, right=True)
data = data.dropna()  # supprimer les lignes NaN
del data['age_estim']

# Répartition des données : 80% apprentissage, 20% test
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


# Répartition des données : séparer features et classes
train_data = train_set.drop("age_group", axis=1)    # features des données d'apprentissage
train_labels = train_set["age_group"].copy()    # classes des données d'apprentissage


# Transformer données texte en données numériques
from sklearn.preprocessing import OrdinalEncoder

# fk_stadedev
enc = OrdinalEncoder()
train_cat = train_data[["fk_stadedev"]]
c = enc.fit_transform(train_cat)
train_data[["fk_stadedev"]] = c

# fk_nomtech
enc2 = OrdinalEncoder()
train_cat2 = train_data[["fk_nomtech"]]
c2 = enc2.fit_transform(train_cat2)
train_data[["fk_nomtech"]] = c2


# Afficher les associations valeurs numériques / valeurs textuelles
#for i, category in enumerate(enc.categories_[0]):
    #print(f"  {category} -> {i}")

#for i, category in enumerate(enc2.categories_[0]):
    #print(f"  {category} -> {i}")




#=================================================================
#=================== Apprentissage supervisé =====================
#=================================================================

# Stochastic Gradient Descent (SGD)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

classifier = SGDClassifier()
classifier.fit(train_data, train_labels)

# prédiction de la premiere instance
pred1 = classifier.predict([train_data.values[0]])
#print(pred1)

# tableau des scores de décision liée à la prédiction
scores_decision = classifier.decision_function([train_data.values[0]])
#print(scores_decision)
# indice de la valeur max du tableau
ind_max = np.argmax(scores_decision)
#print(ind_max)


#=======================================================================
#=================== Evaluation via des métriques  =====================
#=======================================================================

from sklearn.model_selection import cross_val_score

# Taux de classification

folds = cross_val_score(classifier, train_data, train_labels, scoring="accuracy", cv=10)
#print(folds)

from numpy import mean
# moyenne des taux de classification
#print(mean(folds))

# Normalisation des données
#from sklearn.preprocessing import StandardScaler
#scale = StandardScaler()
#scale.fit_transform(train_data, train_labels)

# taux de classification avec normalisation
folds2 = cross_val_score(classifier, train_data, train_labels, scoring="accuracy", cv=10)
#print(folds2)
#print(mean(folds2))


# Matrice de confusion

from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(classifier, train_data, train_labels, cv=10)
print(predictions)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
mc = confusion_matrix(train_labels, predictions)

#disp = ConfusionMatrixDisplay(mc)   # matrice de confusion selon le nombre de prédictions
disp = ConfusionMatrixDisplay.from_predictions(train_labels, predictions, normalize='true', values_format=".0%")    # matrice de confusion selon les pourcentages
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.show()


# Précision et rappel