import pandas as pd
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score


data = pd.read_csv("Data_Arbre.csv")

#=================================================================
#=================== Préparation des données =====================
#=================================================================

# Mettre les noms tech en majuscules
data['fk_nomtech'] = data['fk_nomtech'].str.upper()
# Mettre les fk_stadedev en minuscules
data['fk_stadedev'] = data['fk_stadedev'].str.lower()

# Transformer données texte en données numériques
    # fk_stadedev
enc = OrdinalEncoder()
train_cat = data[['fk_stadedev']]
c = enc.fit_transform(train_cat)
data[['fk_stadedev']] = c

    # fk_nomtech
enc2 = OrdinalEncoder()
train_cat2 = data[['fk_nomtech']]
c2 = enc2.fit_transform(train_cat2)
data[['fk_nomtech']] = c2

# Définir des intervalles d'âge dans une nouvelle colonne 'age_group'
bins = [0, 10, 20, 30, 40, 50, 100, 200]
labels = [0, 1, 2, 3, 4, 5, 6]
data['age_group'] = pd.cut(data['age_estim'], bins=bins, labels=labels, right=True)
data = data.dropna()  # supprimer les lignes NaN

# Séparation des features X et des labels y
X = data[['haut_tot', 'haut_tronc', 'tronc_diam', 'fk_stadedev', 'fk_nomtech']]
y = data['age_group']

# Répartition des données : 80% apprentissage, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Afficher les associations valeurs numériques / valeurs textuelles
#for i, category in enumerate(enc.categories_[0]):
    #print(f"  {category} -> {i}")

#for i, category in enumerate(enc2.categories_[0]):
    #print(f"  {category} -> {i}")



#=================================================================
#=================== Apprentissage supervisé =====================
#=================================================================

#============= Stochastic Gradient Descent (SGD) =================

classifier = SGDClassifier()
classifier.fit(X_train, y_train)


#=======================================================================
#=================== Evaluation via des métriques  =====================
#=======================================================================


#=================== Taux de classification ============================

y_pred = classifier.predict(X_test)

# Tableau des taux de classification
folds = cross_val_score(classifier, X_test, y_test, scoring="accuracy", cv=5)
print("Taux de classification : ", folds)

# moyenne des taux de classification
print("Moyenne des taux : ", mean(folds))

# taux de classification avec normalisation
#folds2 = cross_val_score(classifier, test_data, test_labels, scoring="accuracy", cv=5)
#print(folds2)
#print(mean(folds2))


#=================== Matrice de confusion ==============================

mc = confusion_matrix(y_test, y_pred)

# afficher matrice avec des pourcentages
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', values_format=".0%")
plt.title('Matrice de Confusion')
plt.show()



#=================== Précision, rappel, f1 score ===================================

# précision de chaque classe
precision = precision_score(y_test, y_pred, average=None)
print("Précision : ", precision)

# rappel de chaque classe
rappel = recall_score(y_test, y_pred, average=None)
print("Rappel : ", rappel)

# f1 score de chaque classe
f1score = f1_score(y_test, y_pred, average=None)
print("F1 Score : ", f1score)

#=================== Optimisation des paramètres ==================================

param_grid = {
    'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],   # minimiser fonction de perte
    'penalty': ['l2', 'l1', 'elasticnet'],  # éviter le surapprentissage
}

scv = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")
scv.fit(X_train, y_train)

print("Meilleurs paramètres : ", scv.best_params_)

