import pandas as pd
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
import pickle



data = pd.read_csv("export_IA.csv")

# -----------------------------
# |   PREPARATION DONNEES     |
# -----------------------------

# Définir des intervalles d'âge dans une nouvelle colonne 'age_group'
bins = [0, 10, 20, 30, 40, 50, 100, 200]
labels = [0, 1, 2, 3, 4, 5, 6]
data['age_group'] = pd.cut(data['age_estim'], bins=bins, labels=labels, right=True)
data = data.dropna()  # supprimer les lignes NaN

# Séparation des features X et des labels y
X = data[['haut_tot', 'haut_tronc', 'tronc_diam', 'fk_stadedev', 'nomfrancais']]
y = data['age_group']

# Répartition des données : 80% apprentissage, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
# Faire en sorte que les données aient une moyenne de 0 et une variance de 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# |          MODELES          |
# -----------------------------
#============= Stochastic Gradient Descent (SGD) =================

sgd = SGDClassifier(loss='log_loss', penalty='elasticnet')
sgd.fit(X_train, y_train)

with open("age_SGD.pkl", "wb") as f:
    pickle.dump(sgd, f)

#============== k plus proche voisin =============================
neigh = KNeighborsClassifier(algorithm='brute', n_neighbors=7, p=1, weights='distance')
neigh.fit(X_train, y_train)

with open("age_neigh.pkl", "wb") as f:
    pickle.dump(neigh, f)

#=================== SVM ===================
svm = svm.SVC(C=100, kernel='rbf', probability=True)
svm.fit(X_train, y_train)

with open("age_SVM.pkl", "wb") as f:
    pickle.dump(svm, f)

#========================== arbre de decision ================
tree = tree.DecisionTreeClassifier()
tree.fit(X_train, y_train)

with open("age_tree.pkl", "wb") as f:
    pickle.dump(tree, f)


# -----------------------------
# |           METRIQUES       |
# -----------------------------

# Prédictions
y_pred_sgd = sgd.predict(X_test)
y_pred_neigh = neigh.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_tree = tree.predict(X_test)

# Taux de classification
score_sgd = sgd.score(X_test, y_test)
score_neigh = neigh.score(X_test, y_test)
score_svm = svm.score(X_test, y_test)
score_tree = tree.score(X_test, y_test)

# Affichage scores
print("Taux SGD : ", score_sgd)
print("Taux SVM : ", score_svm)
print("Taux K-neighbor : ", score_neigh)
print("Taux DecisionTree : ", score_tree)


# ------------------------------
# |   MATRICE DE CONFUSION     |
# ------------------------------

mc_sgd = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_sgd, normalize='true', values_format=".0%")
plt.title('Matrice de Confusion - SGD')
plt.show()

mc_svm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, normalize='true', values_format=".0%")
plt.title('Matrice de Confusion - SVM')
plt.show()

mc_neigh = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_neigh, normalize='true', values_format=".0%")
plt.title('Matrice de Confusion - K nearest neighbors')
plt.show()

mc_tree = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tree, normalize='true', values_format=".0%")
plt.title('Matrice de Confusion - Decision Tree')
plt.show()


# -----------------------------
# |   PRECISION ET RAPPEL     |
# -----------------------------

# précision de chaque classe
precision_sgd = precision_score(y_test, y_pred_sgd, average=None, zero_division=1)
precision_svm = precision_score(y_test, y_pred_svm, average=None, zero_division=1)
precision_neigh = precision_score(y_test, y_pred_neigh, average=None, zero_division=1)
precision_tree = precision_score(y_test, y_pred_tree, average=None, zero_division=1)

print("Précision SGD: ", precision_sgd)
print("Précision SVM: ", precision_svm)
print("Précision K nearest neighbors: ", precision_neigh)
print("Précision DecisionTree: ", precision_tree)

# rappel de chaque classe
rappel_sgd = recall_score(y_test, y_pred_sgd, average=None, zero_division=1)
rappel_svm = recall_score(y_test, y_pred_svm, average=None, zero_division=1)
rappel_neigh = recall_score(y_test, y_pred_neigh, average=None, zero_division=1)
rappel_tree = recall_score(y_test, y_pred_tree, average=None, zero_division=1)

print("Rappel SGD: ", rappel_sgd)
print("Rappel SVM: ", rappel_svm)
print("Rappel K nearest neighbors: ", rappel_neigh)
print("Rappel DecisionTree: ", rappel_tree)


# -----------------------------------
# |     OPTIMISATION PARAMETRES     |
# -----------------------------------

# modèle SGD
"""
param_sgd = {
    'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],   # minimiser fonction de perte
    'penalty': ['l2', 'l1', 'elasticnet']   # éviter surapprentissage
}

grid_sgd = GridSearchCV(estimator=sgd, param_grid=param_sgd, cv=5, scoring="accuracy")
grid_sgd.fit(X_train, y_train)

print("Meilleurs paramètres - SGD : ", grid_sgd.best_params_)
"""

# modèle SVM
"""
param_svm = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

grid_svm = GridSearchCV(estimator=svm, param_grid=param_svm, cv=5, scoring="accuracy")
grid_svm.fit(X_train, y_train)

print("Meilleurs paramètres - SVM : ", grid_svm.best_params_)
"""

# modèle k-Neighbors
"""
param_neigh = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

grid_neigh = GridSearchCV(estimator=neigh, param_grid=param_neigh, cv=5, scoring="accuracy")
grid_neigh.fit(X_train, y_train)

print("Meilleurs paramètres - KNeighbors : ", grid_neigh.best_params_)
"""