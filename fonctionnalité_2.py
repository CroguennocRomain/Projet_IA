import pandas as pd
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, scale
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
import pickle
from sklearn.decomposition import PCA
import seaborn as sns



#data = pd.read_csv("export_IA.csv")
data = pd.read_csv("Data_Arbre.csv")

# -----------------------------
# |   PREPARATION DONNEES     |
# -----------------------------

# Transformer données catégorielles en numériques
encoder = OrdinalEncoder()
for colonne in data:
    if data[colonne].dtype.name == 'object':
        data[colonne] = encoder.fit_transform(data[[colonne]])



# Sauvegarde de l'encodeur
with open('OrdinalEncoder/ordinal_encoder2.pkl', 'wb') as f:
    pickle.dump(encoder, f)


# Définir des intervalles d'âge dans une nouvelle colonne 'age_group'
bins = [-1, 10, 20, 30, 40, 50, 100, 200]
labels = [0, 1, 2, 3, 4, 5, 6]
data['age_group'] = pd.cut(data['age_estim'], bins=bins, labels=labels, right=True)
data = data.dropna()  # supprimer les lignes NaN
data['age_group'] = data['age_group'].astype(int)

# Normalisation
Y = data['age_group']

scaler = StandardScaler()
data_norm = scaler.fit_transform(data)
with open('Scaler/scaler2.pkl', 'wb') as f:
    pickle.dump(scaler, f)

data_norm = pd.DataFrame(data_norm, columns=data.columns)
data_norm['age_group'] = Y


# Séparation des features X et des labels y
X = data_norm[['haut_tot', 'haut_tronc', 'tronc_diam', 'fk_stadedev', 'fk_nomtech']]
y = data_norm['age_group']

# Répartition des données : 80% apprentissage, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
# Faire en sorte que les données aient une moyenne de 0 et une variance de 1
"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
with open('normalisation.pkl', 'wb') as f:
    pickle.dump(scaler, f)
"""

# -----------------------------
# |          MODELES          |
# -----------------------------
#============= Stochastic Gradient Descent (SGD) =================

sgd = SGDClassifier(loss='log_loss', penalty='l1', tol=1e-05)
#sgd = SGDClassifier()
sgd.fit(X_train, y_train)

with open("models/age_SGD.pkl", "wb") as f:
    pickle.dump(sgd, f)

#============== k plus proche voisin =============================
neigh = KNeighborsClassifier(algorithm='brute', n_neighbors=7, p=1, weights='distance')
#neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)

with open("models/age_neigh.pkl", "wb") as f:
    pickle.dump(neigh, f)

#=================== SVM ===================
svm = svm.SVC(C=100, kernel='rbf', probability=True)
#svm = svm.SVC()
svm.fit(X_train, y_train)

with open("models/age_SVM.pkl", "wb") as f:
    pickle.dump(svm, f)

#========================== arbre de decision ================
tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=40, min_samples_leaf=1, min_samples_split=2, splitter='best')
#tree = tree.DecisionTreeClassifier()
tree.fit(X_train, y_train)

with open("models/age_tree.pkl", "wb") as f:
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
score_sgd = accuracy_score(y_test, y_pred_sgd)
score_neigh = accuracy_score(y_test, y_pred_neigh)
score_svm = accuracy_score(y_test, y_pred_svm)
score_tree = accuracy_score(y_test, y_pred_tree)

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
    'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'tol': [1e-3, 1e-4, 1e-5]
}

grid_sgd = GridSearchCV(estimator=sgd, param_grid=param_sgd, cv=5, scoring="accuracy")
grid_sgd.fit(X_train, y_train)

print("Meilleurs paramètres - SGD : ", grid_sgd.best_params_)
res_sgd = pd.DataFrame(grid_sgd.cv_results_)
res_sgd.to_csv('GridSearchResults/grid_search_res_sgd.csv', index=False)
"""

# modèle SVM
"""
param_svm = {
    'C': [10, 100],
    'kernel': ['linear', 'rbf', 'sigmoid']
}
grid_svm = GridSearchCV(estimator=svm, param_grid=param_svm, cv=3, scoring="accuracy")
grid_svm.fit(X_train, y_train)

print("Meilleurs paramètres - SVM : ", grid_svm.best_params_)
res_svm = pd.DataFrame(grid_svm.cv_results_)
res_svm.to_csv('GridSearchResults/grid_search_res_svm.csv', index=False)
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
res_neigh = pd.DataFrame(grid_neigh.cv_results_)
res_neigh.to_csv('GridSearchResults/grid_search_res_neigh.csv', index=False)
"""

# modèle arbre de décision
"""
param_tree = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_tree = GridSearchCV(estimator=tree, param_grid=param_tree, cv=5, scoring="accuracy")
grid_tree.fit(X_train, y_train)

print("Meilleurs paramètres - DecisionTree : ", grid_tree.best_params_)
res_tree = pd.DataFrame(grid_tree.cv_results_)
res_tree.to_csv('GridSearchResults/grid_search_res_tree.csv', index=False)
"""

