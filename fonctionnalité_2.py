import pandas as pd
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree


data = pd.read_csv("export_IA.csv")

#=================================================================
#=================== Préparation des données =====================
#=================================================================

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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#=================================================================
#=================== Apprentissage supervisé =====================
#=================================================================

#============= Stochastic Gradient Descent (SGD) =================

classifier = SGDClassifier(loss='log_loss', penalty='l2')
classifier.fit(X_train, y_train)

#============== k plus proche voisin =============================
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

#=================== SVM ===================
svm = svm.SVC()
svm.fit(X_train, y_train)

#========================== arbre de decision ================
tree = tree.DecisionTreeClassifier()
tree.fit(X_train, y_train)


#=======================================================================
#=================== Evaluation via des métriques  =====================
#=======================================================================


#=================== Taux de classification ============================

#y_pred = classifier.predict(X_test)
#y_pred = neigh.predict(X_test)
#y_pred = svm.predict(X_test)
y_pred = tree.predict(X_test)

# Tableau des taux de classification
#folds = cross_val_score(classifier, X_test, y_test, scoring="accuracy", cv=5)
#folds = cross_val_score(neigh, X_test, y_test, scoring="accuracy", cv=5)
#folds = cross_val_score(svm, X_test, y_test, scoring="accuracy", cv=5)
folds = cross_val_score(tree, X_test, y_test, scoring="accuracy", cv=5)

print("Taux de classification : ", folds)

# moyenne des taux de classification
print("Moyenne des taux : ", mean(folds))


#=================== Matrice de confusion ==============================

mc = confusion_matrix(y_test, y_pred)

# afficher matrice avec des pourcentages
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', values_format=".0%")
plt.title('Matrice de Confusion')
plt.show()


#=================== Précision, rappel, f1 score ===================================

# précision de chaque classe
precision = precision_score(y_test, y_pred, average=None, zero_division=1)
print("Précision : ", precision)

# rappel de chaque classe
rappel = recall_score(y_test, y_pred, average=None, zero_division=1)
print("Rappel : ", rappel)


#=================== Optimisation des paramètres ==================================

param_grid = {
    'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],   # minimiser fonction de perte
    'penalty': ['l2', 'l1', 'elasticnet']   # éviter surapprentissage
}

#scv = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")
#scv.fit(X_train, y_train)

#print("Meilleurs paramètres : ", scv.best_params_)

# Afficher les résultats d'optimisation
"""
res = pd.DataFrame(scv.cv_results_)
# Sélectionner les colonnes pertinentes pour le tableau
results = res[['param_loss', 'param_penalty', 'mean_test_score', 'std_test_score', 'rank_test_score']]
# Préparer les données pour le tableau
table_data = results.values
# Définir les en-têtes du tableau
columns = results.columns.tolist()
# Créer la figure et les axes
fig, ax = plt.subplots(figsize=(12, 8))  # Taille de la figure
# Cacher les axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)
# Créer le tableau
table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
# Styliser le tableau
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(columns))))
# Afficher la figure
plt.title('Résultats de GridSearchCV')
plt.show()
"""

