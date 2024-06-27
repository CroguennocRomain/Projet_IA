# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                 IMPORTATIONS                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃            PREPARATION DES DONNEES            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Lecture des données
pd.set_option('display.max_columns', None)
data = pd.read_csv("Data_Arbre.csv")

# Remplacement des valeurs de 'fk_arb_etat' par des valeurs binaires
data['fk_arb_etat'] = data['fk_arb_etat'].replace({
    'Essouché': 1,
    'EN PLACE': 0,
    'SUPPRIMÉ': 0,
    'Non essouché': 1,
    'REMPLACÉ': 0,
    'ABATTU': 0
})

# Sélectionner les colonnes
categorical_columns = [colonne for colonne in data if data[colonne].dtype.name == 'object']
data.info(max)
# Créer et utiliser l'OrdinalEncoder
encoder = OrdinalEncoder()
data[categorical_columns] = encoder.fit_transform(data[categorical_columns])

# Définition de la variable cible
Y = data['fk_arb_etat']

# Normalisation des données
scaler = StandardScaler()
data_norm = scaler.fit_transform(data)
with open('Scaler/scaler3.pkl', 'wb') as f:
    pickle.dump(scaler, f)

data_norm = pd.DataFrame(data_norm, columns=data.columns)
data_norm['fk_arb_etat'] = Y

# Séparation des données en ensembles de caractéristiques pour chaque modèle
# X ET Y POUR RANDOM FOREST
y = data_norm['fk_arb_etat']
X_rf = data_norm[["haut_tronc","latitude","longitude",'fk_stadedev','haut_tot','clc_secteur']]

# X ET Y POUR KNN
X_knn = data_norm[["latitude","longitude","clc_secteur",'fk_port']]

# X ET Y POUR SVM
X_svm = data_norm[['haut_tot','fk_revetement']]

# Division des données en ensembles d'entraînement et de test
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y, test_size=0.2, random_state=42)
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y, test_size=0.2, random_state=42)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y, test_size=0.2, random_state=42)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                APPRENTISSAGE 1                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Configuration des hyperparamètres pour Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV pour trouver les meilleurs hyperparamètres
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train_rf, y_train_rf)

# Affichage des meilleurs paramètres et du score de validation croisée
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

# Entraînement du modèle avec les meilleurs paramètres
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_rf, y_train_rf)

# Prédiction sur l'ensemble de test
y_pred_rf = best_rf.predict(X_test_rf)

# Sauvegarde du modèle entraîné
with open('models/rf_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                APPRENTISSAGE 2                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Configuration des hyperparamètres pour KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# GridSearchCV pour trouver les meilleurs hyperparamètres
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, scoring='accuracy', cv=5, n_jobs=-1)
grid_search_knn.fit(X_train_knn, y_train_knn)

# Affichage des meilleurs paramètres et du score de validation croisée
print(f"KNN Best parameters found: {grid_search_knn.best_params_}")
print(f"KNN Best cross-validation score: {grid_search_knn.best_score_:.2f}")

# Entraînement du modèle avec les meilleurs paramètres
best_knn = grid_search_knn.best_estimator_
best_knn.fit(X_train_knn, y_train_knn)

# Prédiction sur l'ensemble de test
y_pred_knn = best_knn.predict(X_test_knn)

# Sauvegarde du modèle entraîné
with open('models/knn_model.pkl', 'wb') as file:
    pickle.dump(best_knn, file)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                APPRENTISSAGE 3                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Configuration des hyperparamètres pour SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale','auto']
}

# GridSearchCV pour trouver les meilleurs hyperparamètres
svm = SVC(probability=True,random_state=42)
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, scoring='accuracy', cv=5, n_jobs=-1)
grid_search_svm.fit(X_train_svm, y_train_svm)

# Affichage des meilleurs paramètres et du score de validation croisée
print(f"SVM Best parameters found: {grid_search_svm.best_params_}")
print(f"SVM Best cross-validation score: {grid_search_svm.best_score_:.2f}")

# Entraînement du modèle avec les meilleurs paramètres
best_svm = grid_search_svm.best_estimator_
best_svm.fit(X_train_svm, y_train_svm)

# Prédiction sur l'ensemble de test
y_pred_svm = best_svm.predict(X_test_svm)

# Sauvegarde du modèle entraîné
with open('models/svm_model.pkl', 'wb') as file:
    pickle.dump(best_svm, file)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃               ANALYSE RESULTATS               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Calcul et affichage des scores de précision et des rapports de classification
accuracy = accuracy_score(y_test_rf, y_pred_rf)
print(f'Accuracy RF : {accuracy:.2f}')
report = classification_report(y_test_rf, y_pred_rf, zero_division=0)
print(report)

accuracy = accuracy_score(y_test_knn, y_pred_knn)
print(f'Accuracy KNN : {accuracy:.2f}')
report = classification_report(y_test_knn, y_pred_knn, zero_division=0)
print(report)

accuracy = accuracy_score(y_test_svm, y_pred_svm)
print(f'Accuracy SVM : {accuracy:.2f}')
report = classification_report(y_test_svm, y_pred_svm, zero_division=0)
print(report)

# Affichage de la matrice de confusion pour Random Forest
conf_matrix = confusion_matrix(y_test_rf, y_pred_rf)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.title('Matrice de Confusion')
plt.show()

# Affichage de la matrice de confusion pour KNN
conf_matrix = confusion_matrix(y_test_knn, y_pred_knn)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.title('Matrice de Confusion')
plt.show()

# Affichage de la matrice de confusion pour SVM
conf_matrix = confusion_matrix(y_test_svm, y_pred_svm)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.title('Matrice de Confusion')
plt.show()

# Calcul des probabilités prédictives pour Random Forest, KNN et SVM
y_pred_proba_rf = best_rf.predict_proba(X_test_rf)[:, 1]
y_pred_proba_knn = best_knn.predict_proba(X_test_knn)[:, 1]
y_pred_proba_svm = best_svm.predict_proba(X_test_svm)[:, 1]

# Courbe ROC pour Random Forest
fpr, tpr, thresholds = roc_curve(y_test_rf, y_pred_proba_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()

# Courbe ROC pour KNN
fpr, tpr, thresholds = roc_curve(y_test_knn, y_pred_proba_knn)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()

# Courbe ROC pour SVM
fpr, tpr, thresholds = roc_curve(y_test_svm, y_pred_proba_svm)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()
