# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                 IMPORTATIONS                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score,classification_report, roc_curve, auc
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
pd.set_option('display.max_columns', None)

data = pd.read_csv("Data_Arbre.csv")

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

#data.info(max)
# X ET Y POUR RANDOM FOREST
y = data['fk_arb_etat']
X_rf = data[["haut_tronc","latitude","longitude",'fk_stadedev','haut_tot','clc_secteur']]
# X ET Y POUR KNN
X_knn = data[["latitude","longitude","clc_secteur",'fk_port']]


# X ET Y POUR SVM
X_svm = data[['age_estim']]

scaler = StandardScaler()

X_rf = scaler.fit_transform(X_rf)
X_knn = scaler.fit_transform(X_knn)
X_svm = scaler.fit_transform(X_svm)

with open('models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y, test_size=0.2, random_state=42)
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y, test_size=0.2, random_state=42)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y, test_size=0.2, random_state=42)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                APPRENTISSAGE 1                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

param_grid = {
    'n_estimators': [50, 100, 150],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configuration et exécution de GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train_rf, y_train_rf)

# Affichage des meilleurs paramètres et des scores
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

# Entraînement du modèle avec les meilleurs paramètres
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_rf, y_train_rf)

# Prédiction sur l'ensemble de test
y_pred_rf = best_rf.predict(X_test_rf)

"""# Évaluation des performances
accuracy = accuracy_score(y_test_rf, y_pred_rf)
print(f'Accuracy : {accuracy:.2f}')
report = classification_report(y_test_rf, y_pred_rf)
print(report)"""

with open('models/rf_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

"""model_1 = RandomForestClassifier(n_estimators=1000, random_state=42)
model_1.fit(X_train_rf, y_train_rf)

with open('models/rf_model.pkl', 'wb') as file:
    pickle.dump(model_1, file)

#with open('models/rf_model.pkl', 'rb') as file:
#        model_1 = pickle.load(file)
y_pred_rf = model_1.predict(X_test_rf)"""


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                APPRENTISSAGE 2                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, scoring='accuracy', cv=5, n_jobs=-1)
grid_search_knn.fit(X_train_knn, y_train_knn)
print(f"KNN Best parameters found: {grid_search_knn.best_params_}")
print(f"KNN Best cross-validation score: {grid_search_knn.best_score_:.2f}")
best_knn = grid_search_knn.best_estimator_
best_knn.fit(X_train_knn, y_train_knn)
y_pred_knn = best_knn.predict(X_test_knn)
with open('models/knn_model.pkl', 'wb') as file:
    pickle.dump(best_knn, file)


"""model_2 = KNeighborsClassifier(n_neighbors=5)
model_2.fit(X_train_knn, y_train_knn)
with open('models/knn_model.pkl', 'wb') as file:
    pickle.dump(model_2, file)

with open('models/knn_model.pkl', 'rb') as file:
    model_2 = pickle.load(file)
y_pred_knn = model_2.predict(X_test_knn)"""


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                APPRENTISSAGE 3                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale','auto']
}
svm = SVC(probability=True)
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, scoring='accuracy', cv=5, n_jobs=-1)
grid_search_svm.fit(X_train_svm, y_train_svm)
print(f"SVM Best parameters found: {grid_search_svm.best_params_}")
print(f"SVM Best cross-validation score: {grid_search_svm.best_score_:.2f}")
best_svm = grid_search_svm.best_estimator_
best_svm.fit(X_train_svm, y_train_svm)
y_pred_svm = best_svm.predict(X_test_svm)
with open('models/svm_model.pkl', 'wb') as file:
    pickle.dump(best_svm, file)
"""model_3 = SVC(probability=True, random_state=42)
model_3.fit(X_train_svm, y_train_svm)

with open('models/svm_model.pkl', 'wb') as file:
    pickle.dump(model_3, file)

with open('models/svm_model.pkl', 'rb') as file:
    model_3 = pickle.load(file)
y_pred_svm = model_3.predict(X_test_svm)"""
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃               ANALYSE RESULTATS               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


#------------------- Rapport de classification------------------
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




score_1 = best_rf.score(X_test_rf, y_test_rf)
score_2 = best_knn.score(X_test_knn, y_test_knn)
score_3 = best_svm.score(X_test_svm, y_test_svm)
print('score 1 :', score_1)
print('score 2 :', score_2)
print('score 3 :', score_3)


print('----------------')
print(r2_score(y_test_rf, y_pred_rf))
print(r2_score(y_test_knn, y_pred_knn))
print(r2_score(y_test_svm, y_pred_svm))

#---------------- Matrice de confusion----------------------
conf_matrix = confusion_matrix(y_test_rf, y_pred_rf)
print(conf_matrix)

# Affichage de la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.title('Matrice de Confusion')
plt.show()

conf_matrix = confusion_matrix(y_test_knn, y_pred_knn)
print(conf_matrix)

# Affichage de la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.title('Matrice de Confusion')
plt.show()

conf_matrix = confusion_matrix(y_test_svm, y_pred_svm)
print(conf_matrix)

# Affichage de la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.title('Matrice de Confusion')
plt.show()
y_pred_proba_rf = best_rf.predict_proba(X_test_rf)[:, 1]
y_pred_proba_knn = best_knn.predict_proba(X_test_knn)[:, 1]
y_pred_proba_svm = best_svm.predict_proba(X_test_svm)[:, 1]
#------------------------Courbe ROC---------------------------
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