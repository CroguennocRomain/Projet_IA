# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                 IMPORTATIONS                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

import time

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns


import pandas as pd

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃            PREPARATION DES DONNEES            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
pd.set_option('display.max_columns', None)

df = pd.read_csv("Data_Arbre.csv")


colonnes = ["haut_tot","tronc_diam","fk_port","feuillage","fk_stadedev","fk_revetement","age_estim","fk_arb_etat"]
data = df[colonnes].copy()

label_encoder = LabelEncoder()
data['fk_revetement_encoded'] = label_encoder.fit_transform(data['fk_revetement'])
data['fk_port_encoded'] = label_encoder.fit_transform(data['fk_port'])
data['fk_stadedev_encoded'] = label_encoder.fit_transform(data['fk_stadedev'])
data['feuillage_encoded'] = label_encoder.fit_transform(data['feuillage'])

data['fk_arb_etat'] = data['fk_arb_etat'].replace({
    'Essouché': 1,
    'EN PLACE': 0,
    'SUPPRIMÉ': 0,
    'Non essouché': 0,
    'REMPLACÉ': 0,
    'ABATTU': 0
})

print(data)

# Séparation des caractéristiques (X) et de la cible (y)
X = data.drop(['fk_arb_etat', 'fk_port', 'fk_stadedev', 'feuillage', 'fk_revetement'], axis=1)
y = data['fk_arb_etat']
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                 APPRENTISSAGE                 ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle
rf_classifier.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = rf_classifier.predict(X_test)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃               ANALYSE RESULTATS               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


#------------------- Rapport de classification------------------
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy : {accuracy:.2f}')
report = classification_report(y_test, y_pred, zero_division=0)
print(report)

#---------------- Matrice de confusion----------------------
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Affichage de la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.title('Matrice de Confusion')
plt.show()

y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

#------------------------Courbe ROC---------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
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
"""
#-------------------------- GRID-SCV--------------------------
# Définir la grille des hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Créer l'objet GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Exécuter la recherche sur grille sur les données d'entraînement
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres et le meilleur score
print("Meilleurs paramètres trouvés :")
print(grid_search.best_params_)
print("Meilleur score d'entraînement :")
print(grid_search.best_score_)

# Prédire sur l'ensemble de test avec le meilleur modèle trouvé
y_pred = grid_search.best_estimator_.predict(X_test)

# Évaluer la précision du meilleur modèle sur l'ensemble de test
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision sur l'ensemble de test : {accuracy:.2f}")"""